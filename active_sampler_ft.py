from my_parser import parser
import copy
from typing import Tuple

import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import trange
from utils import *
import pickle
from strategies import UncertaintySampler
from custom_dataset import *


def main():
    args = parser.parse_args()

    if args.dataset == "imagenet":
        args.num_images = 1281167
        args.num_classes = 1000

    elif args.dataset == "imagenet_lt":
        args.num_images = 115846

    elif args.dataset == "cifar100":
        args.num_images = 50000
        args.num_classes = 100

    elif args.dataset == "cifar10":
        args.num_images = 50000
        args.num_classes = 10

    else:
        raise NotImplementedError

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    main_worker(args, device)


class FirthRegularizer(torch.nn.Module):

    def __init__(self,
                 coeff: Optional[float] = 10 * torch.rand(1),
                 meta: bool = False,
                 dataloader: torch.utils.data.DataLoader = None,
                 device: torch.device = torch.device("cpu"),
                 max_size: int = None):
        super(FirthRegularizer, self).__init__()
        if not isinstance(coeff, torch.Tensor):
            coeff = torch.tensor(coeff)
        self.coeff = torch.nn.Parameter(coeff)
        self.meta = meta
        self.dataloader = dataloader
        self.device = device
        self.max_size = max_size

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        outputs = torch.tensor([]).to(self.device)
        for i, (images, _, _, _) in enumerate(self.dataloader):
            images = images.to(self.device)

            # compute output
            output = model(images)
            outputs = torch.cat([outputs, output], dim=0)
            if i == self.max_size:
                break

        n_samps_, n_ways_ = outputs.shape
        n_ways_ = torch.tensor(n_ways_).to(self.device)

        logp = torch.div(torch.sub(torch.logsumexp(outputs, dim=-1, keepdim=True), outputs), n_ways_)

        # negce = logp @ prior.reshape(n_ways_, 1)
        ceavg = logp.mean()
        if self.meta:
            return torch.mul(ceavg, self.coeff)
        return torch.mul(ceavg, self.coeff.detach())


def hyper_gradient(validation_loss: torch.Tensor,
                   training_loss: torch.Tensor,
                   lambda_: torch.Generator,
                   w: torch.Generator,
                   lr: float):
    # List[torch.Tensor]. v1[i].shape = w[i].shape
    v1 = torch.autograd.grad(validation_loss, w(), retain_graph=True)

    d_train_d_w = torch.autograd.grad(training_loss, w(), create_graph=True)
    # List[torch.Tensor]. v2[i].shape = w[i].shape
    v2 = approxInverseHVP(v1, d_train_d_w, w, alpha=lr)

    # List[torch.Tensor]. v3[i].shape = lambda_[i].shape
    v3 = torch.autograd.grad(d_train_d_w, lambda_(), grad_outputs=v2, retain_graph=True)

    # d_val_d_lambda = torch.autograd.grad(validation_loss, lambda_())
    return [-v for v in v3]


def approxInverseHVP(v: Tuple[torch.Tensor],
                     f: Tuple[torch.Tensor],
                     w: torch.Generator,
                     i=3, alpha=0.1):
    p = v

    for j in range(i):
        grad = torch.autograd.grad(f, w(), grad_outputs=v, retain_graph=True)
        v = [v_ - alpha * g for v_, g in zip(v, grad)]
        p = [p_ + v_ for p_, v_ in zip(p, v)]  # p += v (Typo in the arxiv version of the paper)

    return p


def compute_loss(outputs: torch.Tensor,
                 targets: torch.Tensor,
                 loss_firth: torch.nn.Module,
                 model: torch.nn.Module,
                 args):
    loss_ce = torch.nn.CrossEntropyLoss()(outputs, targets)
    if args.init_coeff == 0 and not args.meta:
        return None, loss_ce, 0
    loss_firth_out = loss_firth(model)
    loss_total = torch.add(loss_ce, loss_firth_out)

    return loss_total, loss_ce, loss_firth_out


def main_worker(args, device: torch.device):
    cudnn.benchmark = True
    all_indices = np.arange(args.num_images)
    inference_loader = get_inference_loader(args.dataset, all_indices, None, args)

    model = get_model(args.arch, args).to(device)

    # get all dataset labels in eval mode
    features, inference_labels = get_feats(args.dataset, inference_loader, model, device, args)
    firth_val_loader = get_train_loader(args.dataset, all_indices, features, args)

    # # validation data loading code
    test_loader = get_test_loader(args.dataset, args)

    # valid indices loading
    valid_idxs_file = '{}/random_{}_{}.npy'.format(args.indices, args.dataset, args.valid_size)
    if os.path.isfile(valid_idxs_file):
        print("=> Loading valid indices: {}".format(valid_idxs_file))
        valid_indices = np.load(valid_idxs_file)
        print('valid indices size: {}. {}% of all categories are seen'.format(len(valid_indices), len(np.unique(
            inference_labels[valid_indices])) / args.num_classes * 100))
    else:
        raise FileNotFoundError('valid indices file not found: {}'.format(valid_idxs_file))

    current_indices = np.array([])
    current_val_indices = np.array([])
    total_history = {}
    total_steps = args.total_budget_size // args.budget_size
    best_lambda = 0.0
    valid_budget_size = args.valid_size // total_steps
    for step in range(1, total_steps + 1):
        # Training data loading code
        unlabeled_indices = np.setdiff1d(all_indices, np.hstack([current_indices]))
        unqueried_val_indices = np.setdiff1d(valid_indices, np.hstack([current_val_indices]))

        print(f"Current unlabeled indices is {len(unlabeled_indices)} with {args.valid_size} valid samples.")
        if step == 1:
            current_indices_file = '{}/random_{}_{}.npy'.format(args.indices, args.dataset, args.budget_size)
            if os.path.isfile(current_indices_file):
                print("=> Loading first step training indices: {}".format(current_indices_file))
                current_indices = np.load(current_indices_file)
                print('training indices size: {}. {}% of all categories is seen'.format(
                    len(current_indices), len(np.unique(inference_labels[current_indices])) / args.num_classes * 100
                ))
            else:
                raise FileNotFoundError('first step training indices file not found: {}'.format(current_indices_file))
            shuffled_indices = np.random.choice(valid_indices, len(valid_indices), replace=False)
            current_val_indices = shuffled_indices[:valid_budget_size]
        else:
            print(f"Query sampling with {args.name} started ...")
            unlabeled_loader = get_inference_loader(args.dataset, unlabeled_indices, None, args)
            unqueried_val_loader = get_inference_loader(args.dataset, unqueried_val_indices, None, args)

            sampler = UncertaintySampler(unlabeled_loader, unlabeled_indices, model, device, args.budget_size)
            sampled_indices = sampler.sample(args.name)
            sampler = UncertaintySampler(unqueried_val_loader, unqueried_val_indices, model, device, valid_budget_size)
            sampled_val_indices = sampler.sample(args.name)

            current_indices = np.concatenate((current_indices, sampled_indices), axis=-1)
            current_val_indices = np.concatenate((current_val_indices, sampled_val_indices), axis=-1)

        print(f'{len(current_indices)} training, '
              f'{len(current_val_indices)} valid inidices are sampled in total, '
              f'{len(np.unique(current_indices))} of them are unique')

        print('{}% of all categories is seen'.format(
            len(np.unique(inference_labels[current_indices])) / args.num_classes * 100))
        train_loader = get_train_loader(args.dataset, current_indices, None, args)

        cudnn.benchmark = True
        print('Training task model started ...')
        best_val_acc = 0

        if args.accumulate_val:
            val_loader = get_inference_loader(args.dataset, current_val_indices, None, args)
        else:
            val_loader = get_inference_loader(args.dataset, valid_indices, None, args)

        if args.meta:
            model = get_model(args.arch, args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[50, 100], gamma=0.5
            )

            firth_meta_model = FirthRegularizer(args.init_coeff,
                                                meta=True,
                                                device=device,
                                                dataloader=firth_val_loader,
                                                max_size=30)
            meta_optimizer = torch.optim.RMSprop(firth_meta_model.parameters(), lr=args.meta_lr)

            best_lambda_list, history = meta_train(train_loader, val_loader, model, optimizer, lr_scheduler,
                                                   firth_meta_model, meta_optimizer, device, args)
            best_lambda = best_lambda_list[-1]
            test_acc1 = test(test_loader, model, device)
            history.update({'final_test_acc1': test_acc1})
            total_history[step] = history

        else:
            model = get_model(args.arch, args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[50, 100], gamma=0.5
            )

            firth_coeff = args.init_coeff

            loss_firth = FirthRegularizer(firth_coeff, device=device, dataloader=firth_val_loader, max_size=30)
            train_loader = sample_batch(train_loader)
            history = train(train_loader, val_loader, model, optimizer, lr_scheduler, loss_firth, device, args)

            val_acc1 = validate(val_loader, model, device)
            test_acc1 = test(test_loader, model, device)
            print(f'lambda {firth_coeff} results valid acc as: {val_acc1}')
            if val_acc1 > best_val_acc:
                best_val_acc = val_acc1
                best_lambda = firth_coeff
            history.update({'final_test_acc1': test_acc1})
            total_history[step] = history

        # lambda_res.append([best_lambda, best_val_acc])
        # test_acc1 = test(test_loader, model, device)

        print(f'Best lambda: {best_lambda} with top1: {best_val_acc} and test top1: {test_acc1}')
        # res.append([best_val_acc, test_acc1])

    if_meta = ''
    if_accu_val = ''
    if args.meta:
        if_meta = '_meta'
    if args.accumulate_val:
        if_accu_val = '_accuval'
    with open(f'res/AL_{args.name}_{args.dataset}_budget{args.budget_size}to{args.total_budget_size}'
              f'_ftall{if_meta}_lambda{args.init_coeff}{if_accu_val}_history_{args.filenumber}.pkl', 'wb') as f:
        pickle.dump(total_history, f)


def train(train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_firth: torch.nn.Module,
          device: torch.device,
          args,
          disable_pbar: bool = False):
    i = 1
    best_metric = 0
    best_step = 0
    patience = 0
    history = {}
    last_step_log = {}
    with trange(args.steps, desc="[TRAIN]", unit="steps", ncols=150,
                position=0, leave=True, disable=disable_pbar) as pbar:
        while i < args.steps + 1:
            model.train()
            images, target, _, _ = next(train_loader)

            images = images.to(device)
            target = target.to(device)

            outputs = model(images)

            loss, loss_ce, loss_firth_out = compute_loss(outputs, target, loss_firth, model, args)

            # compute gradient and do step
            optimizer.zero_grad()
            if args.init_coeff == 0:
                loss_ce.backward()
            else:
                loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))

            if i % args.valid_step == 0:
                val_metric = validate(valid_loader, model, device)
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_step = i
                    best_model = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    if patience >= args.patience:
                        print('Early stopping at step {}'.format(i))
                        break
                    patience += 1
                history[i] = {
                    'loss_CE': loss_ce.item(),
                    'loss_Firth': loss_firth_out.item() if args.init_coeff != 0 else 0,
                    'train_acc': acc1[0].item(),
                    'val_acc': val_metric,
                    'best_val_acc': best_metric,
                    'best_step': best_step
                }
                last_step_log.update(history[i])

            last_step_log['loss_CE'] = loss_ce.item()
            last_step_log['loss_Firth'] = loss_firth_out.item() if args.init_coeff != 0 else 0
            pbar.update()
            pbar.set_postfix(ordered_dict=last_step_log)
            i += 1
    model.load_state_dict(best_model)
    return history


def meta_train(train_loader: torch.utils.data.DataLoader,
               valid_loader: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler,
               firth_meta_model: torch.nn.Module,
               meta_optimizer: torch.optim.Optimizer,
               device: torch.device,
               args):
    best_metric = 0
    best_lambda_list = []
    patience = 0
    best_step = 0
    history = {}
    last_step_log_meta = {}
    epoch = 1

    train_loader_ = sample_batch(train_loader)
    valid_loader_ = sample_batch(valid_loader)

    with trange(args.nb_epoch, desc="[META TRAIN]", unit="steps", ncols=150, position=0, leave=True,
                disable=False) as pbar1:
        while epoch < args.nb_epoch + 1:
            model.train()

            images, target, _, _ = next(train_loader_)
            images = images.to(device)
            target = target.to(device)

            outputs = model(images)

            loss, loss_ce, loss_firth_out = compute_loss(outputs, target, firth_meta_model, model, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if epoch % args.steps == 0:
                model_copy = copy.deepcopy(model)
                train_sample_data, train_sample_target, _, _ = next(train_loader_)
                valid_sample_data, valid_sample_target, _, _ = next(valid_loader_)

                training_loss_res = compute_loss(model_copy(train_sample_data.to(device)),
                                                 train_sample_target.to(device),
                                                 firth_meta_model, model_copy, args)[0]

                valid_loss_res = compute_loss(model_copy(valid_sample_data.to(device)),
                                              valid_sample_target.to(device),
                                              firth_meta_model, model_copy, args)[1]

                if args.ftall:
                    m_params = model_copy.parameters
                else:
                    m_params = model_copy.classifier.parameters
                hyper_grads = hyper_gradient(valid_loss_res,
                                             training_loss_res,
                                             firth_meta_model.parameters,
                                             m_params, args.meta_lr)
                meta_optimizer.zero_grad()
                for p, g in zip(firth_meta_model.parameters(), hyper_grads):
                    p.grad = g
                meta_optimizer.step()

            if epoch % args.meta_valid_step == 0:
                # val_metric = validate(valid_loader, model, device)
                val_metric = validate(valid_loader, model, device)

                if val_metric > best_metric:
                    best_metric = val_metric
                    best_step = epoch
                    best_model = copy.deepcopy(model.state_dict())
                    patience = 0
                    best_lambda_list.append(firth_meta_model.coeff.item())
                else:
                    patience += 1
                    if patience >= args.meta_patience:
                        print('Meta HO early stopping at step {}'.format(epoch))
                        best_lambda_list = best_lambda_list[:best_step + 1]
                        break
                history[epoch] = {
                    'lambda': firth_meta_model.coeff.item(),
                    'val_acc': val_metric,
                    'best_val_acc': best_metric,
                    'best_step': best_step
                }
                last_step_log_meta.update(history[epoch])

            pbar1.update()
            pbar1.set_postfix(ordered_dict=last_step_log_meta)
            epoch += 1

    model.load_state_dict(best_model)
    return best_lambda_list, history


@torch.no_grad()
def validate(val_loader, model, device: torch.device):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    for i, (images, target, _, _) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
    model.train()

    return top1.avg.detach().tolist()


@torch.no_grad()
def test(test_loader, model, device: torch.device):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    for i, (images, target) in enumerate(test_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
    model.train()

    return top1.avg.detach().tolist()


if __name__ == '__main__':
    main()
