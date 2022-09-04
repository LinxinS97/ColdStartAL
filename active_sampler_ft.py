from torch import Tensor

from my_parser import parser
import copy
from typing import Tuple, Any
import random

import numpy as np
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import trange
from utils import *
import pickle
from strategies import UncertaintySampler
from custom_dataset.custom_vision_datasets import *


def main():
    args = parser.parse_args()
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    if args.dataset == "imagenet":
        args.num_images = 1281167
        args.num_classes = 1000

    elif args.dataset == "imagenet_lt":
        args.num_images = 115846

    elif args.dataset == "cifar100":
        args.num_images = 50000
        args.num_classes = 100

    elif args.dataset in ["cifar10"]:
        args.num_images = 50000
        args.num_classes = 10
    elif args.dataset in ['mnist', 'fashion_mnist']:
        args.num_images = 60000
        args.num_classes = 10
    else:
        raise NotImplementedError

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    main_worker(args, device)


def get_distribution(indices: np.ndarray, args) -> list:
    dataset = FeatureDataset(args.data, transforms.Compose([
        transforms.ToTensor(),
    ]), indices, None, name=args.dataset)
    distribution = torch.zeros(args.num_classes)
    for _, target, _, _ in dataset:
        # measure accuracy and record loss
        distribution[target] += 1

    return distribution.tolist()


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

    def forward(self, outputs: torch.Tensor) -> Tuple[Tensor, Any]:

        n_samps_, n_ways_ = outputs.shape
        n_ways_ = torch.tensor(n_ways_).to(self.device)

        # logp = torch.div(torch.sub(torch.logsumexp(outputs, dim=-1, keepdim=True), outputs), n_ways_)
        logp = (outputs - torch.logsumexp(outputs, dim=-1, keepdim=True)) / n_ways_

        # negce = logp @ prior.reshape(n_ways_, 1)
        ceavg = -logp.mean()
        if self.meta:
            return torch.mul(ceavg, self.coeff), ceavg
        return torch.mul(ceavg, self.coeff.detach()), ceavg


def get_flatten_vectors(gradients_tensors, to_np=False, norm=False):
    if to_np:
        flatten_vectors = []
        for gradient_parts in gradients_tensors:
            if gradient_parts is not None:
                flatten_vectors.append(gradient_parts.flatten().detach().cpu().numpy())
        v = np.concatenate(flatten_vectors)
        if norm:
            v = v / np.linalg.norm(v, ord=2)
        return v
    else:
        flatten_vectors = []
        for gradient_parts in gradients_tensors:
            if gradient_parts is not None:
                flatten_vectors.append(gradient_parts.flatten())
        return torch.concat(flatten_vectors)


def hyper_gradient(validation_loss: torch.Tensor,
                   training_loss: torch.Tensor,
                   w: torch.Generator,
                   lambda_: torch.Generator,
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
                 valid_output: Optional[torch.Tensor],
                 targets: torch.Tensor,
                 loss_firth: torch.nn.Module,
                 args,
                 no_firth: bool = False):
    loss_ce = torch.nn.CrossEntropyLoss()(outputs, targets)
    if (args.init_coeff == 0 and not args.meta) or no_firth:
        return None, loss_ce, 0, 0
    loss_firth_out, ceavg = loss_firth(outputs)
    loss_total = torch.add(loss_ce, loss_firth_out)

    return loss_total, loss_ce, loss_firth_out, ceavg


def main_worker(args, device: torch.device):
    all_indices = np.arange(args.num_images)
    inference_loader = get_inference_loader(args.dataset, all_indices, None, args)

    if args.ftall:
        model = get_backbone_model(args.arch, args).to(device)
    else:
        model = nn.Linear(512, args.num_classes).to(device)

    # get all dataset labels in eval mode
    features, inference_labels = get_feats(inference_loader, device, args)
    firth_val_loader = get_train_loader(args.dataset, all_indices, features[all_indices], args)

    # valid indices loading
    valid_idxs_file = '{}/random_{}_{}.npy'.format(args.indices, args.dataset, args.valid_size)
    if os.path.isfile(valid_idxs_file):
        print("=> Loading valid indices: {}".format(valid_idxs_file))
        valid_indices = np.load(valid_idxs_file)
        print('valid indices size: {}. {}% of all categories are seen'.format(len(valid_indices), len(np.unique(
            inference_labels[valid_indices])) / args.num_classes * 100))
    else:
        raise FileNotFoundError('valid indices file not found: {}'.format(valid_idxs_file))

    # validation data loading
    test_features, _ = get_feats(get_test_loader(dataset=args.dataset, extracted_features=None, args=args), device, args)
    test_loader = get_test_loader(args.dataset, test_features, args)
    val_loader = get_inference_loader(args.dataset, valid_indices, features[valid_indices], args)

    current_indices = np.array([])
    current_val_indices = np.array([])
    total_history = {}
    total_steps = args.total_budget_size // args.budget_size
    best_lambda = 0.0
    valid_budget_size = args.valid_size // total_steps

    for step in range(1, total_steps + 1):
        # Training data loading code
        unlabeled_indices = np.setdiff1d(all_indices, current_indices)
        unqueried_val_indices = np.setdiff1d(valid_indices, current_val_indices)

        print(f"Current unlabeled indices is {len(unlabeled_indices)} with {args.valid_size} valid samples.")
        if step == 1:
            current_indices_file = '{}/random_{}_{}.npy'.format(args.indices, args.dataset, args.budget_size)
            if os.path.isfile(current_indices_file):
                print("=> Loading first step training indices: {}".format(current_indices_file))
                current_indices = np.load(current_indices_file)
                print('training indices size: {}. {}% of all categories is seen'.format(
                    len(current_indices), len(np.unique(inference_labels[current_indices])) / args.num_classes * 100
                ))
                print(f'current training distribution is: {get_distribution(current_indices, args)}')
            else:
                raise FileNotFoundError('first step training indices file not found: {}'.format(current_indices_file))
            shuffled_indices = np.random.choice(valid_indices, len(valid_indices), replace=False)
            current_val_indices = shuffled_indices[:valid_budget_size]
        else:
            print(f"Query sampling with {args.name} started ...")
            unlabeled_loader = get_inference_loader(args.dataset,
                                                    unlabeled_indices,
                                                    features[unlabeled_indices],
                                                    args)
            sampler = UncertaintySampler(unlabeled_loader,
                                         unlabeled_indices,
                                         model,
                                         device,
                                         args.budget_size,
                                         args.ftall)
            sampled_indices = sampler.sample(args.name)
            current_indices = np.concatenate((current_indices, sampled_indices), axis=-1)
            print(f'sampled training distribution is: {get_distribution(sampled_indices, args)}')
            print(f'current training distribution is: {get_distribution(current_indices, args)}')

            if args.accumulate_val:
                unqueried_val_loader = get_inference_loader(args.dataset,
                                                            unqueried_val_indices,
                                                            features[unqueried_val_indices],
                                                            args)
                sampler = UncertaintySampler(unqueried_val_loader, unqueried_val_indices,
                                             model, device, valid_budget_size, args.ftall)
                sampled_val_indices = sampler.sample(args.name)
                current_val_indices = np.concatenate((current_val_indices, sampled_val_indices), axis=-1)

        print(f'{len(current_indices)} training, '
              f'{len(current_val_indices)} valid inidices are sampled in total, '
              f'{len(np.unique(current_indices))} of them are unique')

        print('{}% of all categories is seen'.format(
            len(np.unique(inference_labels[current_indices])) / args.num_classes * 100))
        train_loader = get_train_loader(args.dataset, current_indices, features[current_indices], args)

        print('Training task model started ...')
        best_val_acc = 0

        if args.accumulate_val:
            print('Generating new validation set...')
            val_loader = get_inference_loader(args.dataset, current_val_indices, features[current_val_indices], args)

        if args.meta:
            if not args.ftall:
                model = nn.Linear(512, args.num_classes).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.6)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                            momentum=0.9, weight_decay=5e-4)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

            firth_meta_model = FirthRegularizer(args.init_coeff,
                                                meta=True,
                                                device=device,
                                                dataloader=firth_val_loader,
                                                max_size=20)
            meta_optimizer = torch.optim.Adam(firth_meta_model.parameters(), lr=args.meta_lr)

            if args.ftall:
                history = meta_ftall(train_loader, val_loader, model, optimizer, lr_scheduler,
                                     firth_meta_model, meta_optimizer, device, args)
            else:
                history = meta_train(train_loader, val_loader, model, optimizer, lr_scheduler,
                                     firth_meta_model, meta_optimizer, device, args)

            test_acc1 = validate(test_loader, model, device, args)
            history.update({'final_test_acc1': test_acc1})
            total_history[step] = history

        else:
            for firth_coeff in [-10, -1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0, 10.0]:
                if not args.ftall:
                    model = nn.Linear(512, args.num_classes).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.6)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                                momentum=0.9, weight_decay=5e-4)
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

                if not args.search_coeff:
                    firth_coeff = args.init_coeff

                print(f'Current coeff: {firth_coeff}')

                loss_firth = FirthRegularizer(firth_coeff, device=device, dataloader=firth_val_loader, max_size=20)

                if firth_coeff == 0:
                    no_firth = True
                else:
                    no_firth = False

                if args.ftall and firth_coeff != 0:
                    history = const_ftall(train_loader, val_loader,
                                          model, optimizer, lr_scheduler, loss_firth,
                                          device, args)
                else:
                    history = train(train_loader, val_loader,
                                    model, optimizer, lr_scheduler, loss_firth,
                                    device, args, no_firth=no_firth)

                val_acc1 = validate(val_loader, model, device, args)
                print(f'lambda {firth_coeff} results valid acc as: {val_acc1}')
                if val_acc1 > best_val_acc:
                    best_val_acc = val_acc1
                    best_lambda = firth_coeff
                    best_model = copy.deepcopy(model.state_dict())
                total_history[step] = history
                if not args.search_coeff:
                    break

            model.load_state_dict(best_model)
            test_acc1 = validate(test_loader, model, device, args)
            history.update({'final_test_acc1': test_acc1})
            if args.search_coeff:
                history.update({'best_lambda': best_lambda})

        print(f'Best lambda: {best_lambda} with top1: {best_val_acc} and test top1: {test_acc1}')

    if_meta = ''
    if_accu_val = ''
    if_ftall = ''
    if args.meta:
        if_meta = '_meta'
    if args.accumulate_val:
        if_accu_val = '_accuval'
    if args.ftall:
        if_ftall = '_ftall'
    if args.search_coeff:
        with open(f'res/AL_{args.name}_{args.dataset}_budget{args.budget_size}to{args.total_budget_size}'
                  f'{if_ftall}{if_meta}_searchcoeff{if_accu_val}_history_{args.filenumber}.pkl', 'wb') as f:
            pickle.dump(total_history, f)
    else:
        with open(f'res/AL_{args.name}_{args.dataset}_budget{args.budget_size}to{args.total_budget_size}'
                  f'{if_ftall}{if_meta}_lambda{args.init_coeff}{if_accu_val}_history_{args.filenumber}.pkl', 'wb') as f:
            pickle.dump(total_history, f)


def train(train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_firth: torch.nn.Module,
          device: torch.device,
          args,
          disable_pbar: bool = False,
          no_firth: bool = False) -> dict:
    i = 1
    best_metric = 0
    best_step = 0
    patience = 0
    history = {}
    last_step_log = {}
    train_loader = sample_batch(train_loader)

    prefix = 'TRAIN'

    with trange(args.steps, desc=prefix, unit="steps", ncols=150,
                position=0, leave=True, disable=disable_pbar) as pbar:
        while i < args.steps + 1:
            model.train()
            images, target, features, _ = next(train_loader)
            target = target.to(device)

            if args.ftall:
                images = images.to(device)
                outputs = model(images)
            else:
                features = features.to(device)
                outputs = model(features)

            # valid_output = extract_valid_output(valid_loader, model, device, args)
            loss, loss_ce, loss_firth_out, _ = compute_loss(outputs, None,
                                                            target, loss_firth, args, no_firth=no_firth)

            optimizer.zero_grad()
            if no_firth:
                loss_ce.backward()
            else:
                loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))

            if i % args.valid_step == 0:
                val_metric = validate(valid_loader, model, device, args)
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
                    'loss_Firth': loss_firth_out.item() if not no_firth else 0,
                    'train_acc': acc1[0].item(),
                    'val_acc': val_metric,
                    'best_val_acc': best_metric,
                    'best_step': best_step
                }
                last_step_log.update(history[i])

            last_step_log['loss_CE'] = loss_ce.item()
            last_step_log['loss_Firth'] = loss_firth_out.item() if not no_firth else 0
            pbar.update()
            pbar.set_postfix(ordered_dict=last_step_log)
            i += 1
    model.load_state_dict(best_model)
    return history


def const_ftall(train_loader: torch.utils.data.DataLoader,
                valid_loader: torch.utils.data.DataLoader,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                loss_firth: torch.nn.Module,
                device: torch.device,
                args):
    for p in model.model.parameters():
        p.requires_grad = True
    # Warm-up
    train(train_loader, valid_loader, model, optimizer, scheduler, loss_firth, device, args, no_firth=True)

    model.classifier = torch.nn.Linear(model.classifier.in_features, args.num_classes).to(device)
    for p in model.model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr / 2, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    # Fine-tune
    history = train(train_loader, valid_loader, model, optimizer, scheduler, loss_firth, device, args, no_firth=False)

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
    patience = 0
    best_step = 0
    history = {}
    last_step_log_meta = {}
    i = 1

    train_loader_ = sample_batch(train_loader)
    valid_loader_ = sample_batch(valid_loader)

    with trange(args.steps, desc="[META ADJUST]", unit="steps", ncols=150, position=0, leave=True,
                disable=False) as pbar1:
        while i < args.steps + 1:
            model.train()

            images, target, features, _ = next(train_loader_)
            target = target.to(device)

            if args.ftall:
                images = images.to(device)
                outputs = model(images)
            else:
                features = features.to(device)
                outputs = model(features)
            # valid_output = extract_valid_output(valid_loader, model, device, args)
            loss, loss_ce, loss_firth_out, _ = compute_loss(outputs, None, target, firth_meta_model, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if i % args.meta_update == 0:
                model_copy = copy.deepcopy(model)
                if args.ftall:
                    train_sample_data, train_sample_target, _, _ = next(train_loader_)
                    valid_sample_data, valid_sample_target, _, _ = next(valid_loader_)
                else:
                    _, train_sample_target, train_sample_data, _ = next(train_loader_)
                    _, valid_sample_target, valid_sample_data, _ = next(valid_loader_)

                train_sample_data = train_sample_data.to(device)
                valid_sample_data = valid_sample_data.to(device)
                # valid_output = extract_valid_output(valid_loader, model_copy, device, args)

                training_loss_res, _, firth_loss_res, _ = compute_loss(model_copy(train_sample_data),
                                                                       None,
                                                                       train_sample_target.to(device),
                                                                       firth_meta_model, args)

                valid_loss_res = compute_loss(model_copy(valid_sample_data),
                                              None,
                                              valid_sample_target.to(device),
                                              firth_meta_model, args, True)[1]
                if args.ftall:
                    w = model_copy.classifier.parameters
                else:
                    w = model_copy.parameters
                hyper_grads = hyper_gradient(valid_loss_res,
                                             training_loss_res,
                                             w,
                                             firth_meta_model.parameters,
                                             args.meta_lr)
                meta_optimizer.zero_grad()
                for p, g in zip(firth_meta_model.parameters(), hyper_grads):
                    # p.grad = torch.tensor(-g.item())
                    p.grad = g
                meta_optimizer.step()

            if i % args.valid_step == 0:
                # val_metric = validate(valid_loader, model, device)
                val_metric = validate(valid_loader, model, device, args)

                if val_metric > best_metric:
                    best_metric = val_metric
                    best_step = i
                    best_model = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience >= args.patience:
                        print('Meta HO early stopping at step {}'.format(i))
                        break
                history[i] = {
                    'lambda': firth_meta_model.coeff.item(),
                    'val_acc': val_metric,
                    'best_val_acc': best_metric,
                    'best_step': best_step
                }
                last_step_log_meta.update(history[i])

                pbar1.update()
                pbar1.set_postfix(ordered_dict=last_step_log_meta)

            i += 1

    model.load_state_dict(best_model)
    return history


def meta_ftall(train_loader: torch.utils.data.DataLoader,
               valid_loader: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler,
               firth_meta_model: torch.nn.Module,
               meta_optimizer: torch.optim.Optimizer,
               device: torch.device,
               args):
    for p in model.model.parameters():
        p.requires_grad = True

    # Warm-up
    train(train_loader, valid_loader, model, optimizer, lr_scheduler, firth_meta_model, device, args, no_firth=True)

    model.classifier = torch.nn.Linear(model.classifier.in_features, args.num_classes).to(device)
    for p in model.model.parameters():
        p.requires_grad = False
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr / 2, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    # Meta fine-tuning
    history = meta_train(train_loader, valid_loader,
                         model, optimizer, lr_scheduler,
                         firth_meta_model, meta_optimizer,
                         device, args)

    return history


def validate(val_loader, model, device: torch.device, args):
    with torch.no_grad():
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        for i, (images, target, feature, _) in enumerate(val_loader):
            target = target.to(device)

            # compute output
            if args.ftall:
                images = images.to(device)
                output = model(images)
            else:
                feature = feature.to(device)
                output = model(feature)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        model.train()

    return top1.avg.detach().tolist()


def extract_valid_output(val_loader, model, device, args):
    model.train()
    outputs = torch.tensor([]).to(device)

    for i, (images, _, feature, _) in enumerate(val_loader):

        # compute output
        if args.ftall:
            images = images.to(device)
            output = model(images)
        else:
            feature = feature.to(device)
            output = model(feature)
        outputs = torch.cat((outputs, output), dim=0)

    return outputs


if __name__ == '__main__':
    main()
