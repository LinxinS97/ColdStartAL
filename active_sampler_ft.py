from torch import Tensor

from my_parser import parser
import copy
from typing import Tuple, Any, Optional
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import trange
from utils import *
import pickle
import higher
from strategies import UncertaintySampler
from custom_dataset.custom_vision_datasets import *


def main():
    args = parser.parse_args()
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    if args.dataset == "cifar10":
        args.num_images = 50000
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_images = 50000
        args.num_classes = 100
    elif args.dataset == "svhn":
        args.num_images = 73257
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
        self.act = nn.Softplus()

    def forward(self, outputs: torch.Tensor) -> Tuple[Tensor, Any]:

        n_samps_, n_ways_ = outputs.shape
        n_ways_ = torch.tensor(n_ways_).to(self.device)

        logp = (outputs - torch.logsumexp(outputs, dim=-1, keepdim=True)) / n_ways_

        ceavg = -logp.mean()
        if self.meta:
            return torch.mul(ceavg, self.coeff), ceavg
        return torch.mul(ceavg, self.coeff.detach()), ceavg


def compute_loss(outputs: torch.Tensor,
                 targets: torch.Tensor,
                 loss_firth: torch.nn.Module,
                 args,
                 no_firth: bool = False):
    loss_ce = torch.nn.CrossEntropyLoss()(outputs, targets)
    if no_firth:
        return None, loss_ce, 0, 0
    loss_firth_out, ceavg = loss_firth(outputs)
    loss_total = torch.add(loss_ce, loss_firth_out)

    return loss_total, loss_ce, loss_firth_out, ceavg


def main_worker(args, device: torch.device):
    all_indices = np.arange(args.num_images)
    inference_loader = get_inference_loader(args.dataset, all_indices, None, args)
    if args.dataset in ["cifar10", "cifar100", 'svhn']:
        input_feat = 512
    else:
        input_feat = 784

    if args.ftall:
        model = get_backbone_model(args.arch, args).to(device)
    else:
        model = nn.Linear(input_feat, args.num_classes).to(device)

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
    test_features, _ = get_feats(get_test_loader(dataset=args.dataset, extracted_features=None, args=args), device,
                                 args)
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
            current_indices = np.random.choice(unlabeled_indices, args.budget_size, replace=False)
            shuffled_indices = np.random.choice(valid_indices, len(valid_indices), replace=False)
            current_val_indices = shuffled_indices[:valid_budget_size]
        else:
            print(f"Query sampling with {args.name} started ...")
            unlabeled_loader = get_inference_loader(args.dataset,
                                                    unlabeled_indices,
                                                    features[unlabeled_indices],
                                                    args)
            labeled_loader = None
            if args.name == 'coreset':
                labeled_loader = get_inference_loader(args.dataset, current_indices, features[current_indices], args)

            if args.normal_query:
                sampler = UncertaintySampler(unlabeled_loader,
                                             unlabeled_indices,
                                             query_model,
                                             device,
                                             args.budget_size,
                                             args.ftall,
                                             num_images=args.num_images,
                                             labeled_loader=labeled_loader)
            else:
                sampler = UncertaintySampler(unlabeled_loader,
                                             unlabeled_indices,
                                             model,
                                             device,
                                             args.budget_size,
                                             args.ftall,
                                             num_images=args.num_images,
                                             labeled_loader=labeled_loader)
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
                                             model, device, valid_budget_size, args.ftall, num_images=args.num_images)
                sampled_val_indices = sampler.sample('random')
                current_val_indices = np.concatenate((current_val_indices, sampled_val_indices), axis=-1)

            print(f'{len(current_indices)} training, '
                  f'{len(current_val_indices)} valid inidices are sampled in total, '
                  f'{len(np.unique(current_indices))} of them are unique')

            args.batch_size = len(current_indices) // 10

        print('{}% of all categories is seen'.format(
            len(np.unique(inference_labels[current_indices])) / args.num_classes * 100))
        train_loader = get_train_loader(args.dataset, current_indices, features[current_indices], args)

        print('Training task model started...')
        print(f'current lr: {args.lr}')
        print(f'current bs: {args.batch_size}')
        best_val_acc = -1

        if args.accumulate_val:
            print('Generating new validation set...')
            val_loader = get_inference_loader(args.dataset, current_val_indices, features[current_val_indices], args)

        if args.meta:
            if not args.ftall:
                model = nn.Linear(input_feat, args.num_classes).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
                lr_scheduler = None
            else:
                model = get_backbone_model(args.arch, args).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
                lr_scheduler = None

            firth_meta_model = FirthRegularizer(args.init_coeff,
                                                meta=True,
                                                device=device,
                                                dataloader=firth_val_loader,
                                                max_size=20)
            meta_optimizer = torch.optim.Adam(firth_meta_model.parameters(), lr=args.meta_lr)

            if args.meta_query or args.normal_query:
                print('training w/o FBR')
                if not args.ftall:
                    query_model = nn.Linear(input_feat, args.num_classes).to(device)
                    query_optimizer = torch.optim.Adam(query_model.parameters(), lr=args.lr)
                    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
                    query_lr_scheduler = None
                else:
                    query_model = get_backbone_model(args.arch, args).to(device)
                    query_optimizer = torch.optim.SGD(query_model.parameters(), lr=args.lr, momentum=0.9)
                    query_lr_scheduler = None
                query_firth_meta_model = FirthRegularizer(args.init_coeff,
                                                          meta=False,
                                                          device=device,
                                                          dataloader=firth_val_loader,
                                                          max_size=20)
                normal_history = train(train_loader, val_loader, query_model, query_optimizer, query_lr_scheduler,
                                       query_firth_meta_model, device, args)

            history = meta_train(train_loader, val_loader, model, optimizer, lr_scheduler,
                                 firth_meta_model, meta_optimizer, device, args)

            if args.meta_query:
                test_acc1 = validate(test_loader, query_model, device, args)
                normal_history.update({'final_test_acc1': test_acc1})
                total_history[step] = normal_history
            else:
                test_acc1 = validate(test_loader, model, device, args)
                history.update({'final_test_acc1': test_acc1})
                total_history[step] = history


        else:
            for firth_coeff in [0.0, 0.01, 0.1, 1.0, 3.0]:
                if not args.ftall:
                    model = nn.Linear(input_feat, args.num_classes).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, momentum=0.9)
                    lr_scheduler = None
                else:
                    model = get_backbone_model(args.arch, args).to(device)
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
                    lr_scheduler = None

                if not args.search_coeff:
                    firth_coeff = args.init_coeff

                print(f'Current coeff: {firth_coeff}')

                loss_firth = FirthRegularizer(firth_coeff, device=device, dataloader=firth_val_loader, max_size=20)

                if firth_coeff == 0:
                    no_firth = True
                else:
                    no_firth = False

                history = train(train_loader, val_loader,
                                model, optimizer, lr_scheduler, loss_firth,
                                device, args, no_firth=no_firth)

                val_acc1 = validate(val_loader, model, device, args)
                test_acc1 = validate(test_loader, model, device, args)
                print(f'lambda {firth_coeff} results valid acc as: {val_acc1}')
                print(f'test acc: {test_acc1}')
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
    if_first = ''
    meta_query = ''
    normal_query = ''
    inner = ''
    outer = ''

    if args.meta:
        if_meta = '_meta'
        if not args.first_time_only:
            inner = '_inner' + str(args.meta_inner_loop)
            outer = '_outer' + str(args.meta_update)
        else:
            if_first = '_first'

    if args.accumulate_val:
        if_accu_val = '_accuval'
    if args.ftall:
        if_ftall = '_ftall'
    if args.meta_query:
        meta_query = '_mq'
    if args.normal_query:
        normal_query = '_nq'

    if args.search_coeff:
        with open(f'res/AL_{args.name}_{args.dataset}_budget{args.budget_size}to{args.total_budget_size}'
                  f'{if_ftall}_searchcoeff{if_accu_val}_history_{args.filenumber}.pkl', 'wb') as f:
            pickle.dump(total_history, f)
    else:
        with open(f'res/AL_{args.name}_{args.dataset}_budget{args.budget_size}to{args.total_budget_size}'
                  f'{if_ftall}{if_meta}{meta_query}{normal_query}{if_first}{inner}{outer}_lambda{args.init_coeff}{if_accu_val}_history'
                  f'_{args.filenumber}.pkl', 'wb') as f:
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
    best_metric = -1
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

            loss, loss_ce, loss_firth_out, _ = compute_loss(outputs, target, loss_firth, args, no_firth=no_firth)

            optimizer.zero_grad()
            if no_firth:
                loss_ce.backward()
            else:
                loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # measure accuracy and record loss
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


def meta_train(train_loader: torch.utils.data.DataLoader,
               valid_loader: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler,
               firth_meta_model: torch.nn.Module,
               meta_optimizer: torch.optim.Optimizer,
               device: torch.device,
               args) -> dict:
    best_metric = -1
    patience = 0
    best_step = 0
    history = {}
    last_step_log_meta = {}
    i = 1

    train_loader_ = sample_batch(train_loader)
    valid_loader_ = sample_batch(valid_loader)
    best_model = None

    with trange(args.steps, desc="[META ADJUST]", unit="steps", ncols=150, position=0, leave=True,
                disable=False) as pbar1:
        while i < args.steps + 1:
            model.train()

            images, target, features, _ = next(train_loader_)
            target = target.to(device)

            meta_optimizer.zero_grad()

            if args.ftall:
                images = images.to(device)
            else:
                features = features.to(device)

            if args.first_time_only:
                statement = i == 1
            else:
                statement = i % args.meta_update == 0 or i == 1

            if statement:
                model_copy = copy.deepcopy(model)
                if args.ftall:
                    optimizer_copy = torch.optim.SGD(lr=args.lr, params=model_copy.parameters(), momentum=0.9)
                else:
                    optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=args.lr)
                for _ in range(args.meta_inner_loop):
                    with higher.innerloop_ctx(model_copy, optimizer_copy, device=device) as (meta_model, diffopt):
                        images, target, features, _ = next(train_loader_)
                        target = target.to(device)

                        if args.ftall:
                            images = images.to(device)
                            outputs = meta_model(images)
                        else:
                            features = features.to(device)
                            outputs = meta_model(features)

                        loss, _, _, _ = compute_loss(outputs, target, firth_meta_model, args)
                        diffopt.step(loss)

                        if args.ftall:
                            valid_sample_data, valid_sample_target, _, _ = next(valid_loader_)
                        else:
                            _, valid_sample_target, valid_sample_data, _ = next(valid_loader_)

                        valid_loss_res = compute_loss(meta_model(valid_sample_data.to(device)),
                                                      valid_sample_target.to(device),
                                                      firth_meta_model, args, True)[1]

                        valid_loss_res.backward()
                        meta_optimizer.step()

            if args.ftall:
                outputs = model(images)
            else:
                outputs = model(features)

            loss, _, _, _ = compute_loss(outputs, target, firth_meta_model, args)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Early stopping
            if i % args.valid_step == 0:
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


if __name__ == '__main__':
    main()
