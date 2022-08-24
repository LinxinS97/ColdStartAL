import os
import torch
from torch import nn
from torchvision import models
from tqdm import tqdm


def load_weights(model, wts_path, args):
    if args.backbone == 'compress':
        # each pre-trained model has a different output size
        # it's 128 for MoCoTeacher
        # and 2048 for swAVTeacher
        if os.path.exists(wts_path):
            print(f"=> loading {args.backbone} weights ")
            wts = torch.load(wts_path)
            if 'state_dict' in wts:
                ckpt = wts['state_dict']
            if 'model' in wts:
                ckpt = wts['model']
            else:
                ckpt = wts

            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            state_dict = {}

            for m_key, m_val in model.state_dict().items():
                if m_key in ckpt:
                    state_dict[m_key] = ckpt[m_key]
                else:
                    state_dict[m_key] = m_val
                    print('not copied => ' + m_key)

            model.load_state_dict(state_dict)
            print(f"Weights of {args.backbone} loaded.")
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(wts_path))

    elif args.backbone == "moco":
        if os.path.isfile(wts_path):
            print("=> loading checkpoint '{}'".format(wts_path))
            checkpoint = torch.load(wts_path, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(wts_path))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(wts_path))


def sample_batch(loader):
    while True:
        for batch in loader:
            yield batch


class ResNet(nn.Module):
    def __init__(self, num_class: int, n_hidden_layers: int, hidden_size: int, arch: str,
                 dropout: float = 0.0):
        super(ResNet, self).__init__()
        self.model = models.__dict__[arch]()

        self.model.fc = nn.Linear(self.model.fc.weight.shape[1], 128)

        if n_hidden_layers == 1:
            self.classifier = nn.Linear(128, num_class)
        else:
            layers = [nn.Linear(128, hidden_size), nn.ReLU(), nn.Dropout(p=dropout)]
            for i in range(n_hidden_layers - 1):
                layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(p=dropout)])
            fcs = nn.Sequential(*layers)
            last_layer = nn.Linear(hidden_size, num_class)
            self.classifier = nn.Sequential(fcs, last_layer)

    def forward(self, x):
        out = self.model(x)
        out = self.classifier(out)
        return out

    def extract_features(self, x):
        out = self.model(x)
        return out


def get_model(arch, args):
    if args.ftall:
        model = ResNet(args.num_classes, n_hidden_layers=1, hidden_size=256, arch=arch)
        return model
    else:
        model = ResNet(args.num_classes, n_hidden_layers=1, hidden_size=256, arch=arch)
        for p in model.model.parameters():
            p.requires_grad = False
        return model


@torch.no_grad()
def get_feats(dataset, loader, model, device, args):
    if args.backbone == "compress":
        cached_feats = f'{args.save}/inference_feats_{dataset}_compress18' \
                       f'_{"MoCo" if "MoCo" in args.weights else "swAV"}Teacher.pth.tar'
    elif args.backbone == "moco":
        cached_feats = f'{args.save}/inference_feats_{dataset}_moco.pth.tar'

    if args.load_cache and os.path.exists(cached_feats):
        print(f'=> loading inference feats of {dataset} from cache: {cached_feats}')
        return torch.load(cached_feats)
    else:
        print('get inference feats =>')

        model.eval()
        feats, labels, ptr = None, None, 0

        for images, target, _, _ in tqdm(loader):
            images = images.to(device)
            cur_targets = target.cpu()
            cur_feats = model.model(images).cpu()
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B

        torch.save((feats, labels), cached_feats)
        return feats, labels


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
