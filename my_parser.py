import argparse


parser = argparse.ArgumentParser(description='Unsupervised distillation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset.')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet50', help='model architecture')
parser.add_argument('--steps', default=200, type=int, help='number of total steps to run')
parser.add_argument('--valid_step', default=10, type=int, help='constant gap for validation')
parser.add_argument('--meta_valid_step', default=5, type=int, help='constant gap for meta learning validation')
parser.add_argument('--patience', default=20, type=int, help='number of total steps to run')
parser.add_argument('--meta_patience', default=20, type=int, help='number of total steps to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-tb', '--test-batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--budget-size', type=int, default=200, help='query budget size per step.')
parser.add_argument('--total-budget-size', type=int, default=2000, help='total query budget size.')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--meta_lr', default=0.1, type=float, help='initial learning rate for meta method')
parser.add_argument('--meta_update', default=10, type=int, help='iteration until meta gradient update')
parser.add_argument('--init_coeff', default=0.1, type=float,
                    help='initial coefficient for meta & constant lambda methods')
parser.add_argument('--search-coeff', action="store_true",
                    help='search best coefficient for constant methods')
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-indices', default='', type=str, metavar='PATH',
                    help='path to latest selected indices (default: none)')
parser.add_argument('--seed', default=1e4, type=int, help='seed for initializing training. ')
parser.add_argument('--save', default='./output', type=str, help='experiment output directory')
parser.add_argument('--indices', default='./indices', type=str, help='experiment input directory')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--weights', dest='weights', type=str, required=True, help='pre-trained model weights')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('--lr_schedule', type=str, default='30,60,90', help='lr drop schedule')
parser.add_argument('--splits', type=str, default='', help='splits of unlabeled data to be labeled')
parser.add_argument('--name', type=str, default='', help='name of query strategy')
parser.add_argument('--backbone', type=str, default='compress', help='name of method to do linear evaluation on.')
parser.add_argument('--ftall', action='store_true', help='finetune all layers of resnet')
parser.add_argument('--meta', action='store_true', help='use meta learning method to find the best solution for lambda')
parser.add_argument('--valid_size', type=int, default=1500, help='valid set size.')
parser.add_argument('--accumulate_val', action='store_true', help='let model select validation set itself')
parser.add_argument('--filenumber', type=int, default=0, help='index of saved files')
parser.add_argument('--device', type=int, default=0, help='gpu device id')
