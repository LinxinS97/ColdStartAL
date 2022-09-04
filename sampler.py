import torch.backends.cudnn as cudnn

import strategies
from custom_dataset.custom_vision_datasets import *
from my_parser import parser
from utils import *


def main():
    args = parser.parse_args()

    if not os.path.exists(args.indices):
        os.makedirs(args.indices)

    if args.dataset == "imagenet":
        args.num_images = 1281167
        args.num_classes = 1000

    elif args.dataset == "imagenet_lt":
        args.num_images = 115846
        args.num_classes = 1000

    elif args.dataset == "cifar100":
        args.num_images = 50000
        args.num_classes = 100

    elif args.dataset in ["cifar10", 'mnist']:
        args.num_images = 50000
        args.num_classes = 10

    else:
        raise NotImplementedError

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    main_worker(args)


def main_worker(args):
    all_indices = np.arange(args.num_images)

    inference_loader = get_inference_loader(args.dataset, all_indices, None, args)
    backbone = get_backbone_model(args.arch, args)
    backbone = nn.DataParallel(backbone).cuda()

    cudnn.benchmark = True

    # get all dataset features and labels in eval mode
    inference_feats, inference_labels = get_feats(inference_loader, torch.device('cuda:0'), args)

    current_indices = np.array([])
    if os.path.isfile(args.resume_indices):
        print("=> Loading current indices: {}".format(args.resume_indices))
        current_indices = np.load(args.resume_indices)
        print('current indices size: {}. {}% of all categories is seen'.format(len(current_indices), len(np.unique(
            inference_labels[current_indices])) / args.num_classes * 100))

    splits = [int(x) for x in args.splits.split(',')]

    if args.name == "uniform":
        print(f"Query sampling with {args.name} started ...")
        strategies.uniform(inference_labels, splits, args)
        return

    if args.name == "random":
        print(f"Query sampling with {args.name} started ...")
        strategies.random(all_indices, inference_labels, splits, args)
        return

    for split in splits:

        unlabeled_indices = np.setdiff1d(all_indices, current_indices)
        print(f"Current unlabeled indices is {len(unlabeled_indices)}.")

        if args.name == "kmeans":
            print(f"Query sampling with {args.name} started ...")
            current_indices = strategies.fast_kmeans(inference_feats, split, args)

        elif args.name == "accu_kmeans":
            print(f"Query sampling with {args.name} started ...")
            sampled_indices = strategies.accu_kmeans(inference_feats, split, unlabeled_indices, args)
            current_indices = np.concatenate((current_indices, sampled_indices), axis=-1)

        elif args.name == "coreset":
            print(f"Query sampling with {args.name} started ...")
            sampled_indices = strategies.core_set(inference_feats[unlabeled_indices],
                                                  inference_feats[current_indices],
                                                  unlabeled_indices,
                                                  split, args)
            current_indices = np.concatenate((current_indices, sampled_indices), axis=-1)

        else:
            raise NotImplementedError("Query sampling method is not implemented")

        print('{} inidices are sampled in total, {} of them are unique'.format(len(current_indices),
                                                                               len(np.unique(current_indices))))
        print('{}% of all categories is seen'.format(
            len(np.unique(inference_labels[current_indices])) / args.num_classes * 100))
        np.save(f'{args.indices}/{args.name}_{args.dataset}_{len(current_indices)}.npy', current_indices)


if __name__ == '__main__':
    main()
