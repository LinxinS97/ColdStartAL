python ./active_sampler_ft.py --init_coeff 0.03 --meta --ftall --load_cache --device 2 --name entropy --arch resnet18 --weights weight/compress_pretrain.pth --backbone compress --batch-size 128 --workers 2 --lr 3e-4 --meta_lr 0.001 --lr_schedule 40,100 --steps 10 --valid_step 10 --patience 10 --meta_patience 100 --meta_valid_step 1 --valid_size 500 --nb_epoch 10000 --dataset cifar10 data/
python ./active_sampler_ft.py --init_coeff 0.03 --meta --ftall --load_cache --device 2 --name smallest_margin --arch resnet18 --weights weight/compress_pretrain.pth --backbone compress --batch-size 128 --workers 2 --lr 3e-4 --meta_lr 0.001 --lr_schedule 40,100 --steps 10 --valid_step 10 --patience 10 --meta_patience 100 --meta_valid_step 1 --valid_size 500 --nb_epoch 10000 --dataset cifar10 data/
python ./active_sampler_ft.py --init_coeff 0.03 --meta --ftall --load_cache --device 2 --name largest_margin --arch resnet18 --weights weight/compress_pretrain.pth --backbone compress --batch-size 128 --workers 2 --lr 3e-4 --meta_lr 0.001 --lr_schedule 40,100 --steps 10 --valid_step 10 --patience 10 --meta_patience 100 --meta_valid_step 1 --valid_size 500 --nb_epoch 10000 --dataset cifar10 data/
python ./active_sampler_ft.py --init_coeff 0.03 --meta --ftall --load_cache --device 2 --name least_confidence --arch resnet18 --weights weight/compress_pretrain.pth --backbone compress --batch-size 128 --workers 2 --lr 3e-4 --meta_lr 0.001 --lr_schedule 40,100 --steps 10 --valid_step 10 --patience 10 --meta_patience 100 --meta_valid_step 1 --valid_size 500 --nb_epoch 10000 --dataset cifar10 data/
python ./active_sampler_ft.py --accumulate_val --init_coeff 0.03 --meta --ftall --load_cache --device 2 --name entropy --arch resnet18 --weights weight/compress_pretrain.pth --backbone compress --batch-size 128 --workers 2 --lr 3e-4 --meta_lr 0.001 --lr_schedule 40,100 --steps 10 --valid_step 10 --patience 10 --meta_patience 100 --meta_valid_step 1 --valid_size 500 --nb_epoch 10000 --dataset cifar10 data/
python ./active_sampler_ft.py --accumulate_val --init_coeff 0.03 --meta --ftall --load_cache --device 2 --name smallest_margin --arch resnet18 --weights weight/compress_pretrain.pth --backbone compress --batch-size 128 --workers 2 --lr 3e-4 --meta_lr 0.001 --lr_schedule 40,100 --steps 10 --valid_step 10 --patience 10 --meta_patience 100 --meta_valid_step 1 --valid_size 500 --nb_epoch 10000 --dataset cifar10 data/
python ./active_sampler_ft.py --accumulate_val --init_coeff 0.03 --meta --ftall --load_cache --device 2 --name largest_margin --arch resnet18 --weights weight/compress_pretrain.pth --backbone compress --batch-size 128 --workers 2 --lr 3e-4 --meta_lr 0.001 --lr_schedule 40,100 --steps 10 --valid_step 10 --patience 10 --meta_patience 100 --meta_valid_step 1 --valid_size 500 --nb_epoch 10000 --dataset cifar10 data/
python ./active_sampler_ft.py --accumulate_val --init_coeff 0.03 --meta --ftall --load_cache --device 2 --name least_confidence --arch resnet18 --weights weight/compress_pretrain.pth --backbone compress --batch-size 128 --workers 2 --lr 3e-4 --meta_lr 0.001 --lr_schedule 40,100 --steps 10 --valid_step 10 --patience 10 --meta_patience 100 --meta_valid_step 1 --valid_size 500 --nb_epoch 10000 --dataset cifar10 data/
