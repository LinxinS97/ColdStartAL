# For finetune all
#FTALL="--ftall"
#LEARNING_RATE=0.005
# For one layer Linear
 FTALL=""
 LEARNING_RATE=0.001

ACCU_VAL="--accumulate_val"
SEARCH_PARA="--search-coeff"
BATCH_SIZE=64
BUDGET_SIZE=100
T_BUDGET_SIZE=1000
PATIENCE=250
META_LR=0.1
STRATEGY=(entropy smallest_margin largest_margin least_confidence)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for DEVICE in 0 1 2 3; do
  for FN in 0 1 2; do
    SEED=$((1000 * (FN + 1)))

    nohup python ./active_sampler_ft.py \
    $ACCU_VAL \
    $FTALL \
    --seed $SEED \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --lr $LEARNING_RATE \
    --name "${STRATEGY[DEVICE]}" \
    --filenumber $FN \
    --budget-size $BUDGET_SIZE \
    --total-budget-size $T_BUDGET_SIZE \
    --init_coeff 0.0 \
    --load_cache \
    --arch resnet18 \
    --weights weight/compress_pretrain.pth \
    --backbone compress \
    --workers 0 \
    --steps 10000 \
    --patience $PATIENCE \
    --valid_step 1 \
    --valid_size 500 \
    --dataset cifar10 data/ >"./logs/${STRATEGY[DEVICE]}_ftall_accuval_$FN.output" 2>&1 &

    nohup python ./active_sampler_ft.py \
    $SEARCH_PARA \
    $ACCU_VAL \
    $FTALL \
    --seed $SEED \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --lr $LEARNING_RATE \
    --name "${STRATEGY[DEVICE]}" \
    --filenumber $FN \
    --budget-size $BUDGET_SIZE \
    --total-budget-size $T_BUDGET_SIZE \
    --load_cache \
    --arch resnet18 \
    --weights weight/compress_pretrain.pth \
    --backbone compress \
    --workers 0 \
    --steps 10000 \
    --patience $PATIENCE \
    --valid_step 1 \
    --valid_size 500 \
    --dataset cifar10 data/ >"./logs/${STRATEGY[DEVICE]}_ftall_accuval_search_$FN.output" 2>&1 &

    nohup python ./active_sampler_ft.py \
    $ACCU_VAL \
    $FTALL \
    --seed $SEED \
    --batch-size $BATCH_SIZE \
    --meta_lr $META_LR \
    --device $DEVICE \
    --lr $LEARNING_RATE \
    --name "${STRATEGY[DEVICE]}" \
    --filenumber $FN \
    --budget-size $BUDGET_SIZE \
    --total-budget-size $T_BUDGET_SIZE \
    --meta \
    --init_coeff 0.0 \
    --load_cache \
    --arch resnet18 \
    --weights weight/compress_pretrain.pth \
    --backbone compress \
    --workers 0 \
    --steps 10000 \
    --patience $PATIENCE \
    --valid_step 1 \
    --valid_size 500 \
    --dataset cifar10 data/ >"./logs/${STRATEGY[DEVICE]}_meta_ftall_accuval_$FN.output" 2>&1 &
  done
done