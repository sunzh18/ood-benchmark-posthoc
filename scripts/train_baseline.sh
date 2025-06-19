
in_dataset=$1
model=$2

GPU=0
# wandb=$4

name='baseline'

model_path="checkpoints/network"

CUDA_VISIBLE_DEVICES=${GPU} python train_baseline.py \
--model ${model} \
--name ${name} \
--batch 128 \
--in_dataset ${in_dataset} \
--lr 0.1 \
--epochs 100 \

# --wandb ${wandb} \
# --model_path ${model_path}
