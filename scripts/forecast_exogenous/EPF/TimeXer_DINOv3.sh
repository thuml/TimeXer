#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer_DINOv3
regime=${1:-"regime1"}  # choices: regime1, regime2
seed=${2:-2021}

if [ "$regime" == "regime1" ]; then
  dino_freeze="all"
  learning_rate=0.0001
  dino_lr=0.00001
  des="DINOv3_Frozen_Probe"
elif [ "$regime" == "regime2" ]; then
  dino_freeze="partial"
  learning_rate=0.0001
  dino_lr=0.00001
  des="DINOv3_Partial_FT"
else
  echo "Unknown regime: $regime. Choices: regime1, regime2"
  exit 1
fi

echo "=========================================================="
echo "Starting TimeXer-DINOv3 Benchmark on EPF"
echo "Regime: $regime (freeze: $dino_freeze)"
echo "Learning Rate: head=$learning_rate, backbone=$dino_lr"
echo "Seed: $seed"
echo "=========================================================="

PYTHON_EXEC=".venv/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
  PYTHON_EXEC="python3"
fi

# 1. NP (Nord Pool)
$PYTHON_EXEC -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path NP.csv \
  --model_id NP_168_24_${des}_s${seed} \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len 24 \
  --period_len 24 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --post_e_layers 1 \
  --batch_size 4 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate $learning_rate \
  --dino_lr $dino_lr \
  --dino_freeze $dino_freeze \
  --use_amp \
  --use_pooled_bottleneck 1 \
  --seed $seed \
  --itr 1

# 2. PJM
$PYTHON_EXEC -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path PJM.csv \
  --model_id PJM_168_24_${des}_s${seed} \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len 24 \
  --period_len 24 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --post_e_layers 1 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate $learning_rate \
  --dino_lr $dino_lr \
  --dino_freeze $dino_freeze \
  --use_amp \
  --use_pooled_bottleneck 1 \
  --seed $seed \
  --itr 1

# 3. BE (Belgium)
$PYTHON_EXEC -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path BE.csv \
  --model_id BE_168_24_${des}_s${seed} \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len 24 \
  --period_len 24 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --post_e_layers 1 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate $learning_rate \
  --dino_lr $dino_lr \
  --dino_freeze $dino_freeze \
  --use_amp \
  --use_pooled_bottleneck 1 \
  --seed $seed \
  --itr 1

# 4. FR (France)
$PYTHON_EXEC -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path FR.csv \
  --model_id FR_168_24_${des}_s${seed} \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len 24 \
  --period_len 24 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --post_e_layers 1 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate $learning_rate \
  --dino_lr $dino_lr \
  --dino_freeze $dino_freeze \
  --use_amp \
  --use_pooled_bottleneck 1 \
  --seed $seed \
  --itr 1

# 5. DE (Germany)
$PYTHON_EXEC -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/EPF/ \
  --data_path DE.csv \
  --model_id DE_168_24_${des}_s${seed} \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 168 \
  --pred_len 24 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --des $des \
  --patch_len 24 \
  --period_len 24 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --post_e_layers 1 \
  --batch_size 4 \
  --train_epochs 10 \
  --patience 3 \
  --learning_rate $learning_rate \
  --dino_lr $dino_lr \
  --dino_freeze $dino_freeze \
  --use_amp \
  --use_pooled_bottleneck 1 \
  --seed $seed \
  --itr 1
