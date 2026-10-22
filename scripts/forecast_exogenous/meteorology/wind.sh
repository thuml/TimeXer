export CUDA_VISIBLE_DEVICES=1
model_name=TimeXer

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/meteorology \
  --data_path wind.npy \
  --model_id wind \
  --model $model_name \
  --data Meteorology \
  --features MS \
  --seq_len 168 \
  --label_len 1 \
  --pred_len 72 \
  --patch_len 8 \
  --e_layers 2 \
  --enc_in 37 \
  --d_model 512 \
  --d_ff 512 \
  --des 'global_wind' \
  --learning_rate 0.0001 \
  --batch_size 4096