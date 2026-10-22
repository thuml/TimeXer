param(
    [int[]]$Seeds = @(2021, 2022, 2023),
    [string]$Gpu = "0",
    [int]$NumWorkers = 0
)

$env:CUDA_VISIBLE_DEVICES = $Gpu
$PYTHON = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

$model_name = "TimeXer"
$root_path = "./dataset/electricity/"
$data_path = "electricity.csv"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Running TimeXer on ECL (Electricity) Dataset (MS setting)" -ForegroundColor Green
Write-Host "  Seeds: $($Seeds -join ', ') | Workers: $NumWorkers | GPU: $Gpu" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

foreach ($seed in $Seeds) {
    $des = "Timexer-MS-seed$seed"

    Write-Host "`n>>> [ECL] Horizon 96 | Seed $seed <<<" -ForegroundColor Cyan
    & $PYTHON -u run.py `
      --task_name long_term_forecast `
      --is_training 1 `
      --root_path $root_path `
      --data_path $data_path `
      --model_id "ECL_96_96" `
      --model $model_name `
      --data custom `
      --features MS `
      --seq_len 96 `
      --label_len 48 `
      --pred_len 96 `
      --e_layers 1 `
      --factor 3 `
      --enc_in 321 `
      --dec_in 321 `
      --c_out 321 `
      --des $des `
      --batch_size 4 `
      --num_workers $NumWorkers `
      --seed $seed `
      --itr 1

    Write-Host "`n>>> [ECL] Horizon 192 | Seed $seed <<<" -ForegroundColor Cyan
    & $PYTHON -u run.py `
      --task_name long_term_forecast `
      --is_training 1 `
      --root_path $root_path `
      --data_path $data_path `
      --model_id "ECL_96_192" `
      --model $model_name `
      --data custom `
      --features MS `
      --seq_len 96 `
      --label_len 48 `
      --pred_len 192 `
      --e_layers 1 `
      --factor 3 `
      --enc_in 321 `
      --dec_in 321 `
      --c_out 321 `
      --des $des `
      --batch_size 32 `
      --num_workers $NumWorkers `
      --seed $seed `
      --itr 1

    Write-Host "`n>>> [ECL] Horizon 336 | Seed $seed <<<" -ForegroundColor Cyan
    & $PYTHON -u run.py `
      --task_name long_term_forecast `
      --is_training 1 `
      --root_path $root_path `
      --data_path $data_path `
      --model_id "ECL_96_336" `
      --model $model_name `
      --data custom `
      --features MS `
      --seq_len 96 `
      --label_len 48 `
      --pred_len 336 `
      --e_layers 1 `
      --factor 3 `
      --enc_in 321 `
      --dec_in 321 `
      --c_out 321 `
      --des $des `
      --batch_size 32 `
      --num_workers $NumWorkers `
      --seed $seed `
      --itr 1

    Write-Host "`n>>> [ECL] Horizon 720 | Seed $seed <<<" -ForegroundColor Cyan
    & $PYTHON -u run.py `
      --task_name long_term_forecast `
      --is_training 1 `
      --root_path $root_path `
      --data_path $data_path `
      --model_id "ECL_96_720" `
      --model $model_name `
      --data custom `
      --features MS `
      --seq_len 96 `
      --label_len 48 `
      --pred_len 720 `
      --e_layers 3 `
      --factor 3 `
      --enc_in 321 `
      --dec_in 321 `
      --c_out 321 `
      --des $des `
      --d_model 512 `
      --num_workers $NumWorkers `
      --seed $seed `
      --itr 1
}

if (Test-Path "utils/export_results.py") {
    Write-Host "`n[ECL] Cập nhật kết quả ra docs/results/raw/results.csv..." -ForegroundColor Yellow
    & $PYTHON utils/export_results.py
}
