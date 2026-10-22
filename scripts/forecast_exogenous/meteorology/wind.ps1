param(
    [int[]]$Seeds = @(2021, 2022, 2023),
    [string]$Gpu = "0",
    [int]$NumWorkers = 0
)

$env:CUDA_VISIBLE_DEVICES = $Gpu
$PYTHON = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

$model_name = "TimeXer"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Running TimeXer on Meteorology (Wind)" -ForegroundColor Green
Write-Host "  Seeds: $($Seeds -join ', ') | Workers: $NumWorkers | GPU: $Gpu" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

foreach ($seed in $Seeds) {
    Write-Host "`n>>> [Meteorology - Wind] Seed $seed <<<" -ForegroundColor Cyan
    & $PYTHON -u run.py `
      --task_name long_term_forecast `
      --is_training 1 `
      --root_path ./dataset/meteorology `
      --data_path wind.npy `
      --model_id wind `
      --model $model_name `
      --data Meteorology `
      --features MS `
      --seq_len 168 `
      --label_len 1 `
      --pred_len 72 `
      --patch_len 8 `
      --e_layers 2 `
      --enc_in 37 `
      --d_model 512 `
      --d_ff 512 `
      --des "global_wind_seed$seed" `
      --learning_rate 0.0001 `
      --batch_size 4096 `
      --num_workers $NumWorkers `
      --seed $seed
}

if (Test-Path "utils/export_results.py") {
    & $PYTHON utils/export_results.py
}
