param(
    [int[]]$Seeds = @(2021, 2022, 2023),
    [string]$Gpu = "0",
    [int[]]$Horizons = @(96, 192, 336, 720),
    [int]$NumWorkers = 0
)

$env:CUDA_VISIBLE_DEVICES = $Gpu
$PYTHON = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

$model_name = "TimeXer"
$root_path = "./dataset/ETT-small/"
$data_path = "ETTh2.csv"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Running TimeXer on ETTh2 Dataset (MS setting)" -ForegroundColor Green
Write-Host "  Horizons: $($Horizons -join ', ') | Seeds: $($Seeds -join ', ') | Workers: $NumWorkers | GPU: $Gpu" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

foreach ($seed in $Seeds) {
    $des = "Timexer-MS-seed$seed"

    if ($Horizons -contains 96) {
        Write-Host "`n>>> [ETTh2] Horizon 96 | Seed $seed <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --task_name long_term_forecast `
          --is_training 1 `
          --root_path $root_path `
          --data_path $data_path `
          --model_id "ETTh2_96_96" `
          --model $model_name `
          --data ETTh2 `
          --features MS `
          --seq_len 96 `
          --label_len 48 `
          --pred_len 96 `
          --e_layers 1 `
          --factor 3 `
          --enc_in 7 `
          --dec_in 7 `
          --c_out 7 `
          --d_model 128 `
          --d_ff 128 `
          --batch_size 128 `
          --des $des `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }

    if ($Horizons -contains 192) {
        Write-Host "`n>>> [ETTh2] Horizon 192 | Seed $seed <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --task_name long_term_forecast `
          --is_training 1 `
          --root_path $root_path `
          --data_path $data_path `
          --model_id "ETTh2_96_192" `
          --model $model_name `
          --data ETTh2 `
          --features MS `
          --seq_len 96 `
          --label_len 48 `
          --pred_len 192 `
          --e_layers 1 `
          --factor 3 `
          --enc_in 7 `
          --dec_in 7 `
          --c_out 7 `
          --d_model 128 `
          --d_ff 512 `
          --batch_size 128 `
          --des $des `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }

    if ($Horizons -contains 336) {
        Write-Host "`n>>> [ETTh2] Horizon 336 | Seed $seed <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --task_name long_term_forecast `
          --is_training 1 `
          --root_path $root_path `
          --data_path $data_path `
          --model_id "ETTh2_96_336" `
          --model $model_name `
          --data ETTh2 `
          --features MS `
          --seq_len 96 `
          --label_len 48 `
          --pred_len 336 `
          --e_layers 2 `
          --factor 3 `
          --enc_in 7 `
          --dec_in 7 `
          --c_out 7 `
          --d_model 128 `
          --d_ff 256 `
          --batch_size 16 `
          --des $des `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }

    if ($Horizons -contains 720) {
        Write-Host "`n>>> [ETTh2] Horizon 720 | Seed $seed <<<" -ForegroundColor Cyan
        & $PYTHON -u run.py `
          --task_name long_term_forecast `
          --is_training 1 `
          --root_path $root_path `
          --data_path $data_path `
          --model_id "ETTh2_96_720" `
          --model $model_name `
          --data ETTh2 `
          --features MS `
          --seq_len 96 `
          --label_len 48 `
          --pred_len 720 `
          --e_layers 1 `
          --factor 3 `
          --enc_in 7 `
          --dec_in 7 `
          --c_out 7 `
          --d_model 256 `
          --d_ff 512 `
          --batch_size 128 `
          --des $des `
          --num_workers $NumWorkers `
          --seed $seed `
          --itr 1
    }
}

if (Test-Path "utils/export_results.py") {
    Write-Host "`n[ETTh2] Cập nhật kết quả ra docs/results/raw/results.csv..." -ForegroundColor Yellow
    & $PYTHON utils/export_results.py
}
