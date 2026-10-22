# Script giải nén các bộ dữ liệu từ dataset/all_datasets.zip vào đúng thư mục dataset/

$PYTHON = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Giải nén các bộ dataset từ all_datasets.zip" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

& $PYTHON -c "
import os
import zipfile

zip_path = 'dataset/all_datasets.zip'
if not os.path.exists(zip_path):
    print(f'Khong tim thay {zip_path}')
    exit(1)

targets = {
    'all_datasets/ETT-small/': 'dataset/ETT-small',
    'all_datasets/weather/': 'dataset/weather',
    'all_datasets/electricity/': 'dataset/electricity',
    'all_datasets/traffic/': 'dataset/traffic'
}

with zipfile.ZipFile(zip_path, 'r') as z:
    for prefix, out_dir in targets.items():
        if os.path.exists(out_dir):
            print(f'-> [Bo qua] {out_dir} da ton tai.')
            continue
        print(f'-> [Dang giai nen] {prefix} -> {out_dir}...')
        os.makedirs(out_dir, exist_ok=True)
        for member in z.namelist():
            if member.startswith(prefix) and not member.endswith('/'):
                filename = os.path.basename(member)
                if filename:
                    target_file = os.path.join(out_dir, filename)
                    with z.open(member) as source, open(target_file, 'wb') as target:
                        target.write(source.read())
                    print(f'   + {filename}')
print('Xong!')
"
