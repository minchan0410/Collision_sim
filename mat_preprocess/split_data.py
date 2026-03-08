import argparse
import random
import shutil
from pathlib import Path

def split_dataset(train_ratio, val_ratio, test_ratio):
    # 스크립트(split_data.py)가 위치한 절대 경로(Mat_Process)를 자동으로 찾음
    base_dir = Path(__file__).resolve().parent 
    
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio == 0:
        print("오류: 비율의 합이 0이 될 수 없습니다.")
        return

    train_r = train_ratio / total_ratio
    val_r = val_ratio / total_ratio

    # 동적으로 찾은 base_dir을 기준으로 경로 설정
    source_dir = base_dir / 'mat_txt' / 'all'
    if not source_dir.exists():
        print(f"오류: 경로를 찾을 수 없습니다 -> {source_dir}")
        return

    files = list(source_dir.glob('*.txt'))
    if not files:
        print("오류: 텍스트 파일이 존재하지 않습니다.")
        return

    random.shuffle(files)
    total_files = len(files)

    train_end = int(total_files * train_r)
    val_end = train_end + int(total_files * val_r)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for split in ['train', 'val', 'test']:
        (base_dir / 'mat_txt' / split).mkdir(parents=True, exist_ok=True)

    def copy_files(file_list, split_name):
        target_dir = base_dir / 'mat_txt' / split_name
        for f in file_list:
            shutil.copy(f, target_dir / f.name)
        print(f"{split_name} 폴더: {len(file_list)}개 파일 저장 완료")

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="텍스트 데이터셋을 train/val/test로 분할합니다.")
    parser.add_argument('--train', type=float, default=8, help='Train 세트 비율')
    parser.add_argument('--val', type=float, default=1, help='Validation 세트 비율')
    parser.add_argument('--test', type=float, default=1, help='Test 세트 비율')
    # --dir 옵션은 더 이상 필요 없으므로 제거했습니다.

    args = parser.parse_args()
    split_dataset(args.train, args.val, args.test)