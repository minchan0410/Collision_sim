#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path

def split_dataset(train_ratio, val_ratio, test_ratio, suffix, use_ratio, seed, source_folder):
    # 스크립트가 위치한 절대 경로 자동 탐색
    base_dir = Path(__file__).resolve().parent

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio == 0:
        print("오류: 비율의 합이 0이 될 수 없습니다.")
        return

    # 원본 데이터셋 경로 (파라미터 적용)
    source_dir = base_dir / 'mat_txt' / source_folder
    if not source_dir.exists():
        print(f"오류: 경로를 찾을 수 없습니다 -> {source_dir}")
        return

    files = list(source_dir.glob('*.txt'))
    if not files:
        print(f"오류: '{source_folder}' 폴더에 텍스트 파일이 존재하지 않습니다.")
        return

    # 재현성을 위한 시드 고정
    if seed is not None:
        random.seed(seed)

    # 셔플 전에 리스트 정렬을 해주는 것이 안전합니다 (OS별 파일 읽기 순서 차이 방지)
    files.sort()
    random.shuffle(files)

    # 전체 데이터 중 지정된 비율만큼만 사용
    if not (0 < use_ratio <= 1.0):
        print("오류: 사용 비율은 0보다 크고 1.0 이하의 값이어야 합니다.")
        return

    used_total_files = int(len(files) * use_ratio)
    if used_total_files == 0:
        print("오류: 지정한 비율로 선택된 파일이 0개입니다. 비율을 높여주세요.")
        return

    files = files[:used_total_files]

    # 분할 인덱스 계산
    train_r = train_ratio / total_ratio
    val_r = val_ratio / total_ratio

    train_end = int(used_total_files * train_r)
    val_end = train_end + int(used_total_files * val_r)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    # 폴더명에 접미사(suffix) 적용
    dir_suffix = f"_{suffix}" if suffix else ""
    split_names = {
        'train': f'train{dir_suffix}',
        'val': f'val{dir_suffix}',
        'test': f'test{dir_suffix}'
    }

    # 기존 폴더가 있으면 완전히 삭제 후 새로 생성
    for folder_name in split_names.values():
        folder_path = base_dir / 'mat_txt' / folder_name
        if folder_path.exists():
            shutil.rmtree(folder_path)
            print(f"기존 폴더 삭제: {folder_path}")
        folder_path.mkdir(parents=True, exist_ok=True)

    def copy_files(file_list, folder_name):
        target_dir = base_dir / 'mat_txt' / folder_name
        for f in file_list:
            shutil.copy2(f, target_dir / f.name)
        print(f"{folder_name} 폴더: {len(file_list)}개 파일 저장 완료")

    copy_files(train_files, split_names['train'])
    copy_files(val_files, split_names['val'])
    copy_files(test_files, split_names['test'])

    print("\n분할 완료")
    print(f"소스 폴더: {source_dir}")
    print(f"전체 사용 파일 수: {used_total_files}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="텍스트 데이터셋을 train/val/test로 분할합니다.")
    parser.add_argument('--train', type=float, default=8, help='Train 세트 비율')
    parser.add_argument('--val', type=float, default=1, help='Validation 세트 비율')
    parser.add_argument('--test', type=float, default=1, help='Test 세트 비율')
    parser.add_argument('--suffix', type=str, default='', help='폴더명 뒤에 붙일 이름 (예: exp1 입력 시 train_exp1 폴더 생성)')
    parser.add_argument('--use_ratio', type=float, default=1.0, help='사용할 전체 데이터의 비율 (예: 0.1 -> 10%%만 사용)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 셔플 시드 (실험 재현용)')
    # 새로운 파라미터 추가
    parser.add_argument('--source', type=str, default='collision_extract', help='분할할 원본 데이터가 있는 폴더명 (기본값: all)')

    args = parser.parse_args()
    split_dataset(args.train, args.val, args.test, args.suffix, args.use_ratio, args.seed, args.source)