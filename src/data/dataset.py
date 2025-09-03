from __future__ import annotations                  # 최신 타입 힌트(전방 참조) 사용
import os                                           # 경로/파일 처리
import pandas as pd                                 # CSV 로드·데이터프레임 처리
import torch                                        # 텐서 및 학습 프레임워크
from torch.utils.data import Dataset                # PyTorch Dataset 상속
from typing import Tuple, Any                       # 타입 힌트용 튜플, Any
from ..utils.image_io import imread_rgb, make_blank_image  # 이미지 로더·빈 이미지 생성기

# 문서 분류용 Dataset 정의
class DocTypeDataset(Dataset):
    # 초기화: CSV, 이미지 루트, 모드, 변환 등 설정
    def __init__(self, csv_path: str, img_root: str, mode: str = "train",
                 transform=None, return_id: bool = False, retry_blank: bool = True, img_size: int = 224):
        self.df = pd.read_csv(csv_path)             # CSV 로드
        self.img_root = img_root                    # 이미지 루트 경로
        self.mode = mode                            # 동작 모드 (train/val/test)
        self.transform = transform                  # 변환 파이프라인
        self.return_id = return_id                  # ID 반환 여부
        self.retry_blank = retry_blank              # 이미지 로딩 실패 시 빈 이미지 대체 여부
        self.img_size = img_size                    # 이미지 크기

        # 학습/검증 모드일 때 target 컬럼 확인
        if mode in ("train","val"):
            assert "target" in self.df.columns      # 타겟 컬럼 존재 여부 확인

        # ID 컬럼 필수 확인
        assert "ID" in self.df.columns

    # 데이터셋 길이 반환
    def __len__(self) -> int:
        # 데이터프레임의 행 수 반환
        return len(self.df)

    # 단일 이미지 로딩
    def _load_image(self, _id: str):
        path = os.path.join(self.img_root, f"{_id}.jpg")  # 이미지 경로 구성
        img = imread_rgb(path)                            # RGB 이미지 로드
        
        # 로딩 실패 시 빈 이미지 생성
        if img is None and self.retry_blank:
            # 빈 이미지 생성
            img = make_blank_image(self.img_size, self.img_size, 127)
            
        # 이미지 반환
        return img

    # 샘플 단위 접근
    def __getitem__(self, idx: int) -> Any:
        row = self.df.iloc[idx]             # 인덱스 위치의 데이터 로드
        _id = str(row["ID"])                # ID 추출
        img = self._load_image(_id)         # 이미지 로딩

        # 변환 파이프라인 적용
        if self.transform is not None:
            # 변환 적용
            img = self.transform(image=img)["image"]

        # 학습/검증 모드일 경우
        if self.mode in ("train","val"):
            target = int(row["target"])     # 타깃 라벨 추출
            
            # ID 반환 옵션일 때
            if self.return_id:
                # ID 반환
                return img, target, _id

            # ID 반환하지 않을 경우
            return img, target

        # 테스트 모드일 경우
        else:
            # ID 반환 옵션일 때
            if self.return_id:
                # ID 반환
                return img, _id

            # ID 반환하지 않을 경우
            return img
