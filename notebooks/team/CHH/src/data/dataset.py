from __future__ import annotations                  # 최신 타입 힌트(전방 참조) 사용

import glob                                         # 와일드카드 기반 파일 탐색
import json                                         # JSON 입출력 처리
import os                                           # 파일 경로 및 존재 여부 확인
from pathlib import Path                            # 객체 지향 경로 관리
from typing import Any                              # 타입 힌트 Any

import pandas as pd                                 # CSV 로드 및 데이터프레임 처리
import torch                                        # PyTorch 관련 모듈
from torch.utils.data import Dataset                # PyTorch Dataset 상속 클래스

from ..utils.image_io import imread_rgb, make_blank_image  # 이미지 로드/빈 이미지 생성 함수

EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")  # 허용 이미지 확장자 목록


# --------------------------- Dataset 클래스 정의 --------------------------- #
# 문서 분류용 Dataset 정의
class DocTypeDataset(Dataset):
    """
    문서 타입 분류 데이터셋.
    - file_index_path가 주어지면 {ID->절대경로} 매핑을 우선 사용.
    - 확장자/대소문자 차이를 흡수.
    - 이미지 로드 실패 시 블랭크 이미지 반환하여 파이프라인 내성 확보.
    """
    
    # 초기화: CSV, 이미지 루트, 모드, 변환 등 설정
    def __init__(
        self,
        csv_path: str,                              # CSV 경로
        img_root: str,                              # 이미지 루트 경로
        mode: str = "train",                        # train/val/test 모드
        transform=None,                             # 변환 파이프라인
        return_id: bool = False,                    # ID 반환 여부
        retry_blank: bool = True,                   # 실패 시 블랭크 반환 여부
        img_size: int = 224,                        # 블랭크 이미지 크기
        file_index_path: str | None = None,         # 파일 인덱스 경로
    ):
        # CSV 파일 읽기
        self.df = pd.read_csv(csv_path)
        # 이미지 루트 경로 저장
        self.img_root = img_root
        # 모드 저장
        self.mode = mode
        # 변환 파이프라인 저장
        self.transform = transform
        # ID 반환 여부 저장
        self.return_id = return_id
        # 실패 시 블랭크 반환 여부 저장
        self.retry_blank = retry_blank
        # 이미지 크기 저장
        self.img_size = img_size

        # 학습/검증 모드일 경우 target 컬럼 필수
        if mode in ("train", "val"):
            # 타겟 컬럼 존재 여부 확인
            assert "target" in self.df.columns, "train/val mode requires 'target' column"
        # ID 컬럼 필수
        assert "ID" in self.df.columns, "CSV must contain 'ID' column"

        # 파일 인덱스 초기화
        self.file_index: dict[str, str] = {}
        # 파일 인덱스 경로가 주어지고 존재하면 로드
        if file_index_path and Path(file_index_path).exists():
            # 파일 인덱스 로드
            with open(file_index_path, "r", encoding="utf-8") as f:
                self.file_index = json.load(f)  # ID->절대경로 매핑

    # 데이터셋 길이 반환
    def __len__(self) -> int:
        # 데이터프레임의 행 수 반환
        return len(self.df)
    
    # --------------------- 경로 확인 메서드 --------------------- #
    # ID에 대한 이미지 경로 확인
    def _resolve_path(self, _id: str) -> str | None:
        # 1) 파일 인덱스 우선
        if self.file_index:
            p = self.file_index.get(str(_id))               # ID->절대경로 매핑
            if p and os.path.exists(p):                     # 경로 존재 여부 확인
                return p                                    # 절대경로 반환
            
        # 2) 확장자 스캔
        for ext in EXTS:
            p = os.path.join(self.img_root, f"{_id}{ext}")  # ID와 확장자를 결합한 경로
            if os.path.exists(p):                           # 경로 존재 여부 확인
                return p                                    # 절대경로 반환
            
        # 3) 와일드카드 탐색 (최후 보루)
        m = glob.glob(os.path.join(self.img_root, f"{_id}.*"))  # ID와 확장자를 결합한 경로

        # 경로 존재 여부 확인
        return m[0] if m else None

    # ------------------------ 이미지 로드 ---------------------- #

    # 단일 이미지 로딩
    def _load_image(self, _id: str):
        # 1) 파일 인덱스 / 확장자 스캔 / 와일드카드까지 모두 시도
        path = self._resolve_path(_id)

        # 이미지 로드
        img = imread_rgb(path) if path else None
        
        # 2) 최종 로딩 실패 시 빈 이미지 생성
        if img is None and self.retry_blank:
            img = make_blank_image(self.img_size, self.img_size, 127)
            
        # 이미지 반환
        return img

    # ------------------------- getitem ------------------------- #
    # 샘플 단위 접근
    def __getitem__(self, idx: int) -> Any:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, f"{row['ID']}.jpg")
        
        # 엄격 모드 적용
        img = imread_rgb(img_path, strict=True)

        if self.transform:
            img = self.transform(image=img)["image"]

        if self.mode in ["train", "val"]:
            return img, row["target"], row["ID"]
        else:
            return img, row["ID"]