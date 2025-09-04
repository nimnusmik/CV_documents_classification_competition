# src/data/dataset.py                                       # 파일 경로 주석
import os                                                   # 경로/파일 처리 모듈
import cv2                                                  # OpenCV (이미지 로딩/처리)
import numpy as np                                          # 배열 연산 라이브러리
import pandas as pd                                         # CSV 및 데이터프레임 처리
from torch.utils.data import Dataset                        # PyTorch Dataset 상속 클래스

from src.utils.logger import Logger                         # 로그 기록용 Logger 클래스
from typing import Optional, List                           # 타입 힌트 (옵션, 리스트)

# 파일 탐색 시 고려할 확장자 후보 (대소문자 혼용 대비)
_FALLBACK_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


# 문서 분류용 PyTorch Dataset 정의
class DocClsDataset(Dataset):
    # 초기화 함수
    def __init__(
        self,
        df: pd.DataFrame,                                   # 입력 데이터프레임 (ID, target 포함)
        image_dir: str,                                     # 이미지 저장 디렉터리
        image_ext: str,                                     # 이미지 확장자 (config 지정)
        id_col: str,                                        # ID 컬럼명
        target_col: Optional[str] = None,                   # 타깃 컬럼명 (없으면 추론 모드)
        transform=None,                                     # 변환 파이프라인 (Albumentations 등)
        logger: Logger | None = None,                       # 로거 객체 (없으면 None)
    ):
        self.df = df.reset_index(drop=True)                 # 데이터프레임 인덱스 초기화
        self.image_dir = image_dir                          # 이미지 루트 디렉터리 저장
        self.image_ext = (image_ext or "").strip()          # 이미지 확장자 문자열 정리
        self.id_col = id_col                                # ID 컬럼 저장
        self.target_col = target_col                        # 타깃 컬럼 저장
        self.transform = transform                          # 변환 함수 저장
        self.logger = logger                                # 로거 객체 저장

        # sanity log: 초기 상태 로그 (상위 3개 샘플 확인)
        if self.logger is not None:                         # 로거가 존재하면
            preview = min(3, len(self.df))                  # 최대 3개 샘플 확인
            self.logger.write(                              # 데이터셋 정보 로그 기록
                f"[DATASET] init | size={len(self.df)} image_dir={self.image_dir} "
                f"image_ext='{self.image_ext}' id_col={self.id_col} target_col={self.target_col}"
            )
            for i in range(preview):                        # 샘플 개수만큼 반복
                _id = str(self.df.loc[i, self.id_col])      # 샘플 ID 추출
                p = self._resolve_image_path(_id)           # 경로 해석
                self.logger.write(                          # 샘플 정보 로그 기록
                    f"[DATASET] sample{i} id='{_id}' -> path='{p}' exists={os.path.exists(p)}"
                )

    # 데이터셋 길이 반환
    def __len__(self):
        return len(self.df)                                 # 데이터프레임 길이 반환

    # 파일명에 확장자가 있는지 확인
    def _has_ext(self, filename: str) -> bool:
        return os.path.splitext(filename)[1] != ""          # 확장자 유무 확인

    # 확장자 없이 들어온 경우 여러 후보로 경로 탐색
    def _resolve_with_fallbacks(self, stem: str) -> Optional[str]:
        """확장자 없이 들어온 경우, 여러 확장자를 시도."""
        for ext in _FALLBACK_EXTS:                          # 후보 확장자 반복
            cand = os.path.join(self.image_dir, stem + ext) # 후보 경로 생성
            if os.path.exists(cand):                        # 파일 존재 확인
                return cand                                 # 해당 경로 반환
        return None                                         # 실패 시 None 반환

    # 이미지 경로 해석 규칙
    def _resolve_image_path(self, image_id: str) -> str:
        """
        규칙:
        1) image_id에 확장자가 있으면 그대로 사용
        2) 확장자가 없으면 config의 image_ext 시도
        3) 그래도 없으면 FALLBACK_EXTS 탐색
        """
        image_id = str(image_id)                            # ID 문자열 변환

        # 1) 확장자가 이미 포함된 경우
        if self._has_ext(image_id):                         # 확장자 확인
            path = os.path.join(self.image_dir, image_id)   # 경로 생성
            return path                                     # 그대로 반환

        # 2) config 지정 확장자 시도
        if self.image_ext:                                  # 확장자가 있으면
            cand = os.path.join(self.image_dir, image_id + self.image_ext)
            if os.path.exists(cand):                        # 존재 확인
                return cand                                 # 해당 경로 반환

        # 3) 후보 확장자 반복 탐색
        # - config의 image_ext가 None인 경우만 탐색
        fb = self._resolve_with_fallbacks(image_id)

        # 후보 경로가 존재하면 반환
        if fb is not None:
            return fb

        # 실패 시 기본적으로 확장자 붙여 반환 (에러 메시지용)
        return os.path.join(self.image_dir, image_id + (self.image_ext or ""))

    # 이미지 로딩 함수
    def _read_image(self, image_id: str):
        path = self._resolve_image_path(image_id)   # 이미지 경로 확인
        img = cv2.imread(path, cv2.IMREAD_COLOR)    # OpenCV로 BGR 이미지 로드
        
        # 로드 실패 시
        if img is None:
            # 로그 경고
            if not os.path.exists(path) and self.logger is not None:
                self.logger.write(f"[DATASET][WARN] 이미지 파일 없음: {path}")
            # 예외 발생
            raise FileNotFoundError(f"이미지 로드 실패: {path}")
        
        # RGB 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 최종 이미지 반환
        return img

    # 샘플 단위 접근
    def __getitem__(self, idx):
        row = self.df.iloc[idx]                             # 인덱스 행 추출
        image_id = str(row[self.id_col])                    # 이미지 ID 추출
        img = self._read_image(image_id)                    # 이미지 로드

        if self.transform is not None:                      # 변환 파이프라인 적용 여부 확인
            img = self.transform(image=img)["image"]        # 변환 적용

        if self.target_col is None:                         # 추론 모드 (target 없음)
            return img, image_id                            # 이미지와 ID 반환
        else:                                               # 학습/검증 모드
            label = int(row[self.target_col])               # 타깃 라벨 추출
            return img, label                               # 이미지와 라벨 반환
