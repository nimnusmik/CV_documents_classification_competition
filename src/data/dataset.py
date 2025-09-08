# src/data/dataset.py                                       # 파일 경로 주석
import os                                                   # 경로/파일 처리 모듈
import cv2                                                  # OpenCV (이미지 로딩/처리)
import numpy as np                                          # 배열 연산 라이브러리
import pandas as pd                                         # CSV 및 데이터프레임 처리
import torch                                                # PyTorch 텐서 연산
from torch.utils.data import Dataset                        # PyTorch Dataset 상속 클래스
from PIL import Image                                       # PIL Image 처리
import albumentations as A                                  # Albumentations 증강
from albumentations.pytorch import ToTensorV2               # Tensor 변환

from src.logging.logger import Logger                       # 로그 기록용 Logger 클래스
from typing import Optional, List                           # 타입 힌트 (옵션, 리스트)

# 파일 탐색 시 고려할 확장자 후보 (대소문자 혼용 대비)
_FALLBACK_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]  # 지원 이미지 확장자 목록


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
                f"[DATASET] init | size={len(self.df)} image_dir={self.image_dir} "                 # 데이터셋 크기, 이미지 디렉터리
                f"image_ext='{self.image_ext}' id_col={self.id_col} target_col={self.target_col}"   # 타깃 컬럼
            )
            for i in range(preview):                        # 샘플 개수만큼 반복
                _id = str(self.df.loc[i, self.id_col])      # 샘플 ID 추출
                p = self._resolve_image_path(_id)           # 경로 해석
                self.logger.write(                          # 샘플 정보 로그 기록
                    f"[DATASET] sample{i} id='{_id}' -> path='{p}' exists={os.path.exists(p)}"  # 샘플 경로 확인 로그
                )                                           # 로그 기록 종료

    # ---------------------- 데이터셋 길이 반환 함수 ---------------------- #
    def __len__(self):                                      # 데이터셋 길이 반환 함수 정의
        return len(self.df)                                 # 데이터프레임 길이 반환

    # ---------------------- 확장자 확인 함수 ---------------------- #
    def _has_ext(self, filename: str) -> bool:              # 파일명에 확장자가 있는지 확인 함수 정의
        return os.path.splitext(filename)[1] != ""          # 확장자 유무 확인

    # ---------------------- 확장자 후보 탐색 함수 ---------------------- #
    def _resolve_with_fallbacks(self, stem: str) -> Optional[str]:  # 확장자 없이 들어온 경우 여러 후보로 경로 탐색 함수 정의
        """확장자 없이 들어온 경우, 여러 확장자를 시도."""
        for ext in _FALLBACK_EXTS:                          # 후보 확장자 반복
            cand = os.path.join(self.image_dir, stem + ext) # 후보 경로 생성
            if os.path.exists(cand):                        # 파일 존재 확인
                return cand                                 # 해당 경로 반환
        return None                                         # 실패 시 None 반환

    # ---------------------- 이미지 경로 해석 함수 ---------------------- #
    def _resolve_image_path(self, image_id: str) -> str:    # 이미지 경로 해석 규칙 함수 정의
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
            cand = os.path.join(self.image_dir, image_id + self.image_ext)  # 후보 경로 생성
            if os.path.exists(cand):                        # 존재 확인
                return cand                                 # 해당 경로 반환

        # 3) 후보 확장자 반복 탐색
        # - config의 image_ext가 None인 경우만 탐색
        fb = self._resolve_with_fallbacks(image_id)         # 후보 확장자로 탐색

        # 후보 경로가 존재하면 반환
        if fb is not None:                                  # 후보 경로가 있으면
            return fb                                       # 해당 경로 반환

        # 실패 시 기본적으로 확장자 붙여 반환 (에러 메시지용)
        return os.path.join(self.image_dir, image_id + (self.image_ext or ""))  # 기본 확장자로 경로 반환

    # ---------------------- 이미지 로딩 함수 ---------------------- #
    def _read_image(self, image_id: str):                   # 이미지 로딩 함수 정의
        path = self._resolve_image_path(image_id)           # 이미지 경로 확인
        img = cv2.imread(path, cv2.IMREAD_COLOR)            # OpenCV로 BGR 이미지 로드
        
        # 로드 실패 시
        if img is None:                                     # 이미지 로드 실패 확인
            # 로그 경고
            if not os.path.exists(path) and self.logger is not None:   # 파일 없음 및 로거 존재 확인
                self.logger.write(f"[DATASET][WARN] 이미지 파일 없음: {path}")  # 경고 로그 기록
            # 예외 발생
            raise FileNotFoundError(f"이미지 로드 실패: {path}")    # 파일 없음 예외 발생
        
        # RGB 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # BGR에서 RGB로 색상 변환
        
        # 최종 이미지 반환
        return img                                          # 변환된 이미지 반환

    # ---------------------- 샘플 단위 접근 함수 ---------------------- #
    def __getitem__(self, idx):                             # 샘플 단위 접근 함수 정의
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


# ==================== High-Performance Dataset with Hard Augmentation ==================== #
# 고성능 문서 분류용 Dataset 클래스 정의
class HighPerfDocClsDataset(Dataset):
    """
    고성능 문서 분류용 Dataset
    - Hard Augmentation 지원
    - Epoch에 따른 Augmentation 강도 조절
    - PIL 이미지 로딩 (Albumentations 호환)
    """
    
    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(
        self,
        df: pd.DataFrame,                                   # 입력 데이터프레임
        image_dir: str,                                     # 이미지 저장 디렉터리
        img_size: int = 384,                                # 이미지 크기 (정사각형)
        epoch: int = 0,                                     # 현재 에폭 번호
        total_epochs: int = 10,                             # 총 에폭 수
        is_train: bool = True,                              # 학습 모드 여부
        id_col: str = "ID",                                 # ID 컬럼명
        target_col: Optional[str] = None,                   # 타깃 컬럼명
        logger: Optional[Logger] = None,                    # 로거 객체
    ):
        self.df = df.reset_index(drop=True)                 # 데이터프레임 인덱스 초기화
        self.image_dir = image_dir                          # 이미지 디렉터리 저장
        self.img_size = img_size                            # 이미지 크기 저장
        self.epoch = epoch                                  # 현재 에폭 저장
        self.total_epochs = total_epochs                    # 총 에폭 수 저장
        self.is_train = is_train                            # 학습 모드 플래그 저장
        self.id_col = id_col                                # ID 컬럼명 저장
        self.target_col = target_col                        # 타깃 컬럼명 저장
        self.logger = logger                                # 로거 객체 저장
        
        # Hard augmentation 확률 계산 (epoch이 진행될수록 강해짐)
        self.p_hard = 0.2 + 0.3 * (epoch / total_epochs) if is_train else 0
        
        # 변환 파이프라인 설정
        self._setup_transforms()
        
        # 로거 존재 시
        if self.logger:
            # 데이터셋 정보 로그 기록
            self.logger.write(
                f"[HighPerfDataset] size={len(self.df)} img_size={img_size} "                   # 데이터셋 크기, 이미지 크기
                f"epoch={epoch}/{total_epochs} p_hard={self.p_hard:.3f} is_train={is_train}"    # 에폭 정보, Hard aug 확률, 학습 모드
            )
    
    # ---------------------- 변환 파이프라인 설정 함수 ---------------------- #
    # 변환 파이프라인 설정 함수 정의
    def _setup_transforms(self):
        # 검증/테스트 모드인 경우
        if not self.is_train:
            # 검증/테스트용 변환
            self.transform = A.Compose([                                                                    # 검증용 변환 파이프라인 구성
                A.LongestMaxSize(max_size=self.img_size),                                                   # 최대 크기 유지하며 리사이즈
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0, value=0),   # 필요시 패딩 추가
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),                         # ImageNet 정규화
                ToTensorV2(),                                                                               # PyTorch 텐서로 변환
            ])
            
            # 검증/테스트용 변환 설정 종료
            return
        
        # Normal augmentation 파이프라인 (베이스라인 기반 중급 증강)
        self.normal_aug = A.Compose([                                                       # 일반 증강 파이프라인 구성
            A.LongestMaxSize(max_size=self.img_size),                                       # 최대 크기 유지하며 리사이즈
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0, value=0),  # 필요시 패딩 추가
            
            # 문서 특화 회전 (확률 낮춤)
            A.OneOf([                                                                       # 회전 증강 중 하나 선택
                A.Rotate(limit=90, p=1.0),                                                  # 90도 회전
                A.Rotate(limit=180, p=1.0),                                                 # 180도 회전
                A.Rotate(limit=270, p=1.0),                                                 # 270도 회전
                A.Rotate(limit=15, p=1.0),                                                  # 미세 회전
            ], p=0.5),                                                                      # 50% 확률로 회전 적용
            
            # 기본 기하학적 변환
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=3, p=0.5),   # 기본 변환
            
            # 색상 및 조명 조절
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.7),
            
            # 가벼운 블러/노이즈
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.5),
            
            # 기본 공간 변환
            A.HorizontalFlip(p=0.5),                                                        # 좌우 반전
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),             # ImageNet 정규화
            ToTensorV2(),                                                                   # PyTorch 텐서로 변환
        ])
        
        # Hard augmentation 파이프라인 (베이스라인 기반 고급 증강)
        self.hard_aug = A.Compose([                                                         # Hard augmentation 파이프라인 구성
            # 비율 보존 리사이징 (핵심 개선)
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=0, value=0),
            
            # 문서 특화 회전 + 미세 회전 추가
            A.OneOf([
                A.Rotate(limit=90, p=1.0),
                A.Rotate(limit=180, p=1.0),
                A.Rotate(limit=270, p=1.0),
                A.Rotate(limit=15, p=1.0),  # 미세 회전 추가
            ], p=0.8),
            
            # 기하학적 변환 강화
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, p=1.0),
                A.ElasticTransform(alpha=50, sigma=5, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=1.0),
            ], p=0.6),
            
            # 색상 및 조명 변환 강화
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            ], p=0.9),
            
            # 블러 및 노이즈 강화
            A.OneOf([
                A.MotionBlur(blur_limit=(5, 15), p=1.0),
                A.GaussianBlur(blur_limit=(3, 15), p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.Blur(blur_limit=7, p=1.0),
            ], p=0.8),
            
            # 다양한 노이즈 추가
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 150.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.8), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.8),
            
            # 문서 품질 시뮬레이션 (스캔/복사 효과)
            A.OneOf([
                A.Downscale(scale_min=0.7, scale_max=0.9, p=1.0),
                A.ImageCompression(quality_lower=60, quality_upper=95, p=1.0),
                A.Posterize(num_bits=6, p=1.0),
            ], p=0.5),
            
            # 픽셀 레벨 변환
            A.OneOf([
                A.ChannelShuffle(p=1.0),
                A.InvertImg(p=1.0),
                A.Solarize(threshold=128, p=1.0),
                A.Equalize(p=1.0),
            ], p=0.3),
            
            # 공간 변환
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),  # 문서에서도 유용할 수 있음
                A.Transpose(p=1.0),
            ], p=0.6),
            
            # 조각 제거 (Cutout 계열)
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                               min_holes=1, min_height=8, min_width=8, 
                               fill_value=0, p=1.0),
                A.GridDropout(ratio=0.3, unit_size_min=8, unit_size_max=32, 
                             holes_number_x=5, holes_number_y=5, p=1.0),
            ], p=0.4),
            
            # 최종 정규화
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    # ---------------------- 에폭 업데이트 함수 ---------------------- #
    # 에폭 업데이트 함수 정의
    def update_epoch(self, epoch: int):
        self.epoch = epoch  # 현재 에폭 번호 업데이트
        
        # Hard augmentation 확률 재계산
        self.p_hard = 0.2 + 0.3 * (epoch / self.total_epochs) if self.is_train else 0

        # 로거가 존재하면
        if self.logger:
            # 업데이트 로그 기록
            self.logger.write(f"[HighPerfDataset] updated epoch={epoch}, p_hard={self.p_hard:.3f}")
    
    # ---------------------- 이미지 경로 해석 함수 ---------------------- #
    # 이미지 경로 해석 함수 정의
    def _resolve_image_path(self, image_id: str) -> str:
        # '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG' 확장자가 아닐 경우
        if not image_id.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            # .jpg 확장자 추가
            image_id = image_id + '.jpg'
            
        # 이미지 파일 경로 반환
        return os.path.join(self.image_dir, image_id)
    
    # ---------------------- PIL 이미지 로딩 함수 ---------------------- #
    # PIL로 이미지 로드 함수 정의
    def _read_image(self, image_id: str):
        path = self._resolve_image_path(image_id)           # 이미지 경로 해석
        
        # 파일 존재 여부 확인
        if not os.path.exists(path):
            # 로거 존재 시
            if self.logger:
                # 경고 로그 기록
                self.logger.write(f"[HighPerfDataset][WARN] 이미지 파일 없음: {path}")
            # 파일 없음 예외 발생
            raise FileNotFoundError(f"이미지 로드 실패: {path}")
        
        img = np.array(Image.open(path))                    # PIL로 이미지 로드 후 numpy 배열로 변환
        return img                                          # 이미지 배열 반환
    
    # ---------------------- 데이터셋 길이 반환 함수 ---------------------- #
    # 데이터셋 길이 반환 함수 정의
    def __len__(self):
        return len(self.df)                                 # 데이터프레임 길이 반환
    
    # ---------------------- 샘플 단위 접근 함수 ---------------------- #
    # 샘플 단위 접근 함수 정의
    def __getitem__(self, idx):
        row = self.df.iloc[idx]                             # 인덱스 행 추출
        image_id = str(row[self.id_col])                    # 이미지 ID 추출
        img = self._read_image(image_id)                    # 이미지 로드
        
        # 변환 적용
        if self.is_train and np.random.random() < self.p_hard:     # 학습 모드이고 Hard augmentation 확률에 걸린 경우
            # Hard augmentation 적용
            img = self.hard_aug(image=img)["image"]         # Hard augmentation 적용
        elif self.is_train:                                 # 학습 모드이지만 Normal augmentation인 경우
            # Normal augmentation 적용
            img = self.normal_aug(image=img)["image"]       # Normal augmentation 적용
        else:                                               # 검증/테스트 모드인 경우
            # 검증/테스트 변환 적용
            img = self.transform(image=img)["image"]        # 기본 변환 적용
        
        if self.target_col is None:                         # 추론 모드 (target 없음)
            return img, image_id                            # 이미지와 ID 반환
        else:                                               # 학습/검증 모드
            label = int(row[self.target_col])               # 타깃 라벨 추출
            return img, label                               # 이미지와 라벨 반환


# ==================== Mixup 데이터 증강 함수 ==================== #
# Mixup 데이터 증강 함수 정의
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:                                           # 알파값이 양수인 경우
        lam = np.random.beta(alpha, alpha)                  # 베타 분포에서 람다값 샘플링
    else:                                                   # 알파값이 0 이하인 경우
        lam = 1                                             # 람다값을 1로 설정 (원본 데이터 유지)
    
    batch_size = x.size()[0]                                # 배치 크기 추출
    index = torch.randperm(batch_size).cuda()               # 배치 인덱스 무작위 순열 생성
    mixed_x = lam * x + (1 - lam) * x[index, :]             # 두 이미지를 람다 비율로 혼합
    y_a, y_b = y, y[index]                                  # 원본 라벨과 섞인 라벨 분리
    
    # 혼합된 이미지, 두 라벨, 람다값 반환
    return mixed_x, y_a, y_b, lam
