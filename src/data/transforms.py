from __future__ import annotations                  # 최신 타입 힌트(전방 참조) 사용
import albumentations as A                          # 이미지 증강 라이브러리
from albumentations.pytorch import ToTensorV2       # Albumentations 결과를 PyTorch Tensor로 변환

IMAGENET_MEAN = (0.485, 0.456, 0.406)               # ImageNet 평균값 (정규화용)
IMAGENET_STD  = (0.229, 0.224, 0.225)               # ImageNet 표준편차 (정규화용)

# 공통으로 적용되는 전처리: 크기 제한 후 패딩
def _common(size: int):
    return [
        A.LongestMaxSize(size=size),                                # 긴 변 기준으로 크기 조정
        A.PadIfNeeded(min_height=size, min_width=size,              # 최소 높이/너비 보장
                      border_mode=0, value=0)                       # 빈 공간은 0(검정)으로 패딩
    ]

# 학습용 변환 정의
def get_train_transforms(size: int = 224, cfg: dict | None = None):
    # 증강 리스트 초기화
    augs = [
        *_common(size),                                             # 공통 변환 추가
        A.Rotate(limit=cfg.get("rotate_limit", 25) if cfg else 25,  # 회전 변환
                 border_mode=0, p=0.6),
        A.RandomResizedCrop(size, size,                             # 무작위 크롭 후 리사이즈
                            scale=(0.8,1.0), ratio=(0.9,1.1), p=0.6),
        A.HorizontalFlip(p=(cfg.get("hflip_p", 0.2) if cfg else 0.2)), # 수평 뒤집기
        A.ColorJitter(0.2,0.2,0.2,0.1, p=0.4)                       # 밝기/대비/채도/색조 변화
    ]

    # CLAHE 옵션 적용 여부 확인
    if cfg and cfg.get("use_clahe", True):
        augs.append(A.CLAHE(clip_limit=2.0, p=0.3))                 # CLAHE 추가

    # Blur 옵션 적용 여부 확인
    if cfg and cfg.get("use_blur", True):
        augs.append(A.GaussianBlur(blur_limit=(3,5), p=0.2))        # Gaussian Blur 추가

    # Affine 옵션 적용 여부 확인
    if cfg and cfg.get("use_affine", True):
        augs.append(A.Affine(scale=(0.95,1.05),                     # 스케일 변환
                             shear=(-5,5),                          # 시어 변환
                             translate_percent=(0.0,0.02),          # 이동 변환
                             p=0.3))                                # 적용 확률

    augs += [A.Normalize(IMAGENET_MEAN, IMAGENET_STD),              # 정규화 추가
             ToTensorV2()]                                          # 텐서 변환 추가

    # 최종 변환 파이프라인 반환
    return A.Compose(augs)

# 검증/테스트용 변환 정의
def get_val_transforms(size: int = 224):
    # 공통 변환 추가
    return A.Compose([*_common(size),
                      A.Normalize(IMAGENET_MEAN, IMAGENET_STD),    # 정규화
                      ToTensorV2()])                               # 텐서 변환
