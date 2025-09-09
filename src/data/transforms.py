
import albumentations as A                      # albumentations 라이브러리 불러오기 (데이터 증강)
from albumentations.pytorch import ToTensorV2   # Albumentations → PyTorch 텐서 변환기

# ==================== 공통 변환 유틸리티 ==================== #

def _base_resize_and_pad(img_size=384):
    """기본 리사이즈 + 패딩 (모든 변환에서 공통 사용)"""
    return [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0)
    ]

def _imagenet_normalize():
    """ImageNet 정규화 + 텐서 변환 (모든 변환에서 공통 사용)"""
    return [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]

def _document_rotations():
    """문서 특화 회전 (90도 단위)"""
    return A.OneOf([
        A.Rotate(limit=90, p=1.0),
        A.Rotate(limit=180, p=1.0), 
        A.Rotate(limit=270, p=1.0),
    ], p=0.6)


# 기본 학습용 데이터 변환 파이프라인 (기존 버전 - 가벼운 증강)
def build_train_tfms(img_size=384):
    # 변환 파이프라인을 Compose로 묶어 반환
    return A.Compose([
        # 이미지의 최대 변 길이를 img_size로 제한 (비율 유지)
        A.LongestMaxSize(max_size=img_size),
        # 필요 시 패딩 추가 (정사각형 img_size로 맞춤)
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
        # 작은 범위 이동/스케일/회전 (shift 1%, scale ±5%, 회전 ±3도)
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.05, rotate_limit=3, p=0.5),
        # 밝기/대비 무작위 변화 (±10%)
        A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
        # 채널별 평균/표준편차로 정규화 (ImageNet 기준)
        A.Normalize(),
        # Numpy 이미지를 PyTorch 텐서(C,H,W)로 변환
        ToTensorV2()
    ])


# 고성능 학습용 데이터 변환 파이프라인 (베이스라인 기반 - 강력한 증강)
def build_advanced_train_tfms(img_size=384):
    """
    베이스라인 base_line_wandb.ipynb의 고급 증강 기법 적용
    문서 분류에 특화된 강력한 데이터 증강 파이프라인
    """
    return A.Compose([
        # 비율 보존 리사이징 (핵심 개선)
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                      border_mode=0, value=0),
        
        # 문서 특화 회전 + 미세 회전 추가
        A.OneOf([
            A.Rotate(limit=90, p=1.0),
            A.Rotate(limit=180, p=1.0),
            A.Rotate(limit=270, p=1.0),
            A.Rotate(limit=15, p=1.0),  # 미세 회전 추가
        ], p=0.7),
        
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


# 검증/테스트용 데이터(test 데이터) 변환 파이프라인 정의
def build_valid_tfms(img_size=384):
    # 학습보다 단순한 변환 (리사이즈+정규화)
    return A.Compose([
        # 이미지 최대 변 크기를 img_size로 제한
        A.LongestMaxSize(max_size=img_size),
        # 필요 시 패딩 추가 (정사각형 맞춤)
        A.PadIfNeeded(img_size, img_size, border_mode=0, value=0),
        # 정규화 (ImageNet mean/std 사용)
        A.Normalize(),
        # PyTorch 텐서 변환
        ToTensorV2()
    ])


# ==================== 코드 기반 개선된 증강 변환 ==================== #

# Normal Augmentation (기본 증강)
def build_team_normal_tfms(img_size=384):
    """
    사용한 Normal Augmentation
    - 문서 회전 중심 (90, 180, 270도)
    - 적절한 밝기/대비 조절
    - 노이즈 추가로 견고성 향상
    """
    return A.Compose(
        _base_resize_and_pad(img_size) + [
            # 문서 특화 회전 (버전)
            _document_rotations(),
            
            # 밝기/대비 조절
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            
            # 가우시안 노이즈
            A.GaussNoise(var_limit=(30.0, 100.0), p=0.7),
            
            # 좌우 반전
            A.HorizontalFlip(p=0.5),
        ] + _imagenet_normalize()
    )


# Hard Augmentation (강한 증강)
def build_team_hard_tfms(img_size=384):
    """
    사용한 Hard Augmentation
    - 더 강한 회전 (미세 회전 포함)
    - 강력한 블러 효과
    - 높은 밝기/대비 변화
    - JPEG 압축 시뮬레이션
    """
    return A.Compose(
        _base_resize_and_pad(img_size) + [
            # 문서 회전 + 미세 회전 (버전)
            A.OneOf([
                A.Rotate(limit=90, p=1.0),
                A.Rotate(limit=180, p=1.0),
                A.Rotate(limit=270, p=1.0),
                A.Rotate(limit=15, p=1.0),  # 미세 회전 추가
            ], p=0.8),
            
            # 강한 블러 효과
            A.OneOf([
                A.MotionBlur(blur_limit=15, p=1.0),
                A.GaussianBlur(blur_limit=15, p=1.0),
            ], p=0.95),
            
            # 강한 밝기/대비 변화
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.9),
            
            # 강한 노이즈
            A.GaussNoise(var_limit=(50.0, 150.0), p=0.8),
            
            # JPEG 압축 시뮬레이션
            A.JpegCompression(quality_lower=70, quality_upper=100, p=0.5),
            
            # 좌우 반전
            A.HorizontalFlip(p=0.5),
        ] + _imagenet_normalize()
    )


# ==================== Essential TTA 변환 ==================== #

def get_essential_tta_transforms(img_size=384):
    """
    사용한 Essential TTA 5가지 변환
    - 원본
    - 90도 회전
    - 180도 회전
    - 270도 회전  
    - 밝기 개선
    """
    # 공통 base + normalize 조합을 활용
    base_and_norm = lambda augment=[]: A.Compose(_base_resize_and_pad(img_size) + augment + _imagenet_normalize())
    
    return [
        # 1. 원본
        base_and_norm([]),
        
        # 2. 90도 회전
        base_and_norm([A.Rotate(limit=90, p=1.0)]),
        
        # 3. 180도 회전
        base_and_norm([A.Rotate(limit=180, p=1.0)]),
        
        # 4. 270도 회전
        base_and_norm([A.Rotate(limit=-90, p=1.0)]),
        
        # 5. 밝기 개선
        base_and_norm([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)]),
    ]


def get_comprehensive_tta_transforms(img_size=384):
    """
    포괄적인 TTA 변환들 (중복 없는 전체 TTA)
    Essential TTA + 추가 효과적인 변환들
    
    총 15가지 TTA 변형:
    - 기본: 원본
    - 회전: 90°, 180°, 270°, 미세회전(15°)
    - 반전: 좌우, 상하, 대각선
    - 색상: 밝기개선, 감마보정, CLAHE
    - 블러: 가우시안, 모션
    - 노이즈: 가우시안노이즈, 압축
    - 기하: 미세변형
    """
    base_and_norm = lambda augment=[]: A.Compose(_base_resize_and_pad(img_size) + augment + _imagenet_normalize())
    
    return [
        # === 기본 === 
        # 1. 원본
        base_and_norm([]),
        
        # === 회전 계열 ===
        # 2. 90도 회전
        base_and_norm([A.Rotate(limit=90, p=1.0)]),
        # 3. 180도 회전  
        base_and_norm([A.Rotate(limit=180, p=1.0)]),
        # 4. 270도 회전
        base_and_norm([A.Rotate(limit=-90, p=1.0)]),
        # 5. 미세 회전 (15도)
        base_and_norm([A.Rotate(limit=15, p=1.0)]),
        
        # === 반전 계열 ===
        # 6. 좌우 반전
        base_and_norm([A.HorizontalFlip(p=1.0)]),
        # 7. 상하 반전 (문서에서 드물지만 포함)
        base_and_norm([A.VerticalFlip(p=1.0)]),
        # 8. 대각선 반전 (Transpose)
        base_and_norm([A.Transpose(p=1.0)]),
        
        # === 색상/조명 계열 ===
        # 9. 밝기/대비 개선
        base_and_norm([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0)]),
        # 10. 감마 보정
        base_and_norm([A.RandomGamma(gamma_limit=(80, 120), p=1.0)]),
        # 11. CLAHE (대비 제한 적응적 히스토그램 평활화)
        base_and_norm([A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)]),
        
        # === 블러 계열 ===
        # 12. 가우시안 블러 (가벼운)
        base_and_norm([A.GaussianBlur(blur_limit=(3, 7), p=1.0)]),
        # 13. 모션 블러
        base_and_norm([A.MotionBlur(blur_limit=(3, 7), p=1.0)]),
        
        # === 노이즈 계열 ===
        # 14. 가우시안 노이즈 (가벼운)
        base_and_norm([A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)]),
        # 15. JPEG 압축 (문서 스캔 시뮬레이션)
        base_and_norm([A.JpegCompression(quality_lower=85, quality_upper=100, p=1.0)]),
    ]


def get_tta_transforms_by_type(tta_type="essential", img_size=384):
    """
    TTA 타입에 따른 변환 함수 선택
    
    Args:
        tta_type: "essential" (5가지) 또는 "comprehensive" (15가지)
        img_size: 이미지 크기
    
    Returns:
        선택된 TTA 변환 리스트
    """
    if tta_type == "essential":
        return get_essential_tta_transforms(img_size)
    elif tta_type == "comprehensive":
        return get_comprehensive_tta_transforms(img_size)
    else:
        raise ValueError(f"지원하지 않는 TTA 타입: {tta_type}. 'essential' 또는 'comprehensive' 중 선택하세요.")


# ==================== 선택적 변환 빌더 ==================== #

def build_transforms_by_type(transform_type: str, img_size: int = 384):
    """
    변환 타입에 따른 변환 파이프라인 반환
    
    Args:
        transform_type: 'basic', 'advanced', 'team_normal', 'team_hard', 'valid' 
        img_size: 이미지 크기
    
    Returns:
        Albumentations Compose 객체
    """
    transform_map = {
        'basic': build_train_tfms,
        'advanced': build_advanced_train_tfms,
        'team_normal': build_team_normal_tfms,
        'team_hard': build_team_hard_tfms,
        'valid': build_valid_tfms
    }
    
    if transform_type not in transform_map:
        raise ValueError(f"지원하지 않는 변환 타입: {transform_type}")
    
    return transform_map[transform_type](img_size)