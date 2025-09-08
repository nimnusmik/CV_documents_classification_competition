
import albumentations as A                      # albumentations 라이브러리 불러오기 (데이터 증강)
from albumentations.pytorch import ToTensorV2   # Albumentations → PyTorch 텐서 변환기


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