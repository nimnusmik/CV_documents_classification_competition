
import albumentations as A                      # albumentations 라이브러리 불러오기 (데이터 증강)
from albumentations.pytorch import ToTensorV2   # Albumentations → PyTorch 텐서 변환기


# 학습용 데이터(train 데이터) 변환 파이프라인 정의
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