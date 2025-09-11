# src/inference/infer_highperf.py
"""
고성능 추론 파이프라인
- Swin Transformer & ConvNext 지원: 최신 모델 아키텍처 지원
- Test Time Augmentation (TTA): 다양한 증강으로 예측 정확도 향상
- 앙상블 예측: 여러 모델 결과 통합
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import os                                            # 파일/디렉터리 경로 처리
import torch                                         # PyTorch 메인 모듈
import pandas as pd                                  # 데이터프레임 처리
import numpy as np                                   # 수치 계산 라이브러리
from torch.utils.data import DataLoader, Dataset    # 데이터 로더 클래스, 데이터셋 클래스
import torch.nn.functional as F                      # PyTorch 함수형 인터페이스
from typing import Optional                          # 타입 힌트 (옵셔널)
from tqdm import tqdm                                # 진행률 표시바
from pathlib import Path                             # 경로 처리
from datetime import datetime                        # 날짜/시간 처리
import yaml                                          # YAML 파일 처리

# ------------------------- 프로젝트 유틸 Import ------------------------- #
from src.utils import (
    load_yaml, resolve_path, require_file, require_dir, create_log_path
)  # 핵심 유틸리티
from src.logging.logger import Logger                # 로그 기록 클래스
from src.data.dataset import HighPerfDocClsDataset   # 고성능 문서 분류 데이터셋
from src.data.transforms import get_essential_tta_transforms, get_tta_transforms_by_type  # TTA 변환 함수들
from src.models.build import build_model, get_recommended_model  # 모델 빌드/추천 함수

# ------------------------- 시각화 및 출력 관리 ------------------------- #
from src.utils.visualizations import visualize_inference_pipeline, create_organized_output_structure



# ---------------------- Essential TTA 데이터셋 ---------------------- #
class ConfigurableTTADataset(Dataset):
    """설정 가능한 TTA를 위한 데이터셋"""
    
    def __init__(self, csv_path, image_dir, img_size=384, tta_type="essential", test_df=None, id_col="ID"):
        """
        초기화
        
        Args:
            csv_path: 추론할 데이터 CSV 경로 (None이면 test_df 사용)
            image_dir: 이미지 디렉토리 경로  
            img_size: 이미지 크기
            tta_type: "essential" (5가지) 또는 "comprehensive" (15가지)
            test_df: CSV 대신 사용할 DataFrame (캘리브레이션 모드용)
            id_col: ID 컬럼명
        """
        # 데이터프레임 로드 (CSV 또는 직접 전달)
        if test_df is not None:
            self.df = test_df
        else:
            self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.img_size = img_size
        self.tta_type = tta_type
        self.id_col = id_col
        self.transforms = get_tta_transforms_by_type(tta_type, img_size)  # TTA 변환 함수 리스트
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 샘플의 모든 TTA 변형 반환
        
        Returns:
            (augmented_images, image_id): TTA 변형 리스트, 이미지 ID
        """
        from PIL import Image  # 이미지 처리 라이브러리
        # 데이터 행 추출
        row = self.df.iloc[idx]
        image_id = str(row[self.id_col])  # 이미지 ID 추출
        # 이미지 로드
        img_path = os.path.join(self.image_dir, image_id)
        img = np.array(Image.open(img_path).convert('RGB'))  # RGB 이미지로 변환
        # 모든 TTA 변형 적용
        augmented_images = []
        for transform in self.transforms:
            aug_img = transform(image=img)['image']  # TTA 변형 적용
            augmented_images.append(aug_img)
        return augmented_images, image_id  # 변형된 이미지와 ID 반환


# ---------------------- TTA 예측 함수 ---------------------- #
@torch.no_grad()    # gradient 계산 비활성화
# TTA 예측 함수 정의 (기존 방식 - 호환성 유지)
def predict_with_tta(model, loader, device, num_tta=5):
    """Test Time Augmentation을 사용한 예측 (기존 방식)"""
    model.eval()                                     # 모델을 평가 모드로 설정
    all_preds = []                                   # 모든 TTA 예측 결과 저장 리스트
    
    # TTA 횟수만큼 반복
    for tta_idx in range(num_tta):
        batch_preds = []                             # 배치별 예측 결과 저장 리스트
        # 배치별 추론 시작
        for imgs, _ in tqdm(loader, desc="TTA Inference"):
            imgs = imgs.to(device)                   # 이미지를 GPU로 이동
            logits = model(imgs)                     # 모델 순전파
            probs = F.softmax(logits, dim=1)         # 로짓을 확률로 변환
            batch_preds.append(probs.cpu())          # 예측 결과를 CPU로 이동하여 저장
        # 배치 결합
        tta_preds = torch.cat(batch_preds, dim=0)    # 모든 배치 예측 결과 연결
        all_preds.append(tta_preds)                  # TTA 예측 결과 추가
    # TTA 평균
    final_preds = torch.stack(all_preds).mean(dim=0) # 모든 TTA 결과의 평균 계산
    return final_preds                               # 최종 예측 결과 반환


# ---------------------- Essential TTA 예측 함수 ---------------------- #
@torch.no_grad()    # gradient 계산 비활성화
def predict_with_essential_tta(model, tta_loader, device):
    """Essential TTA를 사용한 예측"""
    model.eval()                                     # 모델을 평가 모드로 설정
    all_predictions = []                             # 모든 예측 결과 저장 리스트
    
    for batch_idx, (images_list, _) in enumerate(tqdm(tta_loader, desc="Essential TTA")):
        batch_size = images_list[0].size(0)          # 배치 크기 추출
        batch_probs = torch.zeros(batch_size, 17).to(device)  # 17개 클래스에 대한 확률 초기화
        
        # 각 TTA 변형별 예측
        for images in images_list:                   # 5가지 TTA 변형 순회
            images = images.to(device)               # 이미지를 GPU로 이동
            logits = model(images)                   # 모델 순전파  
            probs = F.softmax(logits, dim=1)         # 로짓을 확률로 변환
            batch_probs += probs / len(images_list)  # 평균을 위해 누적
            
        all_predictions.append(batch_probs.cpu())    # CPU로 이동하여 저장
    
    # 모든 배치 결합
    final_predictions = torch.cat(all_predictions, dim=0)
    return final_predictions                         # 최종 예측 확률 반환


# ---------------------- 폴드 모델 로드 함수 ---------------------- #
# 폴드별 학습된 모델들 로드 함수 정의
def load_fold_models(fold_results_path, device):
    """
    폴드 결과 파일(들)에서 모델을 로드
    
    Args:
        fold_results_path: 단일 파일 경로 또는 쉼표로 구분된 여러 파일 경로
        device: 모델을 로드할 디바이스
    
    Returns:
        로드된 모델들의 리스트
    """
    models = []  # 모델 리스트 초기화
    
    # 쉼표로 구분된 여러 경로 처리
    if ',' in fold_results_path:
        fold_results_paths = [path.strip() for path in fold_results_path.split(',')]
    else:
        fold_results_paths = [fold_results_path]
    # 각 fold_results 파일 처리
    for path in fold_results_paths:
        if not os.path.exists(path):
            # logger가 없는 경우 print 사용
            if 'logger' in locals():
                logger.write(f"⚠️ [WARNING] Fold results file not found: {path}")
            else:
                print(f"Warning: Fold results file not found: {path}")
            continue
        fold_results = load_yaml(path)               # 폴드 결과 YAML 파일 로드
        # 각 폴드 정보 반복
        for fold_info in fold_results["fold_results"]:
            model_path = fold_info["model_path"]     # 모델 체크포인트 경로 추출
            # 모델 파일 존재 확인
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)  # 체크포인트 로드
                models.append(checkpoint)  # 체크포인트를 모델 리스트에 추가
            else:
                # logger가 없는 경우 print 사용
                if 'logger' in locals():
                    logger.write(f"⚠️ [WARNING] Model not found: {model_path}")
                else:
                    print(f"Warning: Model not found: {model_path}")
    # 로드된 모델들 반환
    return models


# ---------------------- 앙상블 예측 함수 ---------------------- #
# 앙상블 예측 함수 정의 (기존 방식)
def ensemble_predict(models, test_loader, cfg, device, use_tta=True, logger=None):
    all_ensemble_preds = [] # 모든 앙상블 예측 결과 저장 리스트
    
    # 각 모델 체크포인트 반복
    for i, checkpoint in enumerate(models):
        if logger:
            logger.write(f"[MODEL {i+1}/{len(models)}] Processing checkpoint...")
        else:
            print(f"Processing model {i+1}/{len(models)}...")           # 현재 처리 중인 모델 번호 출력
        
        # 모델 생성 및 가중치 로드
        fold_key = f"fold_{i}"
        # 먼저 해당 fold의 모델명을 찾으려 시도
        if "models" in cfg and fold_key in cfg["models"] and "name" in cfg["models"][fold_key]:
            model_name = get_recommended_model(cfg["models"][fold_key]["name"])
        # fold_0의 모델명을 사용 (가장 일반적인 케이스)
        elif "models" in cfg and "fold_0" in cfg["models"] and "name" in cfg["models"]["fold_0"]:
            model_name = get_recommended_model(cfg["models"]["fold_0"]["name"])
        # 기본 모델 설정 사용
        else:
            model_name = get_recommended_model(cfg["model"]["name"])  # fallback
        
        # 모델 빌드
        model = build_model(
            model_name,                                             # 모델명
            cfg["data"]["num_classes"],                             # 클래스 수
            pretrained=False,                                       # 가중치는 체크포인트에서 로드
            drop_rate=cfg["model"]["drop_rate"],                    # 드롭아웃 비율
            drop_path_rate=cfg["model"]["drop_path_rate"],          # 드롭패스 비율
            pooling=cfg["model"]["pooling"]                         # 풀링 타입
        ).to(device)                                                # GPU로 모델 이동
        
        # 체크포인트에서 가중치 로드
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # TTA 사용 시
        if use_tta:                                  
            # TTA 예측 수행
            preds = predict_with_tta(model, test_loader, device, num_tta=3)
        # TTA 미사용 시
        else:
            model.eval()                                            # 모델을 평가 모드로 설정
            batch_preds = []                                        # 배치별 예측 결과 저장 리스트
            for imgs, _ in tqdm(test_loader, desc=f"Model {i+1} Inference"):  # 배치별 추론 시작
                imgs = imgs.to(device)                              # 이미지를 GPU로 이동
                logits = model(imgs)                                # 모델 순전파
                probs = F.softmax(logits, dim=1)                    # 로짓을 확률로 변환
                batch_preds.append(probs.cpu())                     # 예측 결과를 CPU로 이동하여 저장
                
            # 모든 배치 예측 결과 결합
            preds = torch.cat(batch_preds, dim=0)
        
        # 현재 모델 예측 결과를 앙상블 리스트에 추가
        all_ensemble_preds.append(preds)
        
        # 메모리 정리
        del model                   # 모델 객체 삭제
        torch.cuda.empty_cache()    # GPU 메모리 캐시 정리
    
    # 앙상블 평균
    ensemble_preds = torch.stack(all_ensemble_preds).mean(dim=0)    # 모든 모델 예측 결과의 평균 계산
    return ensemble_preds, all_ensemble_preds                       # 앙상블 예측 결과와 개별 모델 예측 결과 반환


# ---------------------- Essential TTA 앙상블 예측 함수 ---------------------- #
def ensemble_predict_with_essential_tta(models, tta_loader, cfg, device, logger=None):
    """Essential TTA를 사용한 앙상블 예측"""
    if logger:
        logger.write(f"🚀 [ENSEMBLE] Starting Essential TTA ensemble prediction with {len(models)} models")
    else:
        print(f"🚀 Essential TTA 앙상블 예측 시작 (모델 수: {len(models)})")
    
    all_ensemble_preds = []  # 모든 앙상블 예측 결과 저장 리스트
    
    # 각 모델 체크포인트 반복
    for i, checkpoint in enumerate(models):
        if logger:
            logger.write(f"[MODEL {i+1}/{len(models)}] Processing Essential TTA prediction...")
        else:
            print(f"📊 모델 {i+1}/{len(models)} 처리 중...")
        
        # 모델 생성 및 가중치 로드
        # fold별 모델 이름 가져오기
        fold_key = f"fold_{i}"
        # 먼저 해당 fold의 모델명을 찾으려 시도
        if "models" in cfg and fold_key in cfg["models"] and "name" in cfg["models"][fold_key]:
            model_name = get_recommended_model(cfg["models"][fold_key]["name"])
        # fold_0의 모델명을 사용 (가장 일반적인 케이스)
        elif "models" in cfg and "fold_0" in cfg["models"] and "name" in cfg["models"]["fold_0"]:
            model_name = get_recommended_model(cfg["models"]["fold_0"]["name"])
        # 기본 모델 설정 사용
        else:
            model_name = get_recommended_model(cfg["model"]["name"])  # fallback
        
        if logger:
            logger.write(f"[MODEL {i+1}] Architecture: {model_name}")
            logger.write(f"[MODEL {i+1}] Drop rate: {cfg['model']['drop_rate']}, Drop path: {cfg['model']['drop_path_rate']}")
            logger.write(f"[MODEL {i+1}] Building model and loading checkpoint...")
        
        # 모델 빌드
        model = build_model(
            model_name,
            cfg["data"]["num_classes"],
            pretrained=False,
            drop_rate=cfg["model"]["drop_rate"],
            drop_path_rate=cfg["model"]["drop_path_rate"],
            pooling=cfg["model"]["pooling"]
        ).to(device)
        
        # 체크포인트에서 가중치 로드
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if logger:
            logger.write(f"[MODEL {i+1}] Model loaded successfully, starting Essential TTA prediction...")
        
        # Essential TTA 예측 수행
        model_preds = predict_with_essential_tta(model, tta_loader, device)
        all_ensemble_preds.append(model_preds)
        
        if logger:
            logger.write(f"[MODEL {i+1}] ✓ Essential TTA prediction completed | shape={model_preds.shape}")
        else:
            print(f"✅ 모델 {i+1} 완료 (예측 형태: {model_preds.shape})")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
    
    # 앙상블 평균 계산
    if logger:
        logger.write("[ENSEMBLE] Computing ensemble average...")
    else:
        print("🔄 앙상블 평균 계산 중...")
    ensemble_preds = torch.stack(all_ensemble_preds).mean(dim=0)
    
    if logger:
        logger.write(f"[ENSEMBLE] ✓ Essential TTA ensemble prediction completed | final_shape={ensemble_preds.shape}")
    else:
        print(f"🎉 Essential TTA 앙상블 예측 완룼! 최종 예측 형태: {ensemble_preds.shape}")
    return ensemble_preds, all_ensemble_preds


# ---------------------- 설정 가능한 TTA 헬퍼 함수 ---------------------- #
def create_configurable_tta_dataloader(sample_csv, test_dir, img_size=384, tta_type="essential", batch_size=32, num_workers=8):
    """설정 가능한 TTA 데이터로더 생성 헬퍼 함수"""
    
    # TTA 데이터셋 생성
    tta_dataset = ConfigurableTTADataset(sample_csv, test_dir, img_size, tta_type)
    
    # TTA 변형 수 계산
    num_tta_transforms = len(tta_dataset.transforms)
    
    # 데이터로더 생성
    tta_loader = DataLoader(
        tta_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: (
            [torch.stack([item[0][i] for item in x]) for i in range(num_tta_transforms)],  # TTA 변형들
            [item[1] for item in x]  # 이미지 ID들
        )
    )
    
    tta_type_name = "Essential (5가지)" if tta_type == "essential" else "Comprehensive (15가지)"
    # logger 전달 필요 - 임시로 print 유지
    print(f"🔧 {tta_type_name} TTA 데이터로더 생성 완료")
    print(f"   - 데이터셋 크기: {len(tta_dataset)}")
    print(f"   - 배치 크기: {batch_size}")  
    print(f"   - TTA 변형 수: {num_tta_transforms}가지")
    
    return tta_loader


# ---------------------- 하위 호환성을 위한 래퍼 함수 ---------------------- #
def create_essential_tta_dataloader(sample_csv, test_dir, img_size=384, batch_size=32, num_workers=8):
    """Essential TTA 데이터로더 생성 (하위 호환성)"""
    return create_configurable_tta_dataloader(sample_csv, test_dir, img_size, "essential", batch_size, num_workers)


# ---------------------- 출력 디렉터리 구조 생성 함수 ---------------------- #
def create_output_structure(cfg):
    """
    올바른 출력 디렉터리 구조를 생성합니다.
    
    구조:
    - experiments/infer/YYYYMMDD/YYYYMMDD_HHMM_run_name/ (모든 결과)
    - experiments/infer/lastest-infer/ (최신 결과 복사)  
    - submissions/YYYYMMDD/ (CSV 파일만)
    
    Returns:
        tuple: (experiments_dir, lastest_dir, submissions_dir, csv_filename)
    """
    current_date = datetime.now().strftime('%Y%m%d')
    current_time = datetime.now().strftime('%H%M')  # 4자리 시간 (HHMM)
    run_name = cfg["project"]["run_name"]
    
    # 올바른 폴더명 형식: 날짜_시간_run_name
    folder_name = f"{current_date}_{current_time}_{run_name}"
    
    # 디렉터리 경로 생성
    experiments_dir = f"experiments/infer/{current_date}/{folder_name}"
    lastest_dir = "experiments/infer/lastest-infer"  # 올바른 폴더명
    submissions_dir = f"submissions/{current_date}"
    
    # lastest-infer 폴더 기존 내용 삭제
    if os.path.exists(lastest_dir):
        import shutil
        shutil.rmtree(lastest_dir)
        print(f"🗑️ 기존 lastest-infer 폴더 삭제 완료")
    
    # 디렉터리 생성
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(lastest_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)
    
    # CSV 파일명 생성
    csv_filename = f"{current_date}_{current_time}_{run_name}_ensemble.csv"
    
    return experiments_dir, lastest_dir, submissions_dir, csv_filename


# ---------------------- 고성능 추론 파이프라인 실행 함수 ---------------------- #
# 고성능 추론 파이프라인 실행 함수 정의
def run_highperf_inference(cfg_path: str, fold_results_path: str, output_path: Optional[str] = None):
    # 설정 로드
    cfg = load_yaml(cfg_path)                                       # YAML 설정 파일 로드
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))            # 설정 파일 디렉터리 경로
    
    # 로거 설정
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    logger = Logger(
        log_path=create_log_path("infer", f"infer_highperf_{timestamp}.log")  # 날짜별 로그 파일 경로
    )
    
    # 파이프라인 시작 로그
    logger.write("[BOOT] high-performance inference pipeline started")
    
    try:
        # GPU/CPU 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.write(f"[BOOT] device={device}")                         # 디바이스 정보 로그
        
        # 경로 확인
        sample_csv = resolve_path(cfg_dir, cfg["data"]["sample_csv"])   # 샘플 CSV 경로 해결
        test_dir = resolve_path(cfg_dir, cfg["data"]["image_dir_test"]) # 테스트 이미지 디렉터리 경로 해결
        require_file(sample_csv, "sample_csv 확인")                     # 샘플 CSV 파일 존재성 검증
        require_dir(test_dir, "test_dir 확인")                          # 테스트 디렉터리 존재성 검증
        
        # 테스트 데이터 로드
        test_df = pd.read_csv(sample_csv)                               # 테스트 데이터 CSV 로드
        logger.write(f"[DATA] loaded test data | shape={test_df.shape}")# 테스트 데이터 로드 로그
        logger.write(f"[DATA] test image directory: {test_dir}")
        logger.write(f"[DATA] image size: {cfg['train']['img_size']}x{cfg['train']['img_size']}")
        logger.write(f"[DATA] batch size: {cfg['train']['batch_size']}")
        
        # 테스트 데이터셋 생성
        test_ds = HighPerfDocClsDataset(
            test_df,                                 # 테스트 데이터프레임
            test_dir,                                # 테스트 이미지 디렉터리
            img_size=cfg["train"]["img_size"],       # 이미지 크기
            is_train=False,                          # 평가 모드 플래그
            id_col=cfg["data"]["id_col"],            # ID 컬럼명
            target_col=None,                         # 추론 모드 (타깃 없음)
            logger=logger                            # 로거 객체
        )                                            # 테스트 데이터셋 생성 완료
        
        # 테스트 데이터로더
        test_loader = DataLoader(                    # 테스트용 데이터로더 생성
            test_ds,                                 # 테스트 데이터셋
            batch_size=cfg["train"]["batch_size"],   # 배치 크기
            shuffle=False,                           # 셔플 비활성화 (추론용)
            num_workers=cfg["project"]["num_workers"],  # 워커 프로세스 수
            pin_memory=True                          # 메모리 고정 활성화
        )
        
        logger.write(f"[DATA] test dataset size: {len(test_ds)}")  # 테스트 데이터셋 크기 로그
        logger.write(f"[DATA] test dataloader created | workers={cfg['project']['num_workers']}")
        
        # TTA 설정 확인
        tta_enabled = cfg.get("inference", {}).get("tta", True)
        tta_type = cfg.get("inference", {}).get("tta_type", "essential")
        logger.write("=" * 50)
        logger.write(f"[TTA] TTA enabled: {tta_enabled}, type: {tta_type}")
        
        # 폴드별 모델 로드
        models = load_fold_models(fold_results_path, device)
        logger.write(f"[MODELS] loaded {len(models)} fold models")
        
        if tta_enabled:
            # 설정 가능한 TTA 데이터로더 생성
            tta_loader = create_configurable_tta_dataloader(
                sample_csv=sample_csv,
                test_dir=test_dir,
                img_size=cfg["train"]["img_size"],
                tta_type=tta_type,
                batch_size=cfg["train"]["batch_size"],
                num_workers=cfg["project"]["num_workers"]
            )
            
            # TTA 앙상블 예측 수행
            logger.write(f"[INFERENCE] starting {tta_type} TTA ensemble prediction...")
            ensemble_preds, individual_model_preds = ensemble_predict_with_essential_tta(models, tta_loader, cfg, device, logger)
        else:
            # 기본 앙상블 예측 (TTA 없음)
            logger.write(f"[INFERENCE] starting basic ensemble prediction (no TTA)...")
            ensemble_preds, individual_model_preds = ensemble_predict(models, test_loader, cfg, device, use_tta=False)
        
        # 최종 예측 클래스
        final_predictions = ensemble_preds.argmax(dim=1).numpy()    # 가장 높은 확률의 클래스 선택
        
        # 신뢰도 점수 계산
        confidence_scores = ensemble_preds.max(dim=1)[0].numpy()    # 최대 확률값을 신뢰도로 사용
        
        #-------------- 결과 저장 및 로그 ---------------------- #
        # 출력 디렉터리 구조 생성
        experiments_dir, lastest_dir, submissions_dir, csv_filename = create_output_structure(cfg)
        
        # TTA 타입에 따라 파일명 수정
        tta_suffix = f"_{tta_type}_tta" if tta_enabled else "_no_tta"
        csv_filename = csv_filename.replace("_ensemble.csv", f"_ensemble{tta_suffix}.csv")
        
        # 각 디렉터리별 경로 설정
        experiments_csv_path = os.path.join(experiments_dir, csv_filename)
        lastest_csv_path = os.path.join(lastest_dir, csv_filename) 
        submissions_csv_path = os.path.join(submissions_dir, csv_filename)
        
        # 출력 경로가 지정된 경우 해당 경로 사용, 아닌 경우 기본 경로 사용
        if output_path is not None:
            main_output_path = output_path
        else:
            main_output_path = experiments_csv_path
        
        # 제출 파일 생성
        submission = test_df.copy()                                             # 테스트 데이터프레임 복사
        submission[cfg["data"]["target_col"]] = final_predictions               # 예측 결과 추가
        
        # 메인 경로에 저장 (사용자 지정 경로 또는 experiments 경로)
        submission.to_csv(main_output_path, index=False)
        
        # 기본 구조에 따라 추가 저장 (사용자 지정 경로가 아닌 경우에만)
        if output_path is None:
            # experiments/infer/lastest-infer/에 복사
            submission.to_csv(lastest_csv_path, index=False)
            
            # submissions/날짜/에 CSV만 저장
            submission.to_csv(submissions_csv_path, index=False)
            
            logger.write(f"[SUCCESS] Files saved to:")
            logger.write(f"  - Main: {main_output_path}")
            logger.write(f"  - Lastest: {lastest_csv_path}")
            logger.write(f"  - Submission: {submissions_csv_path}")
        else:
            logger.write(f"[SUCCESS] Inference completed | output: {main_output_path}")
        
        # 시각화용 기본 디렉터리는 experiments 디렉터리 사용
        viz_base_dir = experiments_dir if output_path is None else os.path.dirname(main_output_path)
        logger.write(f"[RESULT] Prediction distribution:")                      # 예측 분포 로그 시작
        
        # 각 클래스별 예측 수 계산
        for i, count in enumerate(np.bincount(final_predictions)):
            # 클래스별 분포 로그
            logger.write(f"  Class {i}: {count} samples ({count/len(final_predictions)*100:.1f}%)")
        
        #-------------- 추론 결과 시각화 ---------------------- #
        try:
            # 시각화를 위한 출력 디렉터리 설정
            
            if "models" in cfg:
                # 로드된 모델 수에 맞는 모델명 리스트 생성
                model_names = []
                for i in range(len(individual_model_preds)):
                    fold_key = f"fold_{i}"
                    # 먼저 해당 fold의 모델명을 찾으려 시도
                    if fold_key in cfg["models"] and "name" in cfg["models"][fold_key]:
                        model_names.append(cfg["models"][fold_key]["name"])
                    # fold_0의 모델명을 사용 (가장 일반적인 케이스)
                    elif "fold_0" in cfg["models"] and "name" in cfg["models"]["fold_0"]:
                        model_names.append(cfg["models"]["fold_0"]["name"])
                    # 기본 모델 설정 사용
                    else:
                        model_names.append(cfg.get("model", {}).get("name", "unknown_model"))
                
                logger.write("=" * 50)
                logger.write("[VISUALIZATION] Starting individual model visualizations...")
                for i in range(len(individual_model_preds)):
                    # 각 모델별 고유 디렉터리 생성
                    model_viz_dir = os.path.join(viz_base_dir, f"model_{i+1}_{model_names[i]}")
                    os.makedirs(model_viz_dir, exist_ok=True)
                    logger.write(f"[VISUALIZATION] Creating visualizations for model_{i+1}_{model_names[i]} at {model_viz_dir}")
                    
                    # 개별 모델의 예측 결과를 사용하여 시각화 생성
                    individual_preds = individual_model_preds[i].argmax(dim=1).numpy()
                    visualize_inference_pipeline(
                        predictions=individual_preds,
                        model_name=model_names[i],
                        output_dir=model_viz_dir,
                        confidence_scores=individual_model_preds[i].max(dim=1)[0].numpy()
                    )
                    logger.write(f"[VISUALIZATION] ✓ Model {i+1} visualization completed")
                
                # 앙상블 결과 시각화
                logger.write("[VISUALIZATION] Creating ensemble visualizations...")
                ensemble_viz_dir = os.path.join(viz_base_dir, "ensemble")
                os.makedirs(ensemble_viz_dir, exist_ok=True)
                logger.write(f"[VISUALIZATION] Ensemble visualization directory: {ensemble_viz_dir}")
                visualize_inference_pipeline(
                    predictions=final_predictions,
                    model_name="ensemble", 
                    output_dir=ensemble_viz_dir,
                    confidence_scores=confidence_scores
                )
                logger.write("[VISUALIZATION] ✓ Ensemble visualization completed")
                
            else:
                model_names = [cfg["model"].get("name", "unknown")]
                
                # 단일 모델 시각화
                single_model_viz_dir = os.path.join(viz_base_dir, f"single_{model_names[0]}")
                os.makedirs(single_model_viz_dir, exist_ok=True)
            
                visualize_inference_pipeline(
                    predictions=ensemble_preds.numpy(),
                    model_name=model_names[0],
                    output_dir=single_model_viz_dir,
                    confidence_scores=confidence_scores
                )
            
            logger.write(f"[VIZ] Inference visualizations created in {viz_base_dir}")
            
            # lastest-infer에 전체 결과 복사 (기본 구조 사용시에만)
            if output_path is None:
                try:
                    import shutil
                    
                    # 모든 파일과 폴더를 lastest-infer에 복사
                    for item in os.listdir(viz_base_dir):
                        source_item = os.path.join(viz_base_dir, item)
                        dest_item = os.path.join(lastest_dir, item)
                        
                        if os.path.isdir(source_item):
                            # 폴더인 경우 복사
                            shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                        else:
                            # 파일인 경우 복사 (CSV 파일 포함)
                            shutil.copy2(source_item, dest_item)
                    
                    logger.write(f"[SUCCESS] All results copied to lastest-infer folder")
                
                except Exception as copy_error:
                    logger.write(f"[WARNING] Failed to copy to lastest-infer: {str(copy_error)}")
            
        except Exception as viz_error:
            logger.write(f"[WARNING] Visualization failed: {str(viz_error)}")
        
        # 출력 파일 경로 반환
        return main_output_path
        
    # 예외 발생 시
    except Exception as e:
        logger.write(f"[ERROR] Inference failed: {str(e)}") # 에러 로그
        raise                                               # 예외 재발생
    # 최종적으로 실행
    finally:
        logger.write("[SHUTDOWN] Inference pipeline ended") # 파이프라인 종료 로그


# ---------------------- 메인 실행 블록 ---------------------- #
if __name__ == "__main__":
    import sys  # sys 모듈 import
    
    # 인자 개수 확인
    if len(sys.argv) < 3:
        print("Usage: python infer_highperf.py <config_path> <fold_results_path> [output_path]")    # 사용법 출력
        sys.exit(1)                                                                                 # 프로그램 종료
    
    cfg_path = sys.argv[1]                                                          # 설정 파일 경로
    fold_results_path = sys.argv[2]                                                 # 폴드 결과 파일 경로
    output_path = sys.argv[3] if len(sys.argv) > 3 else None                        # 출력 경로 (선택사항)
    
    result_path = run_highperf_inference(cfg_path, fold_results_path, output_path)  # 고성능 추론 실행
    print(f"Inference completed! Results saved to: {result_path}")                  # 추론 완료 메시지 출력
    
    run_highperf_inference(cfg_path, fold_results_path, output_path)                # 고성능 추론 실행
