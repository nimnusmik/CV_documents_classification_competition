# src/inference/infer_calibrated.py
"""
캘리브레이션이 적용된 고성능 추론 파이프라인

Temperature Scaling을 통한 확률 보정이 적용된 앙상블 추론을 제공합니다.
"""

import os
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# 프로젝트 모듈 import
from src.utils import load_yaml, create_log_path
from src.logging.logger import Logger
from src.models.build import build_model
from src.data.dataset import HighPerfDocClsDataset
from src.data.transforms import get_tta_transforms_by_type
from src.inference.infer_highperf import load_fold_models, get_recommended_model
from src.calibration import TemperatureScaling, CalibrationTrainer
from torch.utils.data import DataLoader


def run_calibrated_inference(
    cfg_path: str, 
    fold_results_path: str, 
    output_path: Optional[str] = None,
    use_tta: bool = True
) -> str:
    """
    캘리브레이션이 적용된 고성능 추론 실행
    
    Args:
        cfg_path: 추론 설정 파일 경로
        fold_results_path: 폴드 결과 파일 경로 (fold_results.yaml)
        output_path: 결과 저장 경로 (None시 자동 생성)
        use_tta: TTA 사용 여부
        
    Returns:
        생성된 제출 파일 경로
    """
    # 설정 로드
    cfg = load_yaml(cfg_path)
    
    # 로거 설정
    timestamp = time.strftime("%Y%m%d_%H%M")
    log_path = create_log_path("infer", f"infer_calibrated_{timestamp}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = Logger(log_path=log_path)
    
    logger.write("🌡️ 캘리브레이션 적용 고성능 추론 시작")
    logger.write(f"📋 Config: {cfg_path}")
    logger.write(f"📊 Fold results: {fold_results_path}")
    logger.write(f"🔄 TTA: {use_tta}")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.write(f"🖥️ Device: {device}")
    
    try:
        # 1. 폴드 모델들 로드
        logger.write("\n" + "="*50)
        logger.write("📦 모델 로딩 중...")
        logger.write("="*50)
        
        models = load_fold_models(fold_results_path, device)
        logger.write(f"✅ {len(models)}개 폴드 모델 로드 완료")
        
        # 2. 검증 데이터로 캘리브레이션 수행
        logger.write("\n" + "="*50)
        logger.write("🌡️ Temperature Scaling 캘리브레이션...")
        logger.write("="*50)
        
        temperature_scalings = perform_calibration(cfg, models, device, logger)
        
        # 3. 테스트 데이터 로더 생성
        logger.write("\n" + "="*50)
        logger.write("📊 테스트 데이터 준비...")
        logger.write("="*50)
        
        test_loader = create_test_loader(cfg, logger, use_tta)
        
        # 4. 캘리브레이션된 앙상블 예측 수행
        logger.write("\n" + "="*50)
        logger.write("🎯 캘리브레이션된 앙상블 예측 실행...")
        logger.write("="*50)
        
        ensemble_probs, predictions = ensemble_predict_calibrated(
            models, temperature_scalings, test_loader, device, use_tta, logger
        )
        
        # 5. 제출 파일 생성
        logger.write("\n" + "="*50)
        logger.write("💾 제출 파일 생성...")
        logger.write("="*50)
        
        submission_path = create_submission_file(
            cfg, ensemble_probs, predictions, output_path, logger
        )
        
        logger.write("✅ 캘리브레이션 적용 추론 완료!")
        logger.write(f"📄 제출 파일: {submission_path}")
        
        return submission_path
        
    except Exception as e:
        logger.write(f"❌ 추론 실패: {str(e)}")
        raise


def perform_calibration(
    cfg: dict, 
    models: List[dict], 
    device: torch.device, 
    logger: Logger
) -> List[TemperatureScaling]:
    """
    모델들에 대한 Temperature Scaling 캘리브레이션 수행
    
    Args:
        cfg: 설정 딕셔너리
        models: 로드된 모델 체크포인트 리스트
        device: 연산 디바이스
        logger: 로거
        
    Returns:
        각 모델의 TemperatureScaling 모듈 리스트
    """
    # 검증 데이터 로더 생성 (캘리브레이션용)
    valid_loader = create_validation_loader(cfg, logger)
    
    # 캘리브레이션 트레이너 생성
    calibration_trainer = CalibrationTrainer(device, logger)
    
    # 각 모델을 실제 nn.Module로 복원하고 캘리브레이션
    temperature_scalings = []
    
    for i, checkpoint in enumerate(models):
        logger.write(f"🎯 모델 {i+1}/{len(models)} 캘리브레이션 중...")
        
        # 모델 생성 및 가중치 로드
        model_name = get_recommended_model(cfg["model"]["name"])
        model = build_model(
            model_name,
            cfg["data"]["num_classes"],
            pretrained=False,
            drop_rate=cfg["model"]["drop_rate"],
            drop_path_rate=cfg["model"]["drop_path_rate"],
            pooling=cfg["model"]["pooling"]
        ).to(device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Temperature scaling 캘리브레이션
        temp_scaling = calibration_trainer.calibrate_model(model, valid_loader)
        temperature_scalings.append(temp_scaling)
        
        logger.write(f"✅ 모델 {i+1} 캘리브레이션 완료 (T={temp_scaling.get_temperature():.4f})")
    
    return temperature_scalings


def create_validation_loader(cfg: dict, logger: Logger) -> DataLoader:
    """
    캘리브레이션용 검증 데이터 로더 생성
    
    Args:
        cfg: 설정 딕셔너리
        logger: 로거
        
    Returns:
        검증 데이터 로더
    """
    # 원본 학습 데이터의 일부를 검증용으로 사용
    train_df = pd.read_csv(cfg["data"]["train_csv"])
    
    # 간단한 분할 (마지막 20%를 검증용으로)
    split_idx = int(len(train_df) * 0.8)
    valid_df = train_df.iloc[split_idx:].reset_index(drop=True)
    
    logger.write(f"📊 캘리브레이션용 검증 데이터: {len(valid_df)}개 샘플")
    
    # 검증 데이터셋 생성
    valid_ds = HighPerfDocClsDataset(
        valid_df,
        cfg["data"]["image_dir_train"],
        img_size=cfg["train"]["img_size"],
        epoch=1,  # 검증용이므로 고정
        total_epochs=1,
        is_train=False,  # 검증 모드
        id_col=cfg["data"]["id_col"],
        target_col=cfg["data"]["target_col"]
    )
    
    # 데이터 로더 생성
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["project"]["num_workers"],
        pin_memory=True
    )
    
    return valid_loader


def create_test_loader(cfg: dict, logger: Logger, use_tta: bool = True) -> DataLoader:
    """
    테스트 데이터 로더 생성 (Configurable TTA 지원)
    
    Args:
        cfg: 설정 딕셔너리
        logger: 로거
        use_tta: TTA 사용 여부
        
    Returns:
        테스트 데이터 로더
    """
    # 테스트 이미지 리스트 생성
    test_dir = Path(cfg["data"]["image_dir_test"])
    test_files = list(test_dir.glob(f"*{cfg['data']['image_ext']}"))
    
    # 테스트 데이터프레임 생성
    test_df = pd.DataFrame({
        cfg["data"]["id_col"]: [f.stem for f in test_files]
    })
    
    logger.write(f"📊 테스트 데이터: {len(test_df)}개 이미지")
    
    if use_tta and "inference" in cfg and "tta_type" in cfg["inference"]:
        # Configurable TTA 데이터셋 사용
        from src.inference.infer_highperf import ConfigurableTTADataset
        test_ds = ConfigurableTTADataset(
            None,  # csv_path 대신 DataFrame 사용
            cfg["data"]["image_dir_test"],
            img_size=cfg["train"]["img_size"],
            tta_type=cfg["inference"].get("tta_type", "essential"),
            test_df=test_df,  # DataFrame 직접 전달
            id_col=cfg["data"]["id_col"]
        )
        logger.write(f"🔄 Configurable TTA 사용: {cfg['inference']['tta_type']}")
    else:
        # 기늨 HighPerfDocClsDataset 사용
        test_ds = HighPerfDocClsDataset(
            test_df,
            cfg["data"]["image_dir_test"],
            img_size=cfg["train"]["img_size"],
            epoch=1,
            total_epochs=1,
            is_train=False,
            id_col=cfg["data"]["id_col"],
            target_col=None  # 테스트에는 타겟 없음
        )
        logger.write("📊 기본 데이터셋 사용")
    
    # 데이터 로더 생성
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["project"]["num_workers"],
        pin_memory=True
    )
    
    return test_loader


def ensemble_predict_calibrated(
    models: List[dict],
    temperature_scalings: List[TemperatureScaling],
    test_loader: DataLoader,
    device: torch.device,
    use_tta: bool,
    logger: Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    캘리브레이션된 앙상블 예측 수행
    
    Args:
        models: 모델 체크포인트 리스트
        temperature_scalings: TemperatureScaling 모듈 리스트
        test_loader: 테스트 데이터 로더
        device: 연산 디바이스
        use_tta: TTA 사용 여부
        logger: 로거
        
    Returns:
        (앙상블 확률, 예측 라벨)
    """
    from src.calibration.temperature_scaling import ensemble_predict_with_calibration
    
    # 모델들을 실제 nn.Module로 복원
    model_list = []
    
    for i, checkpoint in enumerate(models):
        # 설정에서 모델 정보 가져오기 (임시로 고정값 사용)
        model_name = "swin_base_patch4_window12_384"  # TODO: 설정에서 읽어오기
        
        model = build_model(
            model_name,
            17,  # num_classes
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.1,
            pooling="avg"
        ).to(device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model_list.append(model)
    
    # 캘리브레이션된 앙상블 예측
    logger.write(f"🔥 {len(model_list)}개 모델로 캘리브레이션된 앙상블 예측 시작...")
    
    ensemble_probs, predictions = ensemble_predict_with_calibration(
        model_list, temperature_scalings, test_loader, device
    )
    
    logger.write(f"✅ 예측 완료: {len(predictions)}개 샘플")
    
    return ensemble_probs, predictions


def create_submission_file(
    cfg: dict,
    ensemble_probs: np.ndarray,
    predictions: np.ndarray,
    output_path: Optional[str],
    logger: Logger
) -> str:
    """
    제출 파일 생성
    
    Args:
        cfg: 설정 딕셔너리
        ensemble_probs: 앙상블 확률
        predictions: 예측 라벨
        output_path: 출력 경로 (None시 자동 생성)
        logger: 로거
        
    Returns:
        생성된 파일 경로
    """
    # 테스트 이미지 리스트 다시 생성
    test_dir = Path(cfg["data"]["image_dir_test"])
    test_files = sorted(list(test_dir.glob(f"*{cfg['data']['image_ext']}")))
    test_ids = [f.stem for f in test_files]
    
    # 제출 데이터프레임 생성
    submission_df = pd.DataFrame({
        cfg["data"]["id_col"]: test_ids,
        cfg["data"]["target_col"]: predictions
    })
    
    # 출력 경로 설정
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M")
        
        # 증강 타입 결정 (학습 설정과 동일한 로직 사용)
        aug_type = "advanced_augmentation" if cfg["train"].get("use_advanced_augmentation", False) else "basic_augmentation"
        
        output_path = f"submissions/{timestamp}/submission_calibrated_{timestamp}_{aug_type}.csv"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 파일 저장
    submission_df.to_csv(output_path, index=False)
    
    logger.write(f"💾 제출 파일 저장: {output_path}")
    logger.write(f"📊 예측 분포: {np.bincount(predictions)}")
    
    return output_path


# 메인 실행 블록
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python infer_calibrated.py <config_path> <fold_results_path> [output_path]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    fold_results_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    result_path = run_calibrated_inference(
        config_path, 
        fold_results_path, 
        output_path,
        use_tta=True
    )
    
    print(f"🎉 캘리브레이션 적용 추론 완료! 결과: {result_path}")
