# src/inference/infer_highperf.py
"""
고성능 추론 파이프라인
- Swin Transformer & ConvNext 지원
- Test Time Augmentation (TTA)
- 앙상블 예측
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

from src.utils.common import load_yaml, resolve_path, require_file, require_dir
from src.utils.logger import Logger
from src.data.dataset import HighPerfDocClsDataset
from src.models.build import build_model, get_recommended_model


@torch.no_grad()
def predict_with_tta(model, loader, device, num_tta=5):
    """Test Time Augmentation을 사용한 예측"""
    model.eval()
    all_preds = []
    
    for _ in range(num_tta):
        batch_preds = []
        for imgs, _ in tqdm(loader, desc="TTA Inference"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            batch_preds.append(probs.cpu())
        
        # 배치 결합
        tta_preds = torch.cat(batch_preds, dim=0)
        all_preds.append(tta_preds)
    
    # TTA 평균
    final_preds = torch.stack(all_preds).mean(dim=0)
    return final_preds


def load_fold_models(fold_results_path, device):
    """폴드별 학습된 모델들 로드"""
    fold_results = load_yaml(fold_results_path)
    models = []
    
    for fold_info in fold_results["fold_results"]:
        model_path = fold_info["model_path"]
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # 모델 구조는 별도로 생성 후 가중치만 로드
            models.append(checkpoint)
        else:
            print(f"Warning: Model not found: {model_path}")
    
    return models


def ensemble_predict(models, test_loader, cfg, device, use_tta=True):
    """앙상블 예측"""
    all_ensemble_preds = []
    
    for i, checkpoint in enumerate(models):
        print(f"Processing model {i+1}/{len(models)}...")
        
        # 모델 생성 및 가중치 로드
        model_name = get_recommended_model(cfg["model"]["name"])
        model = build_model(
            model_name,
            cfg["data"]["num_classes"],
            pretrained=False,  # 가중치는 체크포인트에서 로드
            drop_rate=cfg["model"]["drop_rate"],
            drop_path_rate=cfg["model"]["drop_path_rate"],
            pooling=cfg["model"]["pooling"]
        ).to(device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # 예측
        if use_tta:
            preds = predict_with_tta(model, test_loader, device, num_tta=3)
        else:
            model.eval()
            batch_preds = []
            for imgs, _ in tqdm(test_loader, desc=f"Model {i+1} Inference"):
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = F.softmax(logits, dim=1)
                batch_preds.append(probs.cpu())
            preds = torch.cat(batch_preds, dim=0)
        
        all_ensemble_preds.append(preds)
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
    
    # 앙상블 평균
    ensemble_preds = torch.stack(all_ensemble_preds).mean(dim=0)
    return ensemble_preds


def run_highperf_inference(cfg_path: str, fold_results_path: str, output_path: Optional[str] = None):
    """고성능 추론 파이프라인 실행"""
    # 설정 로드
    cfg = load_yaml(cfg_path)
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    
    # 로거 설정
    logger = Logger(
        log_path=f"logs/infer/infer_highperf_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.log"
    )
    logger.write("[BOOT] high-performance inference pipeline started")
    
    try:
        # 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.write(f"[BOOT] device={device}")
        
        # 경로 확인
        sample_csv = resolve_path(cfg_dir, cfg["data"]["sample_csv"])
        test_dir = resolve_path(cfg_dir, cfg["data"]["image_dir_test"])
        require_file(sample_csv, "sample_csv 확인")
        require_dir(test_dir, "test_dir 확인")
        
        # 테스트 데이터 로드
        test_df = pd.read_csv(sample_csv)
        logger.write(f"[DATA] loaded test data | shape={test_df.shape}")
        
        # 테스트 데이터셋 생성
        test_ds = HighPerfDocClsDataset(
            test_df,
            test_dir,
            img_size=cfg["train"]["img_size"],
            is_train=False,
            id_col=cfg["data"]["id_col"],
            target_col=None,  # 추론 모드
            logger=logger
        )
        
        # 테스트 데이터로더
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["project"]["num_workers"],
            pin_memory=True
        )
        
        logger.write(f"[DATA] test dataset size: {len(test_ds)}")
        
        # 모델 앙상블 예측
        logger.write(f"[INFERENCE] starting ensemble prediction...")
        
        # 폴드별 모델 로드 및 예측
        models = load_fold_models(fold_results_path, device)
        ensemble_preds = ensemble_predict(models, test_loader, cfg, device, use_tta=True)
        
        # 최종 예측 클래스
        final_predictions = ensemble_preds.argmax(dim=1).numpy()
        
        # 결과 저장
        if output_path is None:
            output_path = f"submissions/{pd.Timestamp.now().strftime('%Y%m%d')}/highperf_ensemble.csv"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 제출 파일 생성
        submission = test_df.copy()
        submission[cfg["data"]["target_col"]] = final_predictions
        submission.to_csv(output_path, index=False)
        
        logger.write(f"[SUCCESS] Inference completed | output: {output_path}")
        logger.write(f"[RESULT] Prediction distribution:")
        for i, count in enumerate(np.bincount(final_predictions)):
            logger.write(f"  Class {i}: {count} samples ({count/len(final_predictions)*100:.1f}%)")
        
        return output_path
        
    except Exception as e:
        logger.write(f"[ERROR] Inference failed: {str(e)}")
        raise
    finally:
        logger.write("[SHUTDOWN] Inference pipeline ended")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python infer_highperf.py <config_path> <fold_results_path> [output_path]")
        sys.exit(1)
    
    cfg_path = sys.argv[1]
    fold_results_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    run_highperf_inference(cfg_path, fold_results_path, output_path)
