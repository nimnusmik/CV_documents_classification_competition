# src/calibration/calibration_utils.py
"""
캘리브레이션 유틸리티 함수들

ECE 계산, 캘리브레이션 평가 등의 도우미 함수들을 제공합니다.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from .temperature_scaling import TemperatureScaling


def calculate_ece(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE) 계산
    
    Args:
        logits: 모델 로짓 [N, num_classes]
        labels: 실제 라벨 [N]
        n_bins: 빈 개수
        
    Returns:
        ECE 값
    """
    # Softmax 확률 계산
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    # 빈 생성
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 현재 빈에 속하는 샘플들
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def calibrate_model_ensemble(
    models: List[torch.nn.Module],
    valid_loader,
    device: torch.device
) -> List[TemperatureScaling]:
    """
    앙상블 모델들을 캘리브레이션
    
    Args:
        models: 모델 리스트
        valid_loader: 검증 데이터 로더
        device: 연산 디바이스
        
    Returns:
        각 모델의 TemperatureScaling 모듈 리스트
    """
    from .temperature_scaling import CalibrationTrainer
    
    trainer = CalibrationTrainer(device)
    return trainer.calibrate_ensemble(models, valid_loader)


def apply_calibration_to_predictions(
    predictions: np.ndarray,
    temperature: float
) -> np.ndarray:
    """
    예측 결과에 temperature scaling 적용
    
    Args:
        predictions: 원본 예측 확률 [N, num_classes]
        temperature: Temperature 값
        
    Returns:
        캘리브레이션된 예측 확률
    """
    # numpy array를 torch tensor로 변환
    logits = torch.from_numpy(predictions)
    
    # Temperature scaling 적용
    calibrated_logits = logits / temperature
    
    # Softmax로 확률 계산
    calibrated_probs = F.softmax(calibrated_logits, dim=1)
    
    return calibrated_probs.numpy()


def evaluate_calibration_quality(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature_scaling: TemperatureScaling
) -> dict:
    """
    캘리브레이션 품질 평가
    
    Args:
        logits: 원본 로짓
        labels: 실제 라벨
        temperature_scaling: TemperatureScaling 모듈
        
    Returns:
        평가 메트릭 딕셔너리
    """
    # 원본 ECE
    original_ece = calculate_ece(logits, labels)
    
    # 캘리브레이션 후 ECE
    calibrated_logits = temperature_scaling(logits)
    calibrated_ece = calculate_ece(calibrated_logits, labels)
    
    # 정확도 (변하지 않아야 함)
    original_acc = F.softmax(logits, dim=1).argmax(dim=1).eq(labels).float().mean().item()
    calibrated_acc = F.softmax(calibrated_logits, dim=1).argmax(dim=1).eq(labels).float().mean().item()
    
    return {
        'original_ece': original_ece,
        'calibrated_ece': calibrated_ece,
        'ece_improvement': original_ece - calibrated_ece,
        'original_accuracy': original_acc,
        'calibrated_accuracy': calibrated_acc,
        'temperature': temperature_scaling.get_temperature()
    }
