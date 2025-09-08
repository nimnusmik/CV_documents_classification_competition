# src/calibration/temperature_scaling.py
"""
Temperature Scaling 캘리브레이션

베이스라인 노트북의 Temperature Scaling을 모듈화하여 
모델의 확률 예측을 보정합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional
import numpy as np
from tqdm import tqdm

from src.logging.logger import Logger


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling 모듈
    
    베이스라인에서 사용된 캘리브레이션 기법을 구현합니다.
    모델의 로짓을 temperature로 나누어 확률을 보정합니다.
    """
    
    def __init__(self, temperature: float = 1.5):
        """
        초기화
        
        Args:
            temperature: 초기 temperature 값
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Temperature scaling 적용
        
        Args:
            logits: 모델의 로짓 출력 [batch_size, num_classes]
            
        Returns:
            Temperature로 스케일된 로짓 [batch_size, num_classes]
        """
        return logits / self.temperature
    
    def get_temperature(self) -> float:
        """현재 temperature 값 반환"""
        return self.temperature.item()


class CalibrationTrainer:
    """
    Temperature Scaling 캘리브레이션 학습기
    
    검증 데이터를 사용해서 최적의 temperature를 찾습니다.
    """
    
    def __init__(self, device: torch.device, logger: Optional[Logger] = None):
        """
        초기화
        
        Args:
            device: 연산 디바이스
            logger: 로거 (선택적)
        """
        self.device = device
        self.logger = logger if logger else Logger()
    
    def calibrate_model(
        self, 
        model: nn.Module, 
        valid_loader: DataLoader,
        max_iter: int = 50,
        lr: float = 0.01
    ) -> TemperatureScaling:
        """
        단일 모델에 대한 temperature scaling 캘리브레이션
        
        Args:
            model: 캘리브레이션할 모델
            valid_loader: 검증 데이터 로더
            max_iter: 최대 반복 횟수
            lr: 학습률
            
        Returns:
            학습된 TemperatureScaling 모듈
        """
        self.logger.write("🌡️ Temperature Scaling 캘리브레이션 시작...")
        
        # Temperature scaling 모듈 생성
        temperature_scaling = TemperatureScaling().to(self.device)
        
        # 최적화 설정
        optimizer = torch.optim.LBFGS(
            temperature_scaling.parameters(), 
            lr=lr, 
            max_iter=max_iter
        )
        
        # 검증 데이터에서 로짓과 라벨 수집
        model.eval()
        all_logits = []
        all_labels = []
        
        self.logger.write("📊 검증 데이터에서 로짓 수집 중...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(valid_loader, desc="Collecting logits")):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = model(images)
                all_logits.append(logits)
                all_labels.append(labels)
        
        # 텐서 결합
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        self.logger.write(f"📈 수집된 샘플 수: {len(all_logits)}")
        
        # Temperature 최적화
        def eval_loss():
            optimizer.zero_grad()
            
            # Temperature scaling 적용
            scaled_logits = temperature_scaling(all_logits)
            
            # Cross entropy loss 계산
            loss = F.cross_entropy(scaled_logits, all_labels)
            loss.backward()
            
            return loss
        
        # 최적화 실행
        initial_temp = temperature_scaling.get_temperature()
        self.logger.write(f"🎯 초기 temperature: {initial_temp:.4f}")
        
        optimizer.step(eval_loss)
        
        final_temp = temperature_scaling.get_temperature()
        self.logger.write(f"✅ 최적화 완료! 최종 temperature: {final_temp:.4f}")
        
        # 캘리브레이션 효과 평가
        initial_ece = self._calculate_ece(all_logits, all_labels)
        calibrated_logits = temperature_scaling(all_logits)
        final_ece = self._calculate_ece(calibrated_logits, all_labels)
        
        self.logger.write(f"📊 ECE 개선: {initial_ece:.4f} → {final_ece:.4f}")
        
        return temperature_scaling
    
    def calibrate_ensemble(
        self, 
        models: List[nn.Module], 
        valid_loader: DataLoader
    ) -> List[TemperatureScaling]:
        """
        앙상블 모델들에 대한 개별 캘리브레이션
        
        Args:
            models: 캘리브레이션할 모델 리스트
            valid_loader: 검증 데이터 로더
            
        Returns:
            각 모델에 대한 TemperatureScaling 모듈 리스트
        """
        self.logger.write(f"🔥 앙상블 {len(models)}개 모델 캘리브레이션 시작...")
        
        calibrated_modules = []
        
        for i, model in enumerate(models):
            self.logger.write(f"🎯 모델 {i+1}/{len(models)} 캘리브레이션 중...")
            
            temp_scaling = self.calibrate_model(model, valid_loader)
            calibrated_modules.append(temp_scaling)
        
        self.logger.write("✅ 앙상블 캘리브레이션 완료!")
        return calibrated_modules
    
    def _calculate_ece(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        n_bins: int = 15
    ) -> float:
        """
        Expected Calibration Error (ECE) 계산
        
        Args:
            logits: 모델 로짓
            labels: 실제 라벨
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


def apply_temperature_scaling(
    logits: torch.Tensor, 
    temperature_scaling: TemperatureScaling
) -> torch.Tensor:
    """
    로짓에 temperature scaling 적용
    
    Args:
        logits: 원본 로짓
        temperature_scaling: 학습된 TemperatureScaling 모듈
        
    Returns:
        캘리브레이션된 로짓
    """
    return temperature_scaling(logits)


def ensemble_predict_with_calibration(
    models: List[nn.Module],
    temperature_scalings: List[TemperatureScaling],
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    캘리브레이션이 적용된 앙상블 예측
    
    Args:
        models: 모델 리스트
        temperature_scalings: 각 모델의 TemperatureScaling 모듈
        data_loader: 데이터 로더
        device: 연산 디바이스
        
    Returns:
        (앙상블 확률, 예측 라벨)
    """
    all_ensemble_probs = []
    
    # 모델들을 평가 모드로 설정
    for model in models:
        model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc="Ensemble prediction")):
            images = images.to(device)
            batch_size = images.size(0)
            
            # 앙상블 확률 초기화
            ensemble_probs = torch.zeros(batch_size, 17).to(device)  # 17개 클래스
            
            # 각 모델에서 예측
            for model, temp_scaling in zip(models, temperature_scalings):
                # 로짓 예측
                logits = model(images)
                
                # Temperature scaling 적용
                calibrated_logits = temp_scaling(logits)
                
                # 확률 계산
                probs = F.softmax(calibrated_logits, dim=1)
                
                # 앙상블에 추가
                ensemble_probs += probs
            
            # 평균 계산
            ensemble_probs /= len(models)
            
            all_ensemble_probs.append(ensemble_probs.cpu().numpy())
    
    # 결과 결합
    all_ensemble_probs = np.concatenate(all_ensemble_probs, axis=0)
    predictions = np.argmax(all_ensemble_probs, axis=1)
    
    return all_ensemble_probs, predictions
