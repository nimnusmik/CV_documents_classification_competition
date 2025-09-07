"""
앙상블 모델 관리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
import numpy as np

from ..config import Config
from .model import create_model


class EnsembleModel(nn.Module):
    """
    여러 모델을 앙상블하는 클래스
    
    Features:
        - 다중 모델 병렬 추론
        - 가중 평균 지원
        - 메모리 효율적 처리
        - 다양한 앙상블 전략
    """
    
    def __init__(
        self, 
        models: List[nn.Module], 
        weights: Optional[List[float]] = None,
        strategy: str = "average"
    ):
        """
        Args:
            models: 앙상블할 모델들의 리스트
            weights: 각 모델의 가중치 (None이면 균등 가중)
            strategy: 앙상블 전략 ("average", "weighted", "voting")
        """
        
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        
        # 가중치 설정
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "weights와 models의 길이가 일치해야 합니다"
            # 정규화
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        # 모든 모델을 평가 모드로 설정
        for model in self.models:
            model.eval()
        
        print(f"✅ Ensemble created with {len(models)} models")
        print(f"   - Strategy: {strategy}")
        print(f"   - Weights: {self.weights}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        앙상블 추론 수행
        
        Args:
            x: 입력 텐서
            
        Returns:
            torch.Tensor: 앙상블 결과
        """
        
        # 모든 모델의 예측 수집
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        # 전략에 따른 앙상블
        if self.strategy == "average":
            # 단순 평균
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        
        elif self.strategy == "weighted":
            # 가중 평균
            weighted_preds = []
            for pred, weight in zip(predictions, self.weights):
                weighted_preds.append(pred * weight)
            ensemble_pred = torch.sum(torch.stack(weighted_preds), dim=0)
        
        elif self.strategy == "voting":
            # 하드 보팅 (argmax 기반)
            votes = []
            for pred in predictions:
                vote = torch.argmax(pred, dim=1)
                votes.append(vote)
            
            # 가장 많이 선택된 클래스 선택
            votes_tensor = torch.stack(votes, dim=1)  # (batch_size, num_models)
            ensemble_pred = torch.mode(votes_tensor, dim=1)[0]
            
            # 원-핫 인코딩으로 변환
            batch_size, num_classes = predictions[0].shape
            ensemble_one_hot = torch.zeros(batch_size, num_classes, device=x.device)
            ensemble_one_hot.scatter_(1, ensemble_pred.unsqueeze(1), 1)
            ensemble_pred = ensemble_one_hot
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return ensemble_pred
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        소프트맥스 확률 반환
        
        Args:
            x: 입력 텐서
            
        Returns:
            torch.Tensor: 확률 분포
        """
        
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        클래스 예측 반환
        
        Args:
            x: 입력 텐서
            
        Returns:
            torch.Tensor: 예측 클래스
        """
        
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        각 개별 모델의 예측 반환 (디버깅용)
        
        Args:
            x: 입력 텐서
            
        Returns:
            List[torch.Tensor]: 각 모델의 예측 리스트
        """
        
        predictions = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                pred = model(x)
                predictions.append({
                    "model_idx": i,
                    "logits": pred,
                    "probs": F.softmax(pred, dim=1),
                    "predictions": torch.argmax(pred, dim=1)
                })
        
        return predictions


class KFoldEnsemble:
    """
    K-Fold 모델들을 관리하는 앙상블 클래스
    """
    
    def __init__(self, fold_models: List[Dict[str, Any]], config: Config):
        """
        Args:
            fold_models: K-Fold 훈련 결과 [{"state_dict": ..., "score": ...}, ...]
            config: 설정 객체
        """
        
        self.config = config
        self.fold_models = fold_models
        self.models = []
        
        # 각 fold 모델 로드
        for i, fold_result in enumerate(fold_models):
            model = create_model(config, device=config.device)
            model.load_state_dict(fold_result["state_dict"])
            model.eval()
            self.models.append(model)
            
            score = fold_result.get("score", "N/A")
            print(f"Fold {i+1} model loaded (score: {score})")
        
        # 앙상블 모델 생성
        self.ensemble = EnsembleModel(self.models, strategy="average")
    
    def predict(self, dataloader) -> np.ndarray:
        """
        앙상블 예측 수행
        
        Args:
            dataloader: 데이터로더
            
        Returns:
            np.ndarray: 예측 결과
        """
        
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                if isinstance(images, list):
                    # TTA의 경우 각 변형별로 처리
                    batch_ensemble_probs = torch.zeros(
                        images[0].size(0), 
                        self.config.num_classes
                    ).to(self.config.device)
                    
                    for model in self.models:
                        for img_transforms in images:
                            img_transforms = img_transforms.to(self.config.device)
                            preds = model(img_transforms)
                            probs = F.softmax(preds, dim=1)
                            batch_ensemble_probs += probs / (len(self.models) * len(images))
                    
                    batch_preds = torch.argmax(batch_ensemble_probs, dim=1)
                
                else:
                    # 일반 추론
                    images = images.to(self.config.device)
                    batch_preds = self.ensemble.predict(images)
                
                all_predictions.extend(batch_preds.cpu().numpy())
        
        return np.array(all_predictions)
    
    def get_fold_scores(self) -> List[float]:
        """각 fold의 성능 점수 반환"""
        return [fold.get("score", 0.0) for fold in self.fold_models]
    
    def get_best_fold_idx(self) -> int:
        """최고 성능 fold 인덱스 반환"""
        scores = self.get_fold_scores()
        return np.argmax(scores)
    
    def get_worst_fold_idx(self) -> int:
        """최저 성능 fold 인덱스 반환"""
        scores = self.get_fold_scores()
        return np.argmin(scores)


def create_ensemble_from_folds(
    fold_results: List[Dict[str, Any]], 
    config: Config,
    use_top_k: Optional[int] = None,
    weight_by_performance: bool = False
) -> EnsembleModel:
    """
    K-Fold 결과로부터 앙상블 생성
    
    Args:
        fold_results: Fold 결과 리스트
        config: 설정 객체  
        use_top_k: 상위 K개 모델만 사용 (None이면 전체 사용)
        weight_by_performance: 성능에 따른 가중치 적용 여부
        
    Returns:
        EnsembleModel: 생성된 앙상블 모델
    """
    
    # 성능 기준 정렬 (내림차순)
    if "score" in fold_results[0]:
        sorted_results = sorted(fold_results, key=lambda x: x["score"], reverse=True)
    else:
        sorted_results = fold_results
    
    # 상위 K개만 선택
    if use_top_k is not None:
        selected_results = sorted_results[:use_top_k]
        print(f"Using top {use_top_k} models out of {len(fold_results)}")
    else:
        selected_results = sorted_results
    
    # 모델 생성
    models = []
    scores = []
    
    for i, fold_result in enumerate(selected_results):
        model = create_model(config, device=config.device)
        model.load_state_dict(fold_result["state_dict"])
        model.eval()
        models.append(model)
        
        score = fold_result.get("score", 1.0)
        scores.append(score)
        print(f"Fold model {i+1} loaded (score: {score:.4f})")
    
    # 가중치 계산
    weights = None
    if weight_by_performance and len(scores) > 1:
        # 성능에 비례한 가중치 계산
        min_score = min(scores)
        adjusted_scores = [score - min_score + 0.1 for score in scores]  # 최소값 보정
        total_score = sum(adjusted_scores)
        weights = [score / total_score for score in adjusted_scores]
        print(f"Performance-based weights: {weights}")
    
    # 앙상블 생성
    ensemble = EnsembleModel(
        models=models, 
        weights=weights, 
        strategy="weighted" if weights else "average"
    )
    
    return ensemble


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for calibration
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Temperature scaling 적용
        
        Args:
            logits: 모델 출력 (softmax 전)
            
        Returns:
            torch.Tensor: Temperature scaling 적용된 logits
        """
        return logits / self.temperature
    
    def calibrate(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        검증 데이터로 temperature 보정
        
        Args:
            logits: 검증 데이터의 logits
            targets: 검증 데이터의 정답
            
        Returns:
            float: 보정된 temperature 값
        """
        
        # NLL Loss로 temperature 최적화
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            loss = criterion(self.forward(logits), targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        return self.temperature.item()


if __name__ == "__main__":
    # 테스트 코드
    from ..config import Config
    
    config = Config()
    config.model_name = "efficientnet_b0"  # 테스트용
    config.device = "cpu"
    config.num_classes = 17
    
    print("=== Ensemble Model Test ===")
    
    # 가상의 모델들 생성
    models = []
    for i in range(3):
        model = create_model(config)
        models.append(model)
    
    # 앙상블 생성
    ensemble = EnsembleModel(models, strategy="average")
    
    # 테스트 입력
    test_input = torch.randn(2, 3, 224, 224)  # 배치크기 2
    
    # 앙상블 추론
    ensemble_output = ensemble(test_input)
    individual_preds = ensemble.get_individual_predictions(test_input)
    
    print(f"Ensemble output shape: {ensemble_output.shape}")
    print(f"Individual predictions count: {len(individual_preds)}")
    
    # 확률 및 예측
    probs = ensemble.predict_proba(test_input)
    preds = ensemble.predict(test_input)
    
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions: {preds}")
    
    print("✅ Ensemble test completed successfully")
    
    # Temperature Scaling 테스트
    print("\n=== Temperature Scaling Test ===")
    
    temp_scaling = TemperatureScaling()
    
    # 가상 logits와 targets
    logits = torch.randn(10, 17)
    targets = torch.randint(0, 17, (10,))
    
    # Calibration
    calibrated_temp = temp_scaling.calibrate(logits, targets)
    calibrated_logits = temp_scaling(logits)
    
    print(f"Original temperature: 1.5")
    print(f"Calibrated temperature: {calibrated_temp:.4f}")
    print(f"Calibrated logits shape: {calibrated_logits.shape}")
    
    print("✅ Temperature scaling test completed successfully")
