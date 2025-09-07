"""
TTA(Test Time Augmentation) 및 앙상블 추론 구현
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from ..config import Config
from ..data.transforms import get_tta_transforms
from .predictor import Predictor


class TTAPredictor:
    """
    단일 모델에 대한 TTA 추론 클래스
    
    Features:
        - 다양한 TTA 변환 적용
        - 적응적 TTA (고신뢰도 조기 중단)
        - 메모리 효율적 배치 처리
        - 선택적 TTA 변환
    """
    
    def __init__(self, model: nn.Module, config: Config):
        """
        Args:
            model: TTA를 적용할 모델
            config: 설정 객체
        """
        
        self.model = model
        self.config = config
        self.device = config.device
        self.predictor = Predictor(model, config)
        
        # TTA 변환 가져오기 (문서 특화)
        self.tta_transforms = get_tta_transforms(config)
        
        print("✅ TTA Predictor initialized")
        print(f"   - Device: {self.device}")
        print(f"   - TTA transforms: {len(self.tta_transforms)}")
        print(f"   - Adaptive TTA threshold: {config.confidence_threshold}")
    
    def predict_with_tta(
        self, 
        dataloader: DataLoader,
        adaptive_tta: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        TTA를 사용한 예측
        
        Args:
            dataloader: 원본 데이터로더
            adaptive_tta: 적응적 TTA 사용 여부
            show_progress: 진행률 표시 여부
            
        Returns:
            np.ndarray: TTA 평균 확률 (N, num_classes)
        """
        
        self.model.eval()
        
        # 원본 데이터셋 가져오기
        dataset = dataloader.dataset
        batch_size = self.config.tta_batch_size  # TTA 전용 배치 크기
        
        all_predictions = []
        
        with torch.no_grad():
            # 각 샘플에 대해 TTA 수행
            for batch_start in tqdm(
                range(0, len(dataset), batch_size), 
                desc="TTA Prediction",
                disable=not show_progress
            ):
                batch_end = min(batch_start + batch_size, len(dataset))
                batch_indices = range(batch_start, batch_end)
                
                # 배치별 TTA 예측
                batch_tta_probs = []
                
                for transform_idx, tta_transform in enumerate(self.tta_transforms):
                    # 변환된 배치 생성
                    transformed_batch = []
                    for idx in batch_indices:
                        if isinstance(dataset[idx], (list, tuple)):
                            image, _ = dataset[idx]
                        else:
                            image = dataset[idx]
                        
                        # PIL 이미지를 텐서로 변환하고 TTA 적용
                        if hasattr(image, 'mode'):  # PIL 이미지인 경우
                            image = tta_transform(image)
                        else:  # 이미 텐서인 경우
                            image = tta_transform(image)
                        
                        transformed_batch.append(image)
                    
                    # 배치 텐서로 변환
                    batch_tensor = torch.stack(transformed_batch).to(self.device)
                    
                    # 모델 추론
                    outputs = self.model(batch_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    
                    batch_tta_probs.append(probs)
                    
                    # 적응적 TTA: 첫 번째 변환 후 신뢰도 확인
                    if adaptive_tta and transform_idx == 0:
                        max_probs = np.max(probs, axis=1)
                        high_confidence_mask = max_probs >= self.config.confidence_threshold
                        
                        # 모든 샘플이 고신뢰도인 경우 조기 중단
                        if np.all(high_confidence_mask):
                            print(f"Early stopping at transform {transform_idx + 1} (high confidence)")
                            break
                
                # 배치 TTA 결과 평균
                batch_final_probs = np.mean(batch_tta_probs, axis=0)
                all_predictions.append(batch_final_probs)
        
        # 전체 결과 결합
        final_predictions = np.concatenate(all_predictions, axis=0)
        
        return final_predictions
    
    def predict_with_selective_tta(
        self, 
        dataloader: DataLoader,
        core_transforms_only: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        선택적 TTA (핵심 변환만 사용)
        
        Args:
            dataloader: 데이터로더
            core_transforms_only: 핵심 변환만 사용할지 여부
            show_progress: 진행률 표시 여부
            
        Returns:
            np.ndarray: 선택적 TTA 확률
        """
        
        if core_transforms_only:
            # 핵심 변환만 사용 (속도 우선)
            core_indices = [0, 1, 2, 3, 4]  # 원본 + 90도 회전 3개 + 밝기 보정 1개
            selected_transforms = [
                self.tta_transforms[i] for i in core_indices 
                if i < len(self.tta_transforms)
            ]
        else:
            selected_transforms = self.tta_transforms
        
        # 임시로 변환 교체
        original_transforms = self.tta_transforms
        self.tta_transforms = selected_transforms
        
        try:
            result = self.predict_with_tta(dataloader, adaptive_tta=False, show_progress=show_progress)
        finally:
            # 원본 변환 복구
            self.tta_transforms = original_transforms
        
        return result
    
    def get_tta_info(self) -> Dict[str, Any]:
        """TTA 설정 정보 반환"""
        
        return {
            "num_transforms": len(self.tta_transforms),
            "transform_names": [
                transform.__class__.__name__ 
                for transform in self.tta_transforms
            ],
            "adaptive_tta_enabled": True,
            "confidence_threshold": self.config.confidence_threshold,
            "tta_batch_size": self.config.tta_batch_size
        }


class EnsembleTTAPredictor:
    """
    다중 모델 앙상블 + TTA 추론 클래스
    
    Features:
        - 여러 모델 앙상블
        - 각 모델에 TTA 적용
        - 가중 평균 앙상블
        - 메모리 효율적 처리
    """
    
    def __init__(
        self, 
        models: List[nn.Module], 
        config: Config,
        model_weights: Optional[List[float]] = None
    ):
        """
        Args:
            models: 앙상블할 모델 리스트
            config: 설정 객체
            model_weights: 모델별 가중치 (None이면 균등 가중치)
        """
        
        self.models = models
        self.config = config
        self.device = config.device
        
        # 모델 가중치 설정
        if model_weights is None:
            self.model_weights = [1.0 / len(models)] * len(models)
        else:
            if len(model_weights) != len(models):
                raise ValueError("model_weights 길이가 models와 일치하지 않습니다")
            # 가중치 정규화
            total_weight = sum(model_weights)
            self.model_weights = [w / total_weight for w in model_weights]
        
        # 각 모델용 TTA Predictor 생성
        self.tta_predictors = [
            TTAPredictor(model, config) for model in models
        ]
        
        print("✅ Ensemble TTA Predictor initialized")
        print(f"   - Number of models: {len(models)}")
        print(f"   - Model weights: {[f'{w:.3f}' for w in self.model_weights]}")
        print(f"   - Device: {self.device}")
    
    def predict_ensemble_tta(
        self, 
        dataloader: DataLoader,
        use_adaptive_tta: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        앙상블 + TTA 예측
        
        Args:
            dataloader: 데이터로더
            use_adaptive_tta: 적응적 TTA 사용 여부
            show_progress: 진행률 표시 여부
            
        Returns:
            np.ndarray: 앙상블 TTA 확률 (N, num_classes)
        """
        
        all_model_predictions = []
        
        # 각 모델에 대해 TTA 예측 수행
        for model_idx, tta_predictor in enumerate(self.tta_predictors):
            if show_progress:
                print(f"\n--- Model {model_idx + 1}/{len(self.models)} TTA Prediction ---")
            
            # 모델별 TTA 예측
            model_probs = tta_predictor.predict_with_tta(
                dataloader, 
                adaptive_tta=use_adaptive_tta,
                show_progress=show_progress
            )
            
            all_model_predictions.append(model_probs)
            
            if show_progress:
                mean_confidence = np.mean(np.max(model_probs, axis=1))
                print(f"Model {model_idx + 1} mean confidence: {mean_confidence:.4f}")
        
        # 가중 평균 앙상블
        ensemble_probs = np.zeros_like(all_model_predictions[0])
        
        for model_probs, weight in zip(all_model_predictions, self.model_weights):
            ensemble_probs += weight * model_probs
        
        return ensemble_probs
    
    def predict_with_model_analysis(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """
        모델별 분석과 함께 앙상블 예측
        
        Args:
            dataloader: 데이터로더
            
        Returns:
            Dict[str, Any]: 상세 분석 결과
        """
        
        # 각 모델 예측 수행
        model_results = []
        
        for model_idx, tta_predictor in enumerate(self.tta_predictors):
            print(f"\nAnalyzing Model {model_idx + 1}...")
            
            model_probs = tta_predictor.predict_with_tta(dataloader, adaptive_tta=True)
            model_preds = np.argmax(model_probs, axis=1)
            model_confidences = np.max(model_probs, axis=1)
            
            model_info = {
                "model_id": model_idx + 1,
                "predictions": model_preds,
                "probabilities": model_probs,
                "mean_confidence": float(np.mean(model_confidences)),
                "std_confidence": float(np.std(model_confidences)),
                "high_confidence_ratio": float(np.mean(model_confidences >= self.config.confidence_threshold)),
                "weight": self.model_weights[model_idx]
            }
            
            model_results.append(model_info)
        
        # 앙상블 예측
        ensemble_probs = self.predict_ensemble_tta(dataloader, show_progress=False)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        ensemble_confidences = np.max(ensemble_probs, axis=1)
        
        # 모델 간 일치도 분석
        model_preds_array = np.array([result["predictions"] for result in model_results])
        agreement_ratio = np.mean(
            np.all(model_preds_array == model_preds_array[0], axis=0)
        )
        
        return {
            "model_results": model_results,
            "ensemble_predictions": ensemble_preds,
            "ensemble_probabilities": ensemble_probs,
            "ensemble_mean_confidence": float(np.mean(ensemble_confidences)),
            "model_agreement_ratio": float(agreement_ratio),
            "total_samples": len(ensemble_preds)
        }
    
    def save_ensemble_predictions(
        self, 
        predictions: np.ndarray, 
        file_path: str,
        sample_ids: Optional[List[str]] = None
    ):
        """앙상블 예측 결과 저장"""
        
        # 첫 번째 모델의 predictor를 사용해서 저장
        self.tta_predictors[0].predictor.save_predictions(predictions, file_path, sample_ids)
        
        print(f"📊 Ensemble prediction summary:")
        print(f"   - Models used: {len(self.models)}")
        print(f"   - Model weights: {[f'{w:.3f}' for w in self.model_weights]}")
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """앙상블 설정 정보 반환"""
        
        return {
            "num_models": len(self.models),
            "model_weights": self.model_weights,
            "total_tta_transforms": len(self.tta_predictors[0].tta_transforms),
            "confidence_threshold": self.config.confidence_threshold,
            "tta_batch_size": self.config.tta_batch_size,
            "device": self.device
        }


if __name__ == "__main__":
    # 테스트 코드
    from ..config import Config
    from ..models.model import create_model
    from torch.utils.data import TensorDataset, DataLoader
    
    config = Config()
    config.device = "cpu"  # 테스트용
    config.model_name = "efficientnet_b0"
    config.batch_size = 4
    config.tta_batch_size = 8
    config.confidence_threshold = 0.9
    
    print("=== TTA Predictor Test ===")
    
    # 단일 모델 TTA 테스트
    model = create_model(config)
    tta_predictor = TTAPredictor(model, config)
    
    # 가상 데이터 생성
    fake_images = torch.randn(12, 3, 224, 224)
    fake_dataset = TensorDataset(fake_images)
    test_loader = DataLoader(fake_dataset, batch_size=4, shuffle=False)
    
    print("Testing single model TTA...")
    tta_probs = tta_predictor.predict_with_tta(test_loader, adaptive_tta=True)
    print(f"TTA predictions shape: {tta_probs.shape}")
    
    # TTA 정보 출력
    tta_info = tta_predictor.get_tta_info()
    print(f"TTA info: {tta_info}")
    
    print("\n=== Ensemble TTA Predictor Test ===")
    
    # 다중 모델 앙상블 테스트
    models = [create_model(config) for _ in range(3)]
    model_weights = [0.4, 0.35, 0.25]  # 가중 앙상블
    
    ensemble_predictor = EnsembleTTAPredictor(models, config, model_weights)
    
    print("Testing ensemble TTA...")
    ensemble_probs = ensemble_predictor.predict_ensemble_tta(test_loader, use_adaptive_tta=True)
    print(f"Ensemble predictions shape: {ensemble_probs.shape}")
    
    # 상세 분석 테스트
    print("\nTesting detailed analysis...")
    detailed_results = ensemble_predictor.predict_with_model_analysis(test_loader)
    print(f"Model agreement ratio: {detailed_results['model_agreement_ratio']:.4f}")
    
    # 앙상블 정보 출력
    ensemble_info = ensemble_predictor.get_ensemble_info()
    print(f"Ensemble info: {ensemble_info}")
    
    print("✅ TTA and Ensemble tests completed successfully")
