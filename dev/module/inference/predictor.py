"""
단일 모델 추론 클래스
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional, Dict, Any

from ..config import Config


class Predictor:
    """
    단일 모델 추론을 담당하는 클래스
    
    Features:
        - 배치 추론
        - 메모리 효율성
        - 진행률 표시
        - 유연한 출력 형식
    """
    
    def __init__(self, model: nn.Module, config: Config):
        """
        Args:
            model: 추론할 모델
            config: 설정 객체
        """
        
        self.model = model
        self.config = config
        self.device = config.device
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        self.model.to(self.device)
        
        print("✅ Predictor initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Model: {self.model.__class__.__name__}")
    
    def predict(
        self, 
        dataloader: DataLoader, 
        return_probs: bool = False,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        데이터로더를 사용한 배치 추론
        
        Args:
            dataloader: 추론할 데이터로더
            return_probs: True면 확률 반환, False면 클래스 인덱스 반환
            show_progress: 진행률 표시 여부
            
        Returns:
            np.ndarray: 예측 결과
                - return_probs=True: shape (N, num_classes) 확률
                - return_probs=False: shape (N,) 클래스 인덱스
        """
        
        self.model.eval()
        
        all_predictions = []
        
        with torch.no_grad():
            if show_progress:
                dataloader = tqdm(dataloader, desc="Predicting")
            
            for batch in dataloader:
                # 이미지만 있는 경우와 (이미지, 라벨)이 있는 경우 모두 처리
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                if return_probs:
                    # Softmax 확률로 변환
                    probs = torch.softmax(outputs, dim=1)
                    all_predictions.append(probs.cpu().numpy())
                else:
                    # 클래스 인덱스
                    preds = torch.argmax(outputs, dim=1)
                    all_predictions.append(preds.cpu().numpy())
        
        # 결과 결합
        return np.concatenate(all_predictions, axis=0)
    
    def predict_single_image(
        self, 
        image: torch.Tensor, 
        return_probs: bool = False
    ) -> np.ndarray:
        """
        단일 이미지 추론
        
        Args:
            image: 입력 이미지 텐서 shape (C, H, W)
            return_probs: 확률 반환 여부
            
        Returns:
            np.ndarray: 예측 결과
        """
        
        self.model.eval()
        
        # 배치 차원 추가
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            
            if return_probs:
                probs = torch.softmax(output, dim=1)
                return probs.squeeze().cpu().numpy()
            else:
                pred = torch.argmax(output, dim=1)
                return pred.item()
    
    def predict_with_confidence(
        self, 
        dataloader: DataLoader,
        confidence_threshold: float = 0.9,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        신뢰도와 함께 예측 수행
        
        Args:
            dataloader: 데이터로더
            confidence_threshold: 높은 신뢰도 임계값
            show_progress: 진행률 표시 여부
            
        Returns:
            Dict[str, np.ndarray]: 
                - predictions: 예측 클래스
                - probabilities: 예측 확률
                - confidences: 최대 확률 (신뢰도)
                - high_confidence_mask: 높은 신뢰도 마스크
        """
        
        # 확률 형태로 예측
        probs = self.predict(dataloader, return_probs=True, show_progress=show_progress)
        
        # 예측 클래스와 신뢰도 계산
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        high_confidence_mask = confidences >= confidence_threshold
        
        return {
            "predictions": predictions,
            "probabilities": probs,
            "confidences": confidences,
            "high_confidence_mask": high_confidence_mask,
            "high_confidence_ratio": np.mean(high_confidence_mask)
        }
    
    def get_prediction_stats(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """
        예측 통계 정보 반환
        
        Args:
            dataloader: 데이터로더
            
        Returns:
            Dict[str, Any]: 예측 통계
        """
        
        result = self.predict_with_confidence(dataloader)
        
        stats = {
            "total_samples": len(result["predictions"]),
            "num_classes": self.config.num_classes,
            "mean_confidence": float(np.mean(result["confidences"])),
            "std_confidence": float(np.std(result["confidences"])),
            "min_confidence": float(np.min(result["confidences"])),
            "max_confidence": float(np.max(result["confidences"])),
            "high_confidence_ratio": result["high_confidence_ratio"],
            "class_distribution": {
                f"class_{i}": int(np.sum(result["predictions"] == i)) 
                for i in range(self.config.num_classes)
            }
        }
        
        return stats
    
    def save_predictions(
        self, 
        predictions: np.ndarray, 
        file_path: str,
        sample_ids: Optional[List[str]] = None
    ):
        """
        예측 결과를 CSV로 저장
        
        Args:
            predictions: 예측 결과 (클래스 인덱스)
            file_path: 저장할 파일 경로
            sample_ids: 샘플 ID 목록 (None이면 자동 생성)
        """
        
        import pandas as pd
        
        if sample_ids is None:
            sample_ids = [f"sample_{i:05d}" for i in range(len(predictions))]
        
        df = pd.DataFrame({
            "ID": sample_ids,
            "target": predictions.astype(int)
        })
        
        df.to_csv(file_path, index=False)
        
        print(f"✅ Predictions saved to {file_path}")
        print(f"   - Total samples: {len(predictions)}")
        print(f"   - Unique classes: {len(np.unique(predictions))}")


if __name__ == "__main__":
    # 테스트 코드
    from ..config import Config
    from ..models.model import create_model
    from torch.utils.data import TensorDataset, DataLoader
    
    config = Config()
    config.device = "cpu"  # 테스트용
    config.model_name = "efficientnet_b0"
    config.batch_size = 4
    
    print("=== Predictor Test ===")
    
    # 모델 생성
    model = create_model(config)
    predictor = Predictor(model, config)
    
    # 가상 테스트 데이터 생성
    fake_images = torch.randn(20, 3, 224, 224)
    fake_dataset = TensorDataset(fake_images)
    test_loader = DataLoader(fake_dataset, batch_size=4, shuffle=False)
    
    # 클래스 예측 테스트
    print("Testing class predictions...")
    class_preds = predictor.predict(test_loader, return_probs=False)
    print(f"Class predictions shape: {class_preds.shape}")
    print(f"Class predictions: {class_preds}")
    
    # 확률 예측 테스트
    print("\nTesting probability predictions...")
    prob_preds = predictor.predict(test_loader, return_probs=True)
    print(f"Probability predictions shape: {prob_preds.shape}")
    
    # 신뢰도 예측 테스트
    print("\nTesting confidence predictions...")
    conf_results = predictor.predict_with_confidence(test_loader)
    print(f"High confidence ratio: {conf_results['high_confidence_ratio']:.2f}")
    
    # 예측 통계 테스트
    print("\nTesting prediction statistics...")
    stats = predictor.get_prediction_stats(test_loader)
    print(f"Mean confidence: {stats['mean_confidence']:.4f}")
    print(f"Class distribution: {stats['class_distribution']}")
    
    # 단일 이미지 예측 테스트
    print("\nTesting single image prediction...")
    single_image = torch.randn(3, 224, 224)
    single_pred = predictor.predict_single_image(single_image)
    print(f"Single image prediction: {single_pred}")
    
    # CSV 저장 테스트
    print("\nTesting CSV save...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "test_predictions.csv")
        predictor.save_predictions(class_preds, csv_path)
        
        # 저장된 파일 확인
        import pandas as pd
        saved_df = pd.read_csv(csv_path)
        print(f"Saved CSV shape: {saved_df.shape}")
        print(f"Saved CSV columns: {saved_df.columns.tolist()}")
    
    print("✅ Predictor test completed successfully")
