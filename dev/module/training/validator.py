"""
검증 루프 관리
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any, Optional, List

from ..utils.metrics import calculate_metrics, get_class_wise_f1, get_confusion_matrix_stats
from ..config import Config


class Validator:
    """
    검증 루프를 관리하는 클래스
    
    Features:
        - 메모리 효율적 검증
        - 다양한 평가 지표 계산
        - 클래스별 성능 분석
        - 혼동 행렬 분석
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        
        self.config = config
        
        print("✅ Validator initialized")
        print(f"   - Device: {config.device}")
        print(f"   - Num classes: {config.num_classes}")
    
    def validate_one_epoch(
        self,
        model: nn.Module,
        dataloader,
        loss_fn,
        device: Optional[str] = None
    ) -> Dict[str, float]:
        """
        한 에포크 검증 수행
        
        Args:
            model: 검증할 모델
            dataloader: 검증 데이터로더
            loss_fn: 손실 함수
            device: 디바이스 (None이면 config.device 사용)
            
        Returns:
            Dict[str, float]: 검증 결과 메트릭
        """
        
        if device is None:
            device = self.config.device
        
        model.eval()  # 평가 모드로 전환
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():  # 그래디언트 계산 비활성화
            pbar = tqdm(dataloader, desc="Validating")
            
            for images, targets in pbar:
                images = images.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                
                # 메트릭 수집
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # 진행률 업데이트
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_loss / (len(pbar.desc) + 1):.4f}"
                })
        
        # 결과 계산
        avg_loss = total_loss / len(dataloader)
        metrics = calculate_metrics(all_targets, all_preds)
        
        result = {
            "val_loss": avg_loss,
            "val_accuracy": metrics["accuracy"],
            "val_f1": metrics["macro_f1"],  # 대회 메인 지표
            "val_weighted_f1": metrics["weighted_f1"],
            "val_precision": metrics["macro_precision"],
            "val_recall": metrics["macro_recall"]
        }
        
        return result
    
    def validate_with_detailed_analysis(
        self,
        model: nn.Module,
        dataloader,
        loss_fn,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        상세 분석이 포함된 검증
        
        Args:
            model: 모델
            dataloader: 데이터로더
            loss_fn: 손실 함수
            device: 디바이스
            
        Returns:
            Dict[str, Any]: 상세 검증 결과
        """
        
        # 기본 검증 수행
        basic_results = self.validate_one_epoch(model, dataloader, loss_fn, device)
        
        if device is None:
            device = self.config.device
        
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 클래스별 F1 스코어
        class_f1 = get_class_wise_f1(all_targets, all_preds, self.config.num_classes)
        
        # 혼동 행렬 분석
        cm_stats = get_confusion_matrix_stats(all_targets, all_preds)
        
        # 신뢰도 분석
        confidence_stats = self._analyze_confidence(all_probs, all_preds, all_targets)
        
        detailed_results = {
            **basic_results,
            "class_f1_scores": class_f1,
            "confusion_matrix_stats": cm_stats,
            "confidence_analysis": confidence_stats
        }
        
        return detailed_results
    
    def _analyze_confidence(
        self, 
        all_probs: List[List[float]], 
        all_preds: List[int], 
        all_targets: List[int]
    ) -> Dict[str, float]:
        """
        모델 신뢰도 분석
        
        Args:
            all_probs: 예측 확률들
            all_preds: 예측 클래스들
            all_targets: 실제 클래스들
            
        Returns:
            Dict[str, float]: 신뢰도 분석 결과
        """
        
        import numpy as np
        
        probs_array = np.array(all_probs)
        preds_array = np.array(all_preds)
        targets_array = np.array(all_targets)
        
        # 최대 확률 (신뢰도)
        max_probs = np.max(probs_array, axis=1)
        
        # 정답 예측에 대한 신뢰도
        correct_mask = (preds_array == targets_array)
        correct_confidences = max_probs[correct_mask]
        wrong_confidences = max_probs[~correct_mask]
        
        stats = {
            "mean_confidence": float(np.mean(max_probs)),
            "std_confidence": float(np.std(max_probs)),
            "mean_correct_confidence": float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
            "mean_wrong_confidence": float(np.mean(wrong_confidences)) if len(wrong_confidences) > 0 else 0.0,
            "high_confidence_correct": float(np.sum((max_probs > 0.9) & correct_mask) / np.sum(correct_mask)) if np.sum(correct_mask) > 0 else 0.0,
            "low_confidence_wrong": float(np.sum((max_probs < 0.5) & ~correct_mask) / np.sum(~correct_mask)) if np.sum(~correct_mask) > 0 else 0.0
        }
        
        return stats
    
    def cross_validate(
        self,
        models: List[nn.Module],
        dataloaders: List,
        loss_fn,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        교차 검증 수행
        
        Args:
            models: 검증할 모델들
            dataloaders: 검증 데이터로더들
            loss_fn: 손실 함수
            device: 디바이스
            
        Returns:
            Dict[str, Any]: 교차 검증 결과
        """
        
        if len(models) != len(dataloaders):
            raise ValueError("models와 dataloaders의 길이가 일치해야 합니다")
        
        fold_results = []
        
        for fold_idx, (model, dataloader) in enumerate(zip(models, dataloaders)):
            print(f"\n=== Fold {fold_idx + 1}/{len(models)} Validation ===")
            
            fold_result = self.validate_one_epoch(model, dataloader, loss_fn, device)
            fold_result["fold"] = fold_idx + 1
            fold_results.append(fold_result)
            
            print(f"Fold {fold_idx + 1} - Val F1: {fold_result['val_f1']:.4f}")
        
        # 전체 결과 통계
        import numpy as np
        
        f1_scores = [result["val_f1"] for result in fold_results]
        accuracies = [result["val_accuracy"] for result in fold_results]
        losses = [result["val_loss"] for result in fold_results]
        
        cv_results = {
            "fold_results": fold_results,
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "best_fold": int(np.argmax(f1_scores)) + 1,
            "worst_fold": int(np.argmin(f1_scores)) + 1
        }
        
        return cv_results
    
    def print_validation_summary(self, results: Dict[str, Any]):
        """검증 결과 요약 출력"""
        
        if "fold_results" in results:
            # 교차 검증 결과
            print("\n" + "="*60)
            print("CROSS VALIDATION RESULTS")
            print("="*60)
            
            for fold_result in results["fold_results"]:
                fold = fold_result["fold"]
                f1 = fold_result["val_f1"]
                acc = fold_result["val_accuracy"]
                loss = fold_result["val_loss"]
                print(f"Fold {fold}: F1={f1:.4f} | Acc={acc:.4f} | Loss={loss:.4f}")
            
            print(f"\nMean F1: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
            print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
            print(f"Best Fold: {results['best_fold']}")
            print(f"Worst Fold: {results['worst_fold']}")
        
        else:
            # 단일 검증 결과
            print("\n" + "="*40)
            print("VALIDATION RESULTS")
            print("="*40)
            print(f"Loss: {results['val_loss']:.4f}")
            print(f"Accuracy: {results['val_accuracy']:.4f}")
            print(f"Macro F1: {results['val_f1']:.4f}")
            print(f"Weighted F1: {results['val_weighted_f1']:.4f}")
            print(f"Precision: {results['val_precision']:.4f}")
            print(f"Recall: {results['val_recall']:.4f}")
            
            # 상세 분석 결과가 있는 경우
            if "confidence_analysis" in results:
                conf = results["confidence_analysis"]
                print(f"\nConfidence Analysis:")
                print(f"  Mean Confidence: {conf['mean_confidence']:.4f}")
                print(f"  Correct Predictions Confidence: {conf['mean_correct_confidence']:.4f}")
                print(f"  Wrong Predictions Confidence: {conf['mean_wrong_confidence']:.4f}")


if __name__ == "__main__":
    # 테스트 코드
    from ..config import Config
    from ..models.model import create_model
    import torch.optim as optim
    
    config = Config()
    config.device = "cpu"  # 테스트용
    config.model_name = "efficientnet_b0"
    config.batch_size = 4
    
    print("=== Validator Test ===")
    
    # 모델 생성
    model = create_model(config)
    loss_fn = nn.CrossEntropyLoss()
    
    # 가상 데이터 생성
    from torch.utils.data import TensorDataset, DataLoader
    
    fake_images = torch.randn(16, 3, 224, 224)
    fake_targets = torch.randint(0, 17, (16,))
    fake_dataset = TensorDataset(fake_images, fake_targets)
    fake_loader = DataLoader(fake_dataset, batch_size=4, shuffle=False)
    
    # 검증자 생성 및 검증 테스트
    validator = Validator(config)
    
    # 기본 검증
    val_results = validator.validate_one_epoch(model, fake_loader, loss_fn)
    print("Basic validation results:", val_results)
    
    # 상세 검증
    detailed_results = validator.validate_with_detailed_analysis(model, fake_loader, loss_fn)
    print("Detailed validation completed")
    
    # 결과 요약 출력
    validator.print_validation_summary(detailed_results)
    
    print("✅ Validator test completed successfully")
