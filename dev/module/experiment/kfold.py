"""
K-Fold Cross Validation 실험 관리
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

from ..config import Config
from ..models.model import create_model
from ..data.dataset import ImageDataset
from ..data.transforms import get_train_transforms, get_val_transforms
from ..training.trainer import Trainer
from ..training.validator import Validator
from ..training.loss import get_loss_function
from ..training.scheduler import get_scheduler
from ..utils.seed import set_seed


class KFoldExperiment:
    """
    K-Fold Cross Validation 실험을 관리하는 클래스
    
    Features:
        - Stratified K-Fold 분할
        - 각 Fold별 모델 학습 및 검증
        - 교차 검증 결과 집계
        - 모델 저장 및 로드
        - 실험 재현성 보장
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: 실험 설정 객체
        """
        
        self.config = config
        self.trainer = Trainer(config)
        self.validator = Validator(config)
        
        # 결과 저장용
        self.fold_results = []
        self.fold_models = []
        
        print("✅ K-Fold Experiment initialized")
        print(f"   - Number of folds: {config.n_folds}")
        print(f"   - Random state: {config.fold_random_state}")
        print(f"   - Device: {config.device}")
    
    def prepare_data_splits(self, train_df: pd.DataFrame) -> List[Tuple[List[int], List[int]]]:
        """
        Stratified K-Fold 데이터 분할
        
        Args:
            train_df: 훈련 데이터프레임 (ID, target 컬럼 필요)
            
        Returns:
            List[Tuple[List[int], List[int]]]: (train_indices, val_indices) 튜플들의 리스트
        """
        
        # Stratified K-Fold 설정
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.fold_random_state
        )
        
        # 인덱스와 타겟 추출
        indices = np.arange(len(train_df))
        targets = train_df['target'].values
        
        # 분할 수행
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, targets)):
            splits.append((train_idx.tolist(), val_idx.tolist()))
            
            print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
            
            # 클래스 분포 확인
            train_targets = targets[train_idx]
            val_targets = targets[val_idx]
            print(f"  Train classes: {np.bincount(train_targets)}")
            print(f"  Val classes: {np.bincount(val_targets)}")
        
        return splits
    
    def create_fold_dataloaders(
        self, 
        train_dataset: ImageDataset,
        train_indices: List[int],
        val_indices: List[int]
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Fold별 데이터로더 생성
        
        Args:
            train_dataset: 전체 훈련 데이터셋
            train_indices: 훈련용 인덱스
            val_indices: 검증용 인덱스
            
        Returns:
            Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
        """
        
        # 서브셋 생성
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        # 검증용 변환을 사용하는 검증 데이터셋 생성
        val_dataset_transforms = ImageDataset(
            train_dataset.image_paths, 
            train_dataset.targets,
            transform=get_val_transforms(self.config)
        )
        val_subset_with_transforms = Subset(val_dataset_transforms, val_indices)
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset_with_transforms,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_single_fold(
        self,
        fold_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        단일 Fold 학습 수행
        
        Args:
            fold_idx: Fold 번호 (0부터 시작)
            train_loader: 훈련 데이터로더
            val_loader: 검증 데이터로더
            save_model: 모델 저장 여부
            
        Returns:
            Dict[str, Any]: Fold 학습 결과
        """
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{self.config.n_folds} TRAINING")
        print(f"{'='*60}")
        
        # 시드 고정 (재현성)
        set_seed(self.config.seed + fold_idx)
        
        # 모델, 옵티마이저, 스케줄러, 손실함수 생성
        model = create_model(self.config)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        scheduler = get_scheduler(optimizer, self.config)
        loss_fn = get_loss_function(self.config)
        
        # 최고 성능 추적
        best_val_f1 = 0.0
        best_model_state = None
        fold_history = []
        
        # 에포크별 학습
        for epoch in range(self.config.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.epochs} ---")
            
            # 학습
            train_results = self.trainer.train_one_epoch(
                model, train_loader, optimizer, loss_fn, self.config.device
            )
            
            # 검증
            val_results = self.validator.validate_one_epoch(
                model, val_loader, loss_fn, self.config.device
            )
            
            # 스케줄러 업데이트
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_results['val_loss'])
                else:
                    scheduler.step()
            
            # 에포크 결과 기록
            epoch_result = {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]['lr'],
                **train_results,
                **val_results
            }
            fold_history.append(epoch_result)
            
            # 최고 성능 모델 저장
            current_val_f1 = val_results['val_f1']
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                best_model_state = model.state_dict().copy()
            
            # 진행 상황 출력
            print(f"Train Loss: {train_results['train_loss']:.4f} | "
                  f"Train F1: {train_results['train_f1']:.4f}")
            print(f"Val Loss: {val_results['val_loss']:.4f} | "
                  f"Val F1: {current_val_f1:.4f} (Best: {best_val_f1:.4f})")
        
        # 최고 성능 모델로 복원
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 모델 저장
        if save_model:
            self.save_fold_model(model, fold_idx, best_val_f1)
        
        # Fold 결과 정리
        fold_result = {
            "fold": fold_idx + 1,
            "best_val_f1": best_val_f1,
            "final_train_loss": fold_history[-1]['train_loss'],
            "final_val_loss": fold_history[-1]['val_loss'],
            "epochs_trained": len(fold_history),
            "history": fold_history,
            "model_state_dict": best_model_state
        }
        
        return fold_result
    
    def run_cross_validation(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """
        전체 K-Fold Cross Validation 실행
        
        Args:
            train_df: 훈련 데이터프레임
            
        Returns:
            Dict[str, Any]: 교차 검증 결과
        """
        
        print(f"\n🚀 Starting {self.config.n_folds}-Fold Cross Validation")
        print(f"Dataset size: {len(train_df)}")
        
        # 데이터 분할
        splits = self.prepare_data_splits(train_df)
        
        # 전체 데이터셋 생성 (훈련용 변환 적용)
        train_dataset = ImageDataset(
            image_paths=[f"{self.config.train_path}{img_id}" for img_id in train_df['ID']],
            targets=train_df['target'].tolist(),
            transform=get_train_transforms(self.config)
        )
        
        # 각 Fold 학습
        fold_results = []
        fold_models = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            # Fold별 데이터로더 생성
            train_loader, val_loader = self.create_fold_dataloaders(
                train_dataset, train_indices, val_indices
            )
            
            # Fold 학습 수행
            fold_result = self.train_single_fold(
                fold_idx, train_loader, val_loader, save_model=True
            )
            
            fold_results.append(fold_result)
            
            # 모델 저장 (메모리에)
            model = create_model(self.config)
            model.load_state_dict(fold_result['model_state_dict'])
            fold_models.append(model)
            
            print(f"\n✅ Fold {fold_idx + 1} completed - Best Val F1: {fold_result['best_val_f1']:.4f}")
        
        # 교차 검증 결과 집계
        cv_results = self.aggregate_cv_results(fold_results)
        
        # 결과 저장
        self.fold_results = fold_results
        self.fold_models = fold_models
        
        return cv_results
    
    def aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        교차 검증 결과 집계
        
        Args:
            fold_results: 각 Fold 결과 리스트
            
        Returns:
            Dict[str, Any]: 집계된 결과
        """
        
        # F1 스코어 집계
        f1_scores = [result['best_val_f1'] for result in fold_results]
        
        cv_summary = {
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "min_f1": float(np.min(f1_scores)),
            "max_f1": float(np.max(f1_scores)),
            "fold_f1_scores": f1_scores,
            "best_fold": int(np.argmax(f1_scores)) + 1,
            "worst_fold": int(np.argmin(f1_scores)) + 1,
            "cv_score": f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
            "n_folds": self.config.n_folds,
            "total_epochs": sum(result['epochs_trained'] for result in fold_results)
        }
        
        print(f"\n{'='*60}")
        print(f"CROSS VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"Mean F1: {cv_summary['mean_f1']:.4f} ± {cv_summary['std_f1']:.4f}")
        print(f"Min F1: {cv_summary['min_f1']:.4f} (Fold {cv_summary['worst_fold']})")
        print(f"Max F1: {cv_summary['max_f1']:.4f} (Fold {cv_summary['best_fold']})")
        print(f"Total epochs: {cv_summary['total_epochs']}")
        
        return cv_summary
    
    def save_fold_model(self, model: nn.Module, fold_idx: int, val_f1: float):
        """
        Fold 모델 저장
        
        Args:
            model: 저장할 모델
            fold_idx: Fold 번호
            val_f1: 검증 F1 스코어
        """
        
        # 출력 디렉토리 생성
        output_dir = Path(self.config.output_path) / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 파일 경로
        model_path = output_dir / f"fold_{fold_idx + 1}_f1_{val_f1:.4f}.pth"
        
        # 모델 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': fold_idx + 1,
            'val_f1': val_f1,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        
        print(f"💾 Fold {fold_idx + 1} model saved: {model_path}")
    
    def save_experiment_results(self, cv_results: Dict[str, Any], filename: str = None):
        """
        실험 결과 저장
        
        Args:
            cv_results: 교차 검증 결과
            filename: 저장할 파일명 (None이면 자동 생성)
        """
        
        # 출력 디렉토리 생성
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cv_results_{timestamp}.json"
        
        result_path = output_dir / filename
        
        # 결과 저장 (model_state_dict 제외)
        save_results = {
            "cv_summary": cv_results,
            "config": self.config.__dict__,
            "fold_results": [
                {k: v for k, v in fold_result.items() if k != 'model_state_dict'}
                for fold_result in self.fold_results
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Experiment results saved: {result_path}")
    
    def load_fold_models(self, model_dir: str) -> List[nn.Module]:
        """
        저장된 Fold 모델들 로드
        
        Args:
            model_dir: 모델이 저장된 디렉토리
            
        Returns:
            List[nn.Module]: 로드된 모델 리스트
        """
        
        model_dir = Path(model_dir)
        model_files = sorted(model_dir.glob("fold_*.pth"))
        
        loaded_models = []
        
        for model_file in model_files:
            # 체크포인트 로드
            checkpoint = torch.load(model_file, map_location=self.config.device)
            
            # 모델 생성 및 상태 로드
            model = create_model(self.config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.config.device)
            model.eval()
            
            loaded_models.append(model)
            
            print(f"📂 Loaded Fold {checkpoint['fold']} model (F1: {checkpoint['val_f1']:.4f})")
        
        print(f"✅ Total {len(loaded_models)} fold models loaded")
        
        return loaded_models
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """실험 요약 정보 반환"""
        
        if not self.fold_results:
            return {"status": "No experiment results available"}
        
        f1_scores = [result['best_val_f1'] for result in self.fold_results]
        
        return {
            "n_folds": len(self.fold_results),
            "mean_cv_f1": float(np.mean(f1_scores)),
            "std_cv_f1": float(np.std(f1_scores)),
            "best_fold_f1": float(np.max(f1_scores)),
            "worst_fold_f1": float(np.min(f1_scores)),
            "total_models": len(self.fold_models),
            "config_summary": {
                "model_name": self.config.model_name,
                "img_size": self.config.img_size,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate
            }
        }


if __name__ == "__main__":
    # 테스트 코드
    import tempfile
    import os
    
    # 테스트용 설정
    config = Config()
    config.device = "cpu"
    config.model_name = "efficientnet_b0"
    config.n_folds = 3  # 빠른 테스트를 위해 3-fold
    config.epochs = 2   # 빠른 테스트를 위해 2 epochs
    config.batch_size = 4
    config.num_workers = 0  # 테스트 시 멀티프로세싱 비활성화
    
    print("=== K-Fold Experiment Test ===")
    
    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as tmp_dir:
        config.output_path = tmp_dir + "/"
        
        # 가상 훈련 데이터 생성
        fake_train_df = pd.DataFrame({
            'ID': [f'img_{i:03d}.jpg' for i in range(60)],
            'target': np.random.randint(0, 17, 60)  # 17개 클래스
        })
        
        # K-Fold 실험 초기화
        experiment = KFoldExperiment(config)
        
        print("Testing data splits...")
        splits = experiment.prepare_data_splits(fake_train_df)
        print(f"Created {len(splits)} splits")
        
        # 실험 요약 테스트
        summary = experiment.get_experiment_summary()
        print(f"Experiment summary: {summary}")
        
    print("✅ K-Fold Experiment test completed successfully")
