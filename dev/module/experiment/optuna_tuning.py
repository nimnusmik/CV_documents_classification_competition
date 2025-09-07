"""
Optuna 하이퍼파라미터 튜닝
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime

from ..config import Config, OptunaConfig
from ..models.model import create_model
from ..data.dataset import ImageDataset
from ..data.transforms import get_train_transforms, get_val_transforms
from ..training.trainer import Trainer
from ..training.validator import Validator
from ..training.loss import get_loss_function
from ..utils.seed import set_seed
from .kfold import KFoldExperiment


class OptunaHyperparameterTuner:
    """
    Optuna를 사용한 하이퍼파라미터 자동 튜닝 클래스
    
    Features:
        - 베이지안 최적화 기반 하이퍼파라미터 탐색
        - 빠른 검증을 위한 축소된 K-Fold CV
        - 조기 중단 (Pruning) 지원
        - 최적 하이퍼파라미터 저장 및 로드
        - 튜닝 과정 시각화 데이터 제공
    """
    
    def __init__(self, config: OptunaConfig):
        """
        Args:
            config: Optuna 튜닝 설정 객체
        """
        
        self.config = config
        self.study = None
        self.best_params = None
        self.tuning_results = []
        
        print("✅ Optuna Hyperparameter Tuner initialized")
        print(f"   - Max trials: {config.n_trials}")
        print(f"   - Timeout: {config.optuna_timeout}s" if config.optuna_timeout else "   - No timeout")
        print(f"   - Quick CV folds: {config.quick_cv_folds}")
        print(f"   - Quick epochs: {config.quick_epochs}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna 목적 함수
        주어진 하이퍼파라미터로 빠른 교차 검증 수행 후 성능 반환
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            float: 최적화할 목표 값 (Macro F1 Score)
        """
        
        # 하이퍼파라미터 제안
        params = self.suggest_hyperparameters(trial)
        
        print(f"\n--- Trial {trial.number + 1} ---")
        print(f"Suggested params: {params}")
        
        # 설정 업데이트
        trial_config = self.update_config_with_params(params)
        
        # 빠른 K-Fold CV 수행
        try:
            cv_score = self.quick_cross_validation(trial_config, trial)
            
            # 결과 기록
            trial_result = {
                "trial_number": trial.number,
                "params": params,
                "cv_score": cv_score,
                "status": "COMPLETE"
            }
            self.tuning_results.append(trial_result)
            
            print(f"Trial {trial.number + 1} CV Score: {cv_score:.4f}")
            
            return cv_score
            
        except optuna.TrialPruned:
            print(f"Trial {trial.number + 1} was pruned")
            raise
        except Exception as e:
            print(f"Trial {trial.number + 1} failed: {str(e)}")
            
            # 실패 기록
            trial_result = {
                "trial_number": trial.number,
                "params": params,
                "cv_score": 0.0,
                "status": "FAILED",
                "error": str(e)
            }
            self.tuning_results.append(trial_result)
            
            return 0.0
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        하이퍼파라미터 제안
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            Dict[str, Any]: 제안된 하이퍼파라미터
        """
        
        params = {
            # 학습률
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.config.lr_range[0],
                self.config.lr_range[1],
                log=True
            ),
            
            # 배치 크기
            "batch_size": trial.suggest_categorical(
                "batch_size",
                self.config.batch_size_choices
            ),
            
            # 정규화 관련
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.3),
            "mixup_prob": trial.suggest_float("mixup_prob", 0.0, 0.5),
            "mixup_alpha": trial.suggest_float("mixup_alpha", 0.5, 2.0),
            
            # 그래디언트 클리핑
            "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.5, 2.0),
            
            # 모델 관련
            "model_name": trial.suggest_categorical(
                "model_name",
                ["efficientnet_b3", "efficientnet_b4", "resnet152", "convnext_base"]
            ),
            
            # 이미지 크기 (효율성을 위해 제한적 선택)
            "img_size": trial.suggest_categorical("img_size", [224, 384]),
            
            # 증강 관련
            "rotation_prob": trial.suggest_float("rotation_prob", 0.3, 0.8),
            "blur_prob": trial.suggest_float("blur_prob", 0.7, 1.0),
            "brightness_prob": trial.suggest_float("brightness_prob", 0.5, 0.9)
        }
        
        return params
    
    def update_config_with_params(self, params: Dict[str, Any]) -> Config:
        """
        제안된 파라미터로 설정 업데이트
        
        Args:
            params: 제안된 하이퍼파라미터
            
        Returns:
            Config: 업데이트된 설정 객체
        """
        
        # 기본 설정 복사
        trial_config = Config()
        
        # 기본값 복사
        for key, value in self.config.__dict__.items():
            if hasattr(trial_config, key):
                setattr(trial_config, key, value)
        
        # 제안된 파라미터로 업데이트
        for key, value in params.items():
            if hasattr(trial_config, key):
                setattr(trial_config, key, value)
        
        # 빠른 튜닝을 위한 설정 조정
        trial_config.n_folds = self.config.quick_cv_folds
        trial_config.epochs = self.config.quick_epochs
        trial_config.num_workers = min(trial_config.num_workers, 8)  # 메모리 절약
        
        return trial_config
    
    def quick_cross_validation(self, config: Config, trial: optuna.Trial) -> float:
        """
        빠른 교차 검증 수행
        
        Args:
            config: 시험용 설정
            trial: Optuna trial (조기 중단용)
            
        Returns:
            float: 평균 CV F1 스코어
        """
        
        # 시드 고정
        set_seed(config.seed)
        
        # 가상 데이터 로드 (실제 구현에서는 실제 데이터 사용)
        train_df = self.load_training_data()
        
        # 빠른 K-Fold 실험
        experiment = KFoldExperiment(config)
        
        # 데이터 분할
        splits = experiment.prepare_data_splits(train_df)
        
        # 훈련 데이터셋 생성
        train_dataset = ImageDataset(
            image_paths=[f"{config.train_path}{img_id}" for img_id in train_df['ID']],
            targets=train_df['target'].tolist(),
            transform=get_train_transforms(config)
        )
        
        fold_f1_scores = []
        
        # 각 Fold에 대해 빠른 학습
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            print(f"Quick training Fold {fold_idx + 1}/{len(splits)}...")
            
            # 데이터로더 생성
            train_loader, val_loader = experiment.create_fold_dataloaders(
                train_dataset, train_indices, val_indices
            )
            
            # 빠른 학습 수행
            fold_f1 = self.quick_train_single_fold(config, train_loader, val_loader, trial, fold_idx)
            fold_f1_scores.append(fold_f1)
            
            # 중간 결과로 조기 중단 판단
            current_mean = np.mean(fold_f1_scores)
            trial.report(current_mean, fold_idx)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # 평균 F1 스코어 반환
        mean_f1 = float(np.mean(fold_f1_scores))
        return mean_f1
    
    def quick_train_single_fold(
        self, 
        config: Config, 
        train_loader, 
        val_loader, 
        trial: optuna.Trial,
        fold_idx: int
    ) -> float:
        """
        단일 Fold 빠른 학습
        
        Args:
            config: 설정
            train_loader: 훈련 데이터로더
            val_loader: 검증 데이터로더
            trial: Optuna trial
            fold_idx: Fold 번호
            
        Returns:
            float: 최고 검증 F1 스코어
        """
        
        # 모델, 옵티마이저, 손실함수 생성
        model = create_model(config)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        loss_fn = get_loss_function(config)
        
        trainer = Trainer(config)
        validator = Validator(config)
        
        best_val_f1 = 0.0
        
        # 빠른 학습 (적은 에포크)
        for epoch in range(config.epochs):
            # 학습
            train_results = trainer.train_one_epoch(model, train_loader, optimizer, loss_fn)
            
            # 검증
            val_results = validator.validate_one_epoch(model, val_loader, loss_fn)
            
            # 최고 성능 추적
            current_val_f1 = val_results['val_f1']
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
            
            # 중간 보고 (에포크별)
            trial.report(best_val_f1, fold_idx * config.epochs + epoch)
            
            # 조기 중단 확인
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_f1
    
    def load_training_data(self) -> pd.DataFrame:
        """
        훈련 데이터 로드
        실제 구현에서는 config.get_train_csv_path()를 사용
        
        Returns:
            pd.DataFrame: 훈련 데이터프레임
        """
        
        # 실제 구현
        try:
            return pd.read_csv(self.config.get_train_csv_path())
        except FileNotFoundError:
            # 테스트용 가상 데이터
            print("⚠️ Using fake data for testing (train.csv not found)")
            return pd.DataFrame({
                'ID': [f'img_{i:03d}.jpg' for i in range(100)],
                'target': np.random.randint(0, 17, 100)
            })
    
    def run_optimization(
        self, 
        study_name: str = None,
        direction: str = "maximize"
    ) -> Dict[str, Any]:
        """
        하이퍼파라미터 최적화 실행
        
        Args:
            study_name: Optuna study 이름
            direction: 최적화 방향 ("maximize" or "minimize")
            
        Returns:
            Dict[str, Any]: 최적화 결과
        """
        
        print(f"\n🔍 Starting hyperparameter optimization...")
        print(f"Target trials: {self.config.n_trials}")
        print(f"Timeout: {self.config.optuna_timeout}s" if self.config.optuna_timeout else "No timeout")
        
        # Study 생성
        if study_name is None:
            study_name = f"cv_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Pruner 설정 (MedianPruner 사용)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,  # 최소 3번의 trial 후 pruning 시작
            n_warmup_steps=2,    # 최소 2번의 fold 후 pruning 판단
            interval_steps=1     # 매 step마다 pruning 확인
        )
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=pruner
        )
        
        # 최적화 실행
        try:
            self.study.optimize(
                self.objective,
                n_trials=self.config.n_trials,
                timeout=self.config.optuna_timeout,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n⚠️ Optimization interrupted by user")
        
        # 결과 정리
        optimization_results = self.summarize_optimization_results()
        
        # 최적 파라미터 저장
        self.best_params = self.study.best_params
        
        return optimization_results
    
    def summarize_optimization_results(self) -> Dict[str, Any]:
        """최적화 결과 요약"""
        
        if self.study is None:
            return {"error": "No optimization study found"}
        
        # 기본 통계
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        results = {
            "study_name": self.study.study_name,
            "total_trials": len(self.study.trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial_number": self.study.best_trial.number,
            "optimization_history": [
                {"trial": t.number, "value": t.value} 
                for t in completed_trials
            ],
            "param_importance": {}
        }
        
        # 파라미터 중요도 계산 (완료된 시도가 10개 이상인 경우)
        if len(completed_trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(self.study)
                results["param_importance"] = importance
            except Exception as e:
                print(f"⚠️ Could not calculate parameter importance: {e}")
        
        # 결과 출력
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Best CV Score: {results['best_value']:.4f}")
        print(f"Best Trial: #{results['best_trial_number']}")
        print(f"Total Trials: {results['total_trials']} (Completed: {results['completed_trials']}, Pruned: {results['pruned_trials']})")
        print(f"\nBest Hyperparameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        if results['param_importance']:
            print(f"\nParameter Importance:")
            for param, importance in sorted(results['param_importance'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {param}: {importance:.3f}")
        
        return results
    
    def save_optimization_results(self, results: Dict[str, Any], filename: str = None):
        """최적화 결과 저장"""
        
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optuna_results_{timestamp}.json"
        
        result_path = output_dir / filename
        
        # 결과 저장
        save_data = {
            "optimization_results": results,
            "config": self.config.__dict__,
            "tuning_results": self.tuning_results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Optimization results saved: {result_path}")
    
    def get_best_config(self) -> Config:
        """최적 하이퍼파라미터로 설정 객체 생성"""
        
        if self.best_params is None:
            raise ValueError("No optimization results found. Run optimization first.")
        
        # 전체 학습용 설정으로 복원
        best_config = Config()
        
        # 기본값 복사
        for key, value in self.config.__dict__.items():
            if hasattr(best_config, key) and key not in ['use_optuna', 'n_trials', 'optuna_timeout', 'quick_cv_folds', 'quick_epochs']:
                setattr(best_config, key, value)
        
        # 최적 파라미터 적용
        for key, value in self.best_params.items():
            if hasattr(best_config, key):
                setattr(best_config, key, value)
        
        # 전체 학습용으로 복원
        best_config.n_folds = 5  # 원래 fold 수로 복원
        best_config.epochs = 15  # 원래 epoch 수로 복원
        
        print(f"✅ Best config created with optimized hyperparameters")
        print(f"   - CV Score: {self.study.best_value:.4f}")
        
        return best_config
    
    def create_study_visualization_data(self) -> Dict[str, Any]:
        """Study 시각화를 위한 데이터 생성"""
        
        if self.study is None:
            return {"error": "No study found"}
        
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            return {"error": "No completed trials found"}
        
        visualization_data = {
            "optimization_history": [
                {
                    "trial": t.number,
                    "value": t.value,
                    "params": t.params
                }
                for t in completed_trials
            ],
            "param_relationships": {},
            "best_trial_info": {
                "number": self.study.best_trial.number,
                "value": self.study.best_value,
                "params": self.study.best_params
            }
        }
        
        # 파라미터별 관계 데이터
        for param_name in self.study.best_params.keys():
            param_values = []
            objective_values = []
            
            for trial in completed_trials:
                if param_name in trial.params:
                    param_values.append(trial.params[param_name])
                    objective_values.append(trial.value)
            
            visualization_data["param_relationships"][param_name] = {
                "param_values": param_values,
                "objective_values": objective_values
            }
        
        return visualization_data


if __name__ == "__main__":
    # 테스트 코드
    import tempfile
    
    # 테스트용 설정
    config = OptunaConfig()
    config.device = "cpu"
    config.n_trials = 3  # 빠른 테스트
    config.quick_cv_folds = 2
    config.quick_epochs = 1
    config.optuna_timeout = 60  # 1분 제한
    config.lr_range = (1e-4, 1e-2)
    config.batch_size_choices = [8, 16]
    
    print("=== Optuna Hyperparameter Tuner Test ===")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        config.output_path = tmp_dir + "/"
        
        # 튜너 생성
        tuner = OptunaHyperparameterTuner(config)
        
        # 최적화 실행 (테스트용 짧은 시간)
        print("Testing optimization...")
        results = tuner.run_optimization(study_name="test_study")
        
        print(f"Optimization completed with {results['completed_trials']} trials")
        
        # 최적 설정 생성 테스트
        if tuner.best_params:
            best_config = tuner.get_best_config()
            print(f"Best config created: LR={best_config.learning_rate:.6f}")
        
        # 시각화 데이터 생성 테스트
        viz_data = tuner.create_study_visualization_data()
        print(f"Visualization data created with {len(viz_data.get('optimization_history', []))} points")
    
    print("✅ Optuna Hyperparameter Tuner test completed successfully")
