# src/optimization/optuna_tuner.py
"""
Optuna 하이퍼파라미터 자동 튜닝

베이스라인 노트북의 Optuna 코드를 모듈화하여 자동 하이퍼파라미터 최적화를 제공합니다.
"""

import os
import time
import copy
from typing import Dict, Any, Optional, Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Optuna import (설치 필요시 자동 설치 안내)
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("❌ Optuna가 설치되지 않았습니다.")
    print("📥 설치 명령어: pip install optuna")
    print("💡 또는 requirements.txt에 optuna 추가 후 pip install -r requirements.txt")
    raise

# 프로젝트 모듈 import
from src.utils.common import load_yaml, dump_yaml, create_log_path
from src.logging.logger import Logger
# from src.training.train_highperf import run_fold_training  # 개별 폴드 학습 함수 (향후 구현)
from .hyperopt_utils import (
    OptimizationConfig, 
    create_search_space, 
    create_study, 
    print_optimization_summary,
    update_config_with_best_params
)

# 시각화 모듈 import
from src.utils.visualizations import visualize_optimization_pipeline, create_organized_output_structure


class OptunaTrainer:
    """
    Optuna를 사용한 하이퍼파라미터 자동 최적화 클래스
    """
    
    def __init__(self, config_path: str, optimization_config: OptimizationConfig):
        """
        초기화
        
        Args:
            config_path: 기본 학습 설정 파일 경로
            optimization_config: 최적화 설정
        """
        self.config_path = config_path
        self.base_config = load_yaml(config_path)
        self.opt_config = optimization_config
        
        # 로거 설정
        timestamp = time.strftime("%Y%m%d_%H%M")
        log_path = create_log_path("optimization", f"optuna_{timestamp}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.logger = Logger(log_path=log_path)
        
        # Optuna study 생성
        self.study = create_study(optimization_config)
        
        self.logger.write("🔍 Optuna 하이퍼파라미터 튜닝 초기화 완료")
        self.logger.write(f"📋 Base config: {config_path}")
        self.logger.write(f"🎯 Target trials: {optimization_config.n_trials}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective 함수
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            최적화할 메트릭 값 (F1 score)
        """
        try:
            # 1. 하이퍼파라미터 샘플링
            sampled_params = create_search_space(trial, self.opt_config)
            
            self.logger.write(f"🔬 Trial {trial.number}: {sampled_params}")
            
            # 2. 설정 업데이트
            trial_config = copy.deepcopy(self.base_config)
            trial_config['train'].update(sampled_params)
            
            # 3. 빠른 검증을 위한 3-fold CV
            f1_scores = self._quick_cross_validation(trial_config, trial)
            
            # 4. 평균 F1 점수 계산
            mean_f1 = np.mean(f1_scores)
            
            self.logger.write(f"✅ Trial {trial.number} 완료: F1 {mean_f1:.4f} (fold scores: {f1_scores})")
            
            return mean_f1
            
        except Exception as e:
            self.logger.write(f"❌ Trial {trial.number} 실패: {str(e)}")
            # 실패한 trial은 낮은 점수 반환
            return 0.0
    
    def _quick_cross_validation(self, config: Dict[str, Any], trial: optuna.Trial) -> list:
        """
        빠른 교차 검증 (3-fold, 짧은 epoch)
        
        Args:
            config: 학습 설정
            trial: Optuna trial (조기 중단용)
            
        Returns:
            각 fold의 F1 점수 리스트
        """
        # 빠른 검증을 위한 설정 조정
        quick_config = copy.deepcopy(config)
        quick_config['train']['epochs'] = 3  # 짧은 epoch
        quick_config['data']['folds'] = 3    # 3-fold만 사용
        
        # CSV 데이터 로드
        import pandas as pd
        train_df = pd.read_csv(config['data']['train_csv'])
        
        # Stratified K-Fold 설정
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[config['data']['target_col']])):
            # 조기 중단 체크 (Optuna pruning)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            self.logger.write(f"  📁 Fold {fold_idx + 1}/3 시작...")
            
            # 개별 폴드 학습 실행 (빠른 버전)
            fold_f1 = self._train_single_fold(
                quick_config, 
                train_df.iloc[train_idx], 
                train_df.iloc[val_idx],
                fold_idx
            )
            
            fold_scores.append(fold_f1)
            
            # 중간 결과 보고 (Optuna pruning 판단용)
            trial.report(fold_f1, fold_idx)
            
            self.logger.write(f"  ✅ Fold {fold_idx + 1}/3 완료: F1 {fold_f1:.4f}")
        
        return fold_scores
    
    def _train_single_fold(self, config: Dict[str, Any], train_df, val_df, fold_idx: int) -> float:
        """
        단일 폴드 학습 (빠른 검증용)
        
        Args:
            config: 학습 설정
            train_df: 학습 데이터프레임
            val_df: 검증 데이터프레임
            fold_idx: 폴드 인덱스
            
        Returns:
            검증 F1 점수
        """
        # TODO: 실제 구현에서는 train_highperf.py의 개별 폴드 학습 함수 호출
        # 현재는 플레이스홀더로 랜덤 점수 반환 (데모용)
        
        # 실제 구현에서는 다음과 같이 호출:
        # from src.training.train_highperf import train_single_fold_quick
        # return train_single_fold_quick(config, train_df, val_df, fold_idx)
        
        # 플레이스홀더: 실제 학습 대신 시뮬레이션
        import random
        time.sleep(1)  # 학습 시간 시뮬레이션
        
        # 하이퍼파라미터에 따른 가상의 성능 계산
        lr = config['train']['lr']
        batch_size = config['train']['batch_size']
        
        # 간단한 성능 추정 공식 (실제로는 진짜 학습 결과)
        base_score = 0.85
        lr_bonus = max(0, 0.05 - abs(lr - 0.0003) * 100)  # 0.0003 근처에서 최적
        batch_bonus = 0.02 if batch_size == 64 else 0.0   # 64가 최적
        noise = random.uniform(-0.02, 0.02)               # 랜덤 노이즈
        
        simulated_f1 = base_score + lr_bonus + batch_bonus + noise
        return max(0.5, min(0.95, simulated_f1))  # 0.5~0.95 범위로 제한
    
    def optimize(self) -> Dict[str, Any]:
        """
        하이퍼파라미터 최적화 실행
        
        Returns:
            최적 파라미터 딕셔너리
        """
        self.logger.write("🚀 Optuna 최적화 시작...")
        start_time = time.time()
        
        try:
            # 최적화 실행
            self.study.optimize(
                self.objective,
                n_trials=self.opt_config.n_trials,
                timeout=self.opt_config.timeout
            )
            
            # 결과 정리
            elapsed_time = time.time() - start_time
            
            self.logger.write(f"⏱️ 최적화 완료: {elapsed_time:.1f}초 소요")
            print_optimization_summary(self.study)
            
            # 최적 파라미터 저장
            best_params_path = f"experiments/optimization/best_params_{time.strftime('%Y%m%d_%H%M')}.yaml"
            os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
            
            best_config = update_config_with_best_params(self.base_config, self.study.best_params)
            dump_yaml(best_config, best_params_path)
            
            self.logger.write(f"💾 최적 설정 저장: {best_params_path}")
            
            #-------------- 최적화 결과 시각화 ---------------------- #
            try:
                # 시각화를 위한 출력 디렉터리 설정
                viz_output_dir = os.path.dirname(best_params_path)
                model_name = self.base_config.get("model", {}).get("name", "unknown")
                
                # Study 객체 저장 (시각화용)
                import pickle
                study_path = os.path.join(viz_output_dir, f"study_{time.strftime('%Y%m%d_%H%M')}.pkl")
                with open(study_path, 'wb') as f:
                    pickle.dump(self.study, f)
                
                # 시각화 생성
                visualize_optimization_pipeline(
                    study_path=study_path,
                    model_name=model_name,
                    output_dir=viz_output_dir,
                    trials_df=None
                )
                self.logger.write(f"[VIZ] Optimization visualizations created in {viz_output_dir}")
                
            except Exception as viz_error:
                self.logger.write(f"[WARNING] Visualization failed: {str(viz_error)}")
            
            return self.study.best_params
            
        except Exception as e:
            self.logger.write(f"❌ 최적화 실패: {str(e)}")
            raise


def run_hyperparameter_optimization(
    config_path: str,
    n_trials: int = 20,
    timeout: int = 3600,
    output_path: Optional[str] = None
) -> str:
    """
    하이퍼파라미터 최적화 실행 함수
    
    Args:
        config_path: 기본 설정 파일 경로
        n_trials: 시도 횟수
        timeout: 최대 시간 (초)
        output_path: 최적 설정 파일 저장 경로
        
    Returns:
        최적화된 설정 파일 경로
    """
    # 최적화 설정 생성
    opt_config = OptimizationConfig(
        n_trials=n_trials,
        timeout=timeout
    )
    
    # Optuna 튜너 생성 및 실행
    tuner = OptunaTrainer(config_path, opt_config)
    best_params = tuner.optimize()
    
    # 최적 설정으로 새 설정 파일 생성
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M")
        output_path = f"configs/train_optimized_{timestamp}.yaml"
    
    base_config = load_yaml(config_path)
    optimized_config = update_config_with_best_params(base_config, best_params)
    dump_yaml(optimized_config, output_path)
    
    print(f"🎯 최적화 완료! 새 설정 파일: {output_path}")
    return output_path
