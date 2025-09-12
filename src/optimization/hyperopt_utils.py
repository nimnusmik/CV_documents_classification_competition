# src/optimization/hyperopt_utils.py
"""
하이퍼파라미터 최적화 유틸리티

Optuna 최적화를 위한 설정 및 도우미 함수들을 제공합니다.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Union
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


@dataclass
class OptimizationConfig:
    """최적화 설정 클래스"""
    n_trials: int = 10                          # 시도 횟수
    timeout: int = 3600                         # 최대 시간 (초)
    study_name: str = "doc-classification"      # 스터디 이름
    direction: str = "maximize"                 # 최적화 방향 (maximize/minimize)
    pruner_patience: int = 5                    # 조기 중단 patience
    sampler_n_startup_trials: int = 10          # 초기 랜덤 탐색 횟수
    
    # 탐색할 하이퍼파라미터 범위
    lr_range: List[float] = None                # 학습률 범위 [min, max]
    batch_size_choices: List[int] = None        # 배치 크기 선택지
    weight_decay_range: List[float] = None      # weight decay 범위 [min, max]
    dropout_range: List[float] = None           # dropout 범위 [min, max]
    
    def __post_init__(self):
        # 기본값 설정
        if self.lr_range is None:
            self.lr_range = [1e-5, 1e-2]
        if self.batch_size_choices is None:
            self.batch_size_choices = [16, 32, 64, 128]
        if self.weight_decay_range is None:
            self.weight_decay_range = [0.0, 0.1]
        if self.dropout_range is None:
            self.dropout_range = [0.0, 0.3]


def create_search_space(trial: optuna.Trial, config: OptimizationConfig) -> Dict[str, Any]:
    """
    Optuna trial을 사용해서 하이퍼파라미터 탐색 공간을 생성
    
    Args:
        trial: Optuna trial 객체
        config: 최적화 설정
        
    Returns:
        샘플링된 하이퍼파라미터 딕셔너리
    """
    params = {}
    
    # 학습률 (로그 스케일)
    params['lr'] = trial.suggest_float(
        'lr', 
        config.lr_range[0], 
        config.lr_range[1],
        log=True
    )
    
    # 배치 크기 (카테고리)
    params['batch_size'] = trial.suggest_categorical(
        'batch_size', 
        config.batch_size_choices
    )
    
    # Weight decay (균등 분포)
    params['weight_decay'] = trial.suggest_float(
        'weight_decay',
        config.weight_decay_range[0],
        config.weight_decay_range[1]
    )
    
    # Dropout (균등 분포) 
    params['dropout'] = trial.suggest_float(
        'dropout',
        config.dropout_range[0], 
        config.dropout_range[1]
    )
    
    return params


def create_study(config: OptimizationConfig) -> optuna.Study:
    """
    Optuna Study 객체 생성
    
    Args:
        config: 최적화 설정
        
    Returns:
        설정된 optuna.Study 객체
    """
    # Pruner 설정 (성능이 안 좋은 trial 조기 중단)
    pruner = MedianPruner(
        n_startup_trials=config.sampler_n_startup_trials,
        n_warmup_steps=config.pruner_patience
    )
    
    # Sampler 설정 (TPE: Tree-structured Parzen Estimator)
    sampler = TPESampler(
        n_startup_trials=config.sampler_n_startup_trials
    )
    
    # Study 생성
    study = optuna.create_study(
        study_name=config.study_name,
        direction=config.direction,
        pruner=pruner,
        sampler=sampler
    )
    
    return study


def print_optimization_summary(study: optuna.Study) -> None:
    """
    최적화 결과 요약 출력
    
    Args:
        study: 완료된 optuna.Study 객체
    """
    print("=" * 60)
    print("🎯 Optuna 하이퍼파라미터 최적화 완료!")
    print("=" * 60)
    
    print(f"📊 총 시도 횟수: {len(study.trials)}")
    print(f"🏆 최고 성능: {study.best_value:.4f}")
    print(f"⚙️ 최적 파라미터:")
    
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.6f}")
        else:
            print(f"   - {key}: {value}")
    
    print("=" * 60)


def update_config_with_best_params(config: Dict[str, Any], best_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    최적 파라미터로 설정 딕셔너리 업데이트
    
    Args:
        config: 원본 설정 딕셔너리
        best_params: Optuna에서 찾은 최적 파라미터
        
    Returns:
        업데이트된 설정 딕셔너리
    """
    updated_config = config.copy()
    
    # train 섹션 업데이트
    if 'train' not in updated_config:
        updated_config['train'] = {}
    
    # 최적 파라미터 적용
    for key, value in best_params.items():
        updated_config['train'][key] = value
    
    return updated_config
