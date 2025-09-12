#!/usr/bin/env python3
"""
단일 폴드 Optuna 최적화 테스트 스크립트
"""

import os
import sys
sys.path.append('/home/ieyeppo/AI_Lab/computer-vision-competition-1SEN')

from .optuna_tuner import OptunaTrainer
from .hyperopt_utils import OptimizationConfig
from ..utils.core.common import load_yaml

def test_single_fold_optuna():
    """단일 폴드 Optuna 최적화 테스트"""
    
    print("🧪 단일 폴드 Optuna 최적화 테스트 시작...")
    
    try:
        # 설정 파일 로드 (프로젝트 루트 기준)
        config_path = "../../configs/train_highperf.yaml"
        optuna_config_path = "../../configs/optuna_single_fold_config.yaml"
        
        print(f"📋 기본 설정: {config_path}")
        print(f"🔧 Optuna 설정: {optuna_config_path}")
        
        # Optuna 설정 로드
        optuna_config_dict = load_yaml(optuna_config_path)
        opt_config = OptimizationConfig.from_dict(optuna_config_dict['optuna'])
        
        # OptunaTrainer 초기화
        trainer = OptunaTrainer(config_path, opt_config)
        
        print("✅ OptunaTrainer 초기화 완료")
        
        # 테스트용으로 1번만 최적화 실행
        print("🚀 테스트 최적화 실행 (1 trial)...")
        
        # trial 수를 1로 제한
        opt_config.n_trials = 1
        trainer.opt_config.n_trials = 1
        
        # 최적화 실행
        best_params = trainer.optimize()
        
        print("🎉 테스트 성공!")
        print(f"📊 최적 파라미터: {best_params}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_fold_optuna()
    sys.exit(0 if success else 1)