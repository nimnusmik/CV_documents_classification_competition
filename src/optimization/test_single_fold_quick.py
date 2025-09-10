#!/usr/bin/env python3
"""
단일 폴드 빠른 학습 함수 직접 테스트
"""

import os
import sys
sys.path.append('/home/ieyeppo/AI_Lab/computer-vision-competition-1SEN')

from src.utils.core.common import load_yaml

def test_single_fold_quick():
    """run_single_fold_quick 함수 직접 테스트"""
    
    print("🧪 단일 폴드 빠른 학습 함수 직접 테스트...")
    
    try:
        # 설정 로드 (프로젝트 루트 기준)
        config = load_yaml("configs/train_highperf.yaml")
        
        print(f"📋 설정 로드 완료")
        print(f"   - 모델: {config['model']['name']}")
        print(f"   - 이미지 크기: {config['train']['img_size']}")
        print(f"   - 배치 크기: {config['train']['batch_size']}")
        print(f"   - 에포크: {config['train']['epochs']}")
        
        # 빠른 학습 함수 import
        from src.training.train_highperf import run_single_fold_quick
        
        print("\n🚀 빠른 학습 실행...")
        
        # 빠른 학습 실행
        result_f1 = run_single_fold_quick(config)
        
        print(f"\n🎉 테스트 완료!")
        print(f"📊 결과 F1: {result_f1:.4f}")
        
        if result_f1 > 0.0:
            print("✅ 성공: F1 스코어가 정상적으로 반환됨")
            return True
        else:
            print("❌ 실패: F1 스코어가 0.0")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_fold_quick()
    print(f"\n🏁 최종 결과: {'성공' if success else '실패'}")
    sys.exit(0 if success else 1)