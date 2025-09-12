#!/usr/bin/env python3
"""
시각화 시스템 테스트 스크립트 - 간단 버전
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os

def create_test_visualization():
    """테스트용 시각화 생성"""
    print("🧪 Testing visualization system...")
    
    try:
        # 새로운 모듈 구조 사용
        from src.utils.visualizations import (
            create_training_visualizations,
            create_inference_visualizations
        )
        
        # 테스트 출력 디렉터리
        test_dir = "experiments/test_viz"
        os.makedirs(test_dir, exist_ok=True)
        
        # 1. 학습 시각화 테스트
        print("📊 Testing training visualization...")
        
        # 가짜 폴드 결과 데이터
        fold_results = {
            'fold_results': [
                {'fold': 1, 'best_f1': 0.85},
                {'fold': 2, 'best_f1': 0.87},
                {'fold': 3, 'best_f1': 0.83},
                {'fold': 4, 'best_f1': 0.86},
                {'fold': 5, 'best_f1': 0.88}
            ],
            'average_f1': 0.858,
            'total_folds': 5
        }
        
        # 가짜 히스토리 데이터
        history_data = {
            'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25],
            'val_loss': [0.7, 0.55, 0.45, 0.35, 0.3],
            'val_f1': [0.75, 0.82, 0.85, 0.87, 0.88],
            'epochs': [1, 2, 3, 4, 5]
        }
        
        create_training_visualizations(fold_results, "test_model", test_dir, history_data)
        print("  ✅ Training visualization completed")
        
        # 2. 추론 시각화 테스트
        print("📈 Testing inference visualization...")
        
        # 가짜 예측 데이터
        predictions = np.random.rand(100, 3)  # 100 samples, 3 classes
        confidence_scores = np.random.rand(100)
        
        create_inference_visualizations(predictions, "test_model", test_dir, confidence_scores)
        print("  ✅ Inference visualization completed")
        
        # 3. 생성된 파일 확인
        images_dir = Path(test_dir) / "images"
        if images_dir.exists():
            image_files = list(images_dir.glob("*.png"))
            print(f"  📸 Generated {len(image_files)} visualization images:")
            for img_file in image_files:
                print(f"    - {img_file.name}")
        
        print("✅ Visualization system test completed successfully!")
        print(f"📁 Test results saved in: {test_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_test_visualization()
    if success:
        print("\n🎉 시각화 시스템이 정상적으로 작동합니다!")
    else:
        print("\n💥 시각화 시스템에 문제가 있습니다.")
        exit(1)
