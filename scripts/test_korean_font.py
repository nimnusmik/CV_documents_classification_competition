#!/usr/bin/env python3
"""
한글 폰트 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.visualizations import create_training_visualizations
import matplotlib.pyplot as plt
import numpy as np

def test_korean_font():
    """한글 폰트 테스트"""
    print("🧪 한글 폰트 테스트 시작...")
    
    # 간단한 한글 포함 시각화 테스트
    try:
        plt.figure(figsize=(8, 6))
        
        # 한글 데이터
        categories = ['학습 정확도', '검증 정확도', '테스트 정확도', '평균 F1 점수']
        values = [0.85, 0.82, 0.78, 0.80]
        
        # 막대그래프 생성
        bars = plt.bar(categories, values, color=['#2E86C1', '#28B463', '#F39C12', '#E74C3C'])
        
        # 한글 제목과 라벨
        plt.title('🎯 모델 성능 평가 결과', fontsize=16, fontweight='bold')
        plt.xlabel('평가 지표', fontsize=12)
        plt.ylabel('점수', fontsize=12)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장
        test_dir = "experiments/test_viz/images"
        os.makedirs(test_dir, exist_ok=True)
        plt.savefig(f"{test_dir}/korean_font_test.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ 한글 폰트 테스트 성공!")
        print(f"📊 테스트 이미지 저장: {test_dir}/korean_font_test.png")
        
    except Exception as e:
        print(f"❌ 한글 폰트 테스트 실패: {e}")

def test_training_visualization_with_korean():
    """한글이 포함된 학습 시각화 테스트"""
    print("🧪 한글 포함 학습 시각화 테스트...")
    
    # 한글 모델명으로 테스트
    test_fold_results = {
        'fold_results': [
            {'fold': 0, 'best_f1': 0.85, 'best_accuracy': 0.87},
            {'fold': 1, 'best_f1': 0.82, 'best_accuracy': 0.84},
            {'fold': 2, 'best_f1': 0.88, 'best_accuracy': 0.89},
            {'fold': 3, 'best_f1': 0.86, 'best_accuracy': 0.87},
            {'fold': 4, 'best_f1': 0.84, 'best_accuracy': 0.85}
        ]
    }
    
    test_history = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25],
        'val_loss': [0.7, 0.55, 0.45, 0.4, 0.38],
        'val_f1': [0.75, 0.80, 0.83, 0.85, 0.84]
    }
    
    # 한글 모델명으로 테스트
    model_name = "나눔고딕_테스트_모델"
    output_dir = "experiments/test_viz"
    
    try:
        create_training_visualizations(
            fold_results=test_fold_results,
            model_name=model_name,
            output_dir=output_dir,
            history_data=test_history
        )
        print("✅ 한글 포함 학습 시각화 테스트 성공!")
        
    except Exception as e:
        print(f"❌ 한글 포함 학습 시각화 테스트 실패: {e}")

if __name__ == "__main__":
    test_korean_font()
    test_training_visualization_with_korean()
    
    print("\n🎉 모든 한글 폰트 테스트 완료!")
    print("📁 결과 이미지는 experiments/test_viz/images/ 폴더에서 확인하세요.")
