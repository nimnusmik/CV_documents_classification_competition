#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
팀원 GPU 호환성 빠른 체크 도구
Quick GPU compatibility check for team members
"""

import torch
import sys

def check_gpu_compatibility():
    """팀 협업을 위한 GPU 호환성 체크"""
    print("🔍 팀 GPU 호환성 체크")
    print("=" * 40)
    
    # CUDA 확인
    if not torch.cuda.is_available():
        print("❌ CUDA가 사용 불가능합니다")
        print("💡 해결책:")
        print("   - NVIDIA 드라이버 설치 확인")
        print("   - CUDA 설치 확인")
        print("   - PyTorch CUDA 버전 확인")
        return False
    
    # GPU 정보 출력
    device_count = torch.cuda.device_count()
    print(f"✅ CUDA 사용 가능")
    print(f"🔧 GPU 개수: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        print(f"\n📊 GPU {i}: {device_name}")
        print(f"💾 메모리: {memory_gb:.1f} GB")
        
        # GPU 등급 분류
        if any(gpu in device_name for gpu in ['RTX 4090', 'RTX 4080', 'RTX 3090', 'A100', 'V100']):
            tier = "🏆 HIGH-END"
            batch_rec = "64-128 (224px), 32-64 (384px)"
            note = "최고 성능! Multi-GPU 훈련 가능"
        elif any(gpu in device_name for gpu in ['RTX 3080', 'RTX 3070', 'RTX 4070']):
            tier = "🥈 MID-RANGE"
            batch_rec = "32-64 (224px), 16-32 (384px)"
            note = "우수한 성능! gradient_accumulation_steps=2 권장"
        elif any(gpu in device_name for gpu in ['RTX 3060', 'RTX 2070', 'RTX 2080']):
            tier = "🥉 BUDGET"
            batch_rec = "16-32 (224px), 8-16 (384px)"
            note = "적절한 성능! gradient_accumulation_steps=3-4 권장"
        else:
            tier = "⚠️ LOW-END"
            batch_rec = "8-16 (224px), 4-8 (384px)"
            note = "주의! mixed precision 비활성화, gradient_accumulation_steps=6-8 권장"
        
        print(f"🏷️ 등급: {tier}")
        print(f"📏 권장 배치: {batch_rec}")
        print(f"💡 팁: {note}")
    
    # 권장 명령어
    print(f"\n🚀 다음 단계:")
    print(f"   1. 자동 배치 크기 최적화:")
    print(f"      python src/utils/auto_batch_size.py --config configs/train.yaml --test-only")
    print(f"   2. 설정 파일 업데이트:")
    print(f"      python src/utils/auto_batch_size.py --config configs/train.yaml")
    print(f"   3. 훈련 시작:")
    print(f"      python src/training/train_main.py --mode highperf")
    
    # PyTorch 정보
    print(f"\n🐍 PyTorch 정보:")
    print(f"   버전: {torch.__version__}")
    print(f"   CUDA 지원: {'Yes' if torch.cuda.is_available() else 'No'}")
    
    if torch.cuda.is_available():
        print(f"   CUDA 장치 개수: {torch.cuda.device_count()}")
    
    print(f"   cuDNN 사용 가능: {'Yes' if torch.backends.cudnn.enabled else 'No'}")
    
    return True

if __name__ == "__main__":
    print("팀 협업용 GPU 호환성 체크 도구")
    print("Team GPU Compatibility Checker")
    print()
    
    try:
        success = check_gpu_compatibility()
        if success:
            print(f"\n✅ GPU 설정 완료! 팀 협업 준비 완료!")
        else:
            print(f"\n❌ GPU 설정 문제 발견. 위의 해결책을 참고하세요.")
    except Exception as e:
        print(f"\n💥 오류 발생: {e}")
        print(f"💡 Python 환경과 패키지 설치를 확인하세요.")
