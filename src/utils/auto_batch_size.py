#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
팀 협업용 GPU 최적화 자동 배치 크기 찾기 도구
다양한 GPU 환경에서 최적의 배치 크기를 자동으로 찾아주는 도구

Author: AI Team
Date: 2025-01-05
"""

import os
import sys
import torch
import gc
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# YAML import with fallback
try:
    import yaml
except ImportError:
    print("❌ PyYAML이 설치되지 않았습니다. 다음 명령어로 설치하세요:")
    print("   pip install PyYAML")
    sys.exit(1)


def get_gpu_info_and_recommendations() -> Dict[str, Any]:
    """GPU 정보를 확인하고 권장 설정을 반환"""
    if not torch.cuda.is_available():
        return {
            'name': 'CPU',
            'total_memory': 0,
            'tier': 'cpu',
            'profile': {
                'batch_224': {'start': 4, 'max': 8, 'safety': 0.9},
                'batch_384': {'start': 2, 'max': 4, 'safety': 0.9},
                'batch_512': {'start': 1, 'max': 2, 'safety': 0.9}
            }
        }
    
    device_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    # GPU 등급별 분류
    if any(gpu in device_name for gpu in ['RTX 4090', 'RTX 4080', 'RTX 3090', 'A100', 'V100']):
        tier = 'high_end'
        profile = {
            'batch_224': {'start': 64, 'max': 128, 'safety': 0.8},
            'batch_384': {'start': 32, 'max': 64, 'safety': 0.8},
            'batch_512': {'start': 16, 'max': 32, 'safety': 0.8}
        }
    elif any(gpu in device_name for gpu in ['RTX 3080', 'RTX 3070', 'RTX 4070']):
        tier = 'mid_range'
        profile = {
            'batch_224': {'start': 32, 'max': 64, 'safety': 0.8},
            'batch_384': {'start': 16, 'max': 32, 'safety': 0.8},
            'batch_512': {'start': 8, 'max': 16, 'safety': 0.8}
        }
    elif any(gpu in device_name for gpu in ['RTX 3060', 'RTX 2070', 'RTX 2080']):
        tier = 'budget'
        profile = {
            'batch_224': {'start': 16, 'max': 32, 'safety': 0.85},
            'batch_384': {'start': 8, 'max': 16, 'safety': 0.85},
            'batch_512': {'start': 4, 'max': 8, 'safety': 0.85}
        }
    else:  # GTX 1660, GTX 1080 등 구형 GPU
        tier = 'low_end'
        profile = {
            'batch_224': {'start': 8, 'max': 16, 'safety': 0.9},
            'batch_384': {'start': 4, 'max': 8, 'safety': 0.9},
            'batch_512': {'start': 2, 'max': 4, 'safety': 0.9}
        }
    
    return {
        'name': device_name,
        'total_memory': total_memory,
        'tier': tier,
        'profile': profile
    }


def test_batch_size(model_name: str, img_size: int, batch_size: int) -> Tuple[bool, Optional[float]]:
    """특정 배치 크기로 메모리 테스트"""
    try:
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
        
        device = torch.device('cuda')
        
        # 간단한 모델로 테스트 (실제 모델 크기와 유사하게)
        if 'swin' in model_name.lower():
            # Swin Transformer 근사 모델
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 128, 3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((7, 7)),
                torch.nn.Flatten(),
                torch.nn.Linear(128 * 7 * 7, 1000),
                torch.nn.Linear(1000, 100)
            ).to(device)
        elif 'convnext' in model_name.lower():
            # ConvNext 근사 모델
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 96, 4, stride=4),
                torch.nn.LayerNorm([96, img_size//4, img_size//4]),
                torch.nn.Conv2d(96, 192, 1),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(192, 100)
            ).to(device)
        else:
            # 기본 ResNet 스타일 모델
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(3, stride=2, padding=1),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, 100)
            ).to(device)
        
        # 테스트 데이터 생성
        test_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
        test_target = torch.randint(0, 100, (batch_size,), device=device)
        
        # Forward pass
        output = model(test_input)
        loss = torch.nn.functional.cross_entropy(output, test_target)
        
        # Backward pass
        loss.backward()
        
        # 메모리 사용량 측정
        memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        # 정리
        del model, test_input, test_target, output, loss
        torch.cuda.empty_cache()
        gc.collect()
        
        return True, memory_used
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            return False, None
        else:
            raise e
    except Exception as e:
        torch.cuda.empty_cache()
        gc.collect()
        return False, None


def find_optimal_batch_size(model_name: str, img_size: int, gpu_info: Dict[str, Any]) -> int:
    """최적의 배치 크기 찾기 (GPU 등급별 최적화)"""
    print(f"🔍 {gpu_info['tier']} GPU 최적 배치 크기 탐색 중...")
    print(f"   GPU: {gpu_info['name']}")
    print(f"   메모리: {gpu_info['total_memory']:.1f} GB")
    print(f"   모델: {model_name}")
    print(f"   이미지 크기: {img_size}")
    
    # 이미지 크기별 프로필 선택
    if img_size <= 224:
        batch_config = gpu_info['profile']['batch_224']
    elif img_size <= 384:
        batch_config = gpu_info['profile']['batch_384']
    else:
        batch_config = gpu_info['profile']['batch_512']
    
    start_batch = batch_config['start']
    max_batch = batch_config['max']
    safety_factor = batch_config['safety']
    
    print(f"   📊 {gpu_info['tier']} GPU 권장 범위: {start_batch} ~ {max_batch}")
    print(f"   🛡️ 안전 마진: {int((1-safety_factor)*100)}%")
    
    optimal_batch = start_batch
    
    # 이진 탐색으로 최적 배치 크기 찾기
    low, high = start_batch, max_batch
    
    while low <= high:
        mid = (low + high) // 2
        
        print(f"   배치 크기 {mid} 테스트 중...", end=" ")
        
        success, memory_used = test_batch_size(model_name, img_size, mid)
        
        if success:
            optimal_batch = mid
            if memory_used:
                print(f"✅ (메모리: {memory_used:.2f} GB)")
            else:
                print("✅")
            low = mid + 1  # 더 큰 배치 시도
        else:
            print("❌ (메모리 부족)")
            high = mid - 1  # 더 작은 배치로 시도
    
    # 안전 마진 적용
    final_batch = max(4, int(optimal_batch * safety_factor))
    
    # 4의 배수로 조정 (모든 GPU에서 효율적)
    final_batch = (final_batch // 4) * 4
    final_batch = max(4, final_batch)  # 최소 4
    
    print(f"\n🎯 {gpu_info['tier']} GPU 최적 배치 크기: {final_batch}")
    
    # GPU별 추가 권장사항
    recommendations = []
    if gpu_info['total_memory'] < 8:
        recommendations.append("💡 낮은 GPU 메모리: gradient_accumulation_steps 사용 권장")
    if "GTX" in gpu_info['name']:
        recommendations.append("💡 구형 GPU: mixed precision (AMP) 비활성화 권장")
    if gpu_info['total_memory'] >= 20:
        recommendations.append("💡 고성능 GPU: 더 큰 모델이나 더 높은 해상도 고려 가능")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    return final_batch


def update_config_file(config_path: str, batch_size: int):
    """설정 파일의 배치 크기 업데이트"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'training' not in config:
            config['training'] = {}
        
        config['training']['batch_size'] = batch_size
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"✅ 설정 파일 업데이트 완료: batch_size = {batch_size}")
        
    except Exception as e:
        print(f"❌ 설정 파일 업데이트 실패: {e}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='팀 협업용 GPU 최적화 자동 배치 크기 찾기')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='YAML 설정 파일 경로')
    parser.add_argument('--model', type=str, help='모델 이름 (옵션)')
    parser.add_argument('--img-size', type=int, help='이미지 크기 (옵션)')
    parser.add_argument('--test-only', action='store_true',
                        help='테스트만 수행하고 설정 파일을 수정하지 않음')
    
    args = parser.parse_args()
    
    print("🚀 팀 협업용 GPU 최적화 자동 배치 크기 찾기 도구")
    print("=" * 55)
    
    if not os.path.exists(args.config):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)
    
    # GPU 확인
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다!")
        sys.exit(1)
    
    # GPU 정보 및 권장사항 가져오기
    gpu_info = get_gpu_info_and_recommendations()
    
    print(f"🔧 GPU: {gpu_info['name']}")
    print(f"💾 GPU 메모리: {gpu_info['total_memory']:.1f} GB")
    print(f"🏆 GPU 등급: {gpu_info['tier']}")
    
    batch_range = gpu_info['profile']['batch_224']
    print(f"💡 권장 배치 범위: {batch_range['start']} ~ {batch_range['max']}")
    
    # 설정 파일 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 모델 및 이미지 크기 추출
    model_name = args.model or config.get('model', {}).get('name', 'swin_base_patch4_window7_224')
    
    # 이미지 크기 찾기 (여러 경로 시도)
    img_size = args.img_size
    if not img_size:
        img_size = (config.get('model', {}).get('img_size') or 
                   config.get('train', {}).get('img_size') or 
                   config.get('training', {}).get('img_size') or 
                   config.get('data', {}).get('img_size') or
                   384)
    
    print(f"📊 모델: {model_name}")
    print(f"📏 이미지 크기: {img_size}")
    
    # 최적 배치 크기 찾기
    optimal_batch = find_optimal_batch_size(model_name, img_size, gpu_info)
    
    print("\n" + "=" * 55)
    print(f"🎉 최종 결과:")
    print(f"   최적 배치 크기: {optimal_batch}")
    print(f"   GPU 등급: {gpu_info['tier']}")
    print(f"   예상 메모리 사용률: ~{(optimal_batch/batch_range['max'])*100:.0f}%")
    
    if not args.test_only:
        # 설정 파일 업데이트
        update_config_file(args.config, optimal_batch)
        
        print(f"\n✅ 완료! 이제 다음 명령어로 최적화된 훈련을 시작하세요:")
        print(f"   python src/training/train_main.py --mode highperf")
        
        # GPU별 추가 권장사항
        print(f"\n💡 {gpu_info['tier']} GPU 추가 권장사항:")
        if gpu_info['total_memory'] < 8:
            print(f"   - gradient_accumulation_steps = 2-4 사용 권장 (낮은 메모리)")
            print(f"   - mixed precision 비활성화 고려")
        elif gpu_info['total_memory'] >= 20:
            print(f"   - 더 큰 모델이나 ensemble 고려 가능")
            print(f"   - Multi-GPU training 가능")
        
        print(f"   - 실제 훈련 시작 전에 작은 epoch로 테스트해보세요")
        print(f"   - 모니터링: nvidia-smi -l 1 명령어로 GPU 사용량 확인")
        print(f"   - 팀원과 배치 크기 설정 공유하여 일관성 유지")
        
    else:
        print(f"\n💡 테스트 모드: 설정 파일이 업데이트되지 않았습니다.")
        print(f"   수동으로 batch_size를 {optimal_batch}로 설정하세요.")
    
    print(f"\n✨ {gpu_info['tier']} GPU 최적화 완료!")
    print(f"🤝 다른 팀원들과 설정을 공유하여 협업하세요!")


if __name__ == "__main__":
    main()
