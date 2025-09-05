# src/training/train_main.py
"""
학습 실행 진입점 스크립트

CLI에서 실행되는 메인 파일로, 다양한 학습 모드를 지원합니다.
- basic: 기본 학습 모드
- highperf: 고성능 학습 모드 (F1 ~0.934 목표)
- full-pipeline: 학습 + 추론 통합 파이프라인
"""

import argparse                                             # CLI 인자 파싱 라이브러리
import sys                                                  # 시스템 관련 기능 (프로그램 종료 코드 제어)
from src.training.train import run_training                 # 기존 학습 실행 함수
from src.training.train_highperf import run_highperf_training  # 고성능 학습 실행 함수
from src.pipeline.full_pipeline import run_full_pipeline   # 통합 파이프라인 실행 함수


# ==================== 메인 함수 ==================== #
# 메인 함수 정의
def main():
    """CLI 인자를 파싱하고 선택된 모드에 따라 학습 파이프라인을 실행"""
    # ArgumentParser 객체 생성
    ap = argparse.ArgumentParser(description="Document Classification Training Pipeline")   # CLI 인자 파서 생성
    
    # 필수 설정 파일 인자 추가
    ap.add_argument("--config", type=str, required=True,                                    # 설정 파일 경로 (필수)
                   help="Path to training config YAML file")                                # 설정 파일 도움말
    
    # 실행 모드 선택 인자 추가
    ap.add_argument("--mode", type=str,                                                     # 실행 모드 선택
                   choices=["basic", "highperf", "full-pipeline"],                          # 선택지 지정
                   default="full-pipeline",                                                 # 기본값 설정
                   help="Execution mode: basic (original), highperf (training only), full-pipeline (train+inference)")  # 모드 도움말
    
    # 학습 스킵 옵션 추가 (full-pipeline 모드 전용)
    ap.add_argument("--skip-training", action="store_true",                                 # 학습 스킵 플래그
                   help="Skip training and run inference only (full-pipeline mode)")        # 스킵 도움말
    
    # CLI 인자 파싱 실행
    args = ap.parse_args()

    # 파이프라인 실행 예외 처리
    try:
        # 파이프라인 시작 정보 출력
        print(f"🚀 Starting training pipeline...")  # 파이프라인 시작 메시지
        print(f"📋 Config: {args.config}")          # 설정 파일 경로 출력
        print(f"🎯 Mode: {args.mode}")              # 실행 모드 출력
        print("=" * 50)                             # 구분선 출력
        
        #------------------- 실행 모드별 분기 처리 -------------------#
        # 통합 파이프라인 모드인 경우
        if args.mode == "full-pipeline":
            print("🎯 Running FULL PIPELINE (Training + Inference)")                    # 모드 안내 메시지
            print("🏆 Target: F1 ~0.934 with automatic submission file generation")     # 목표 성능 안내
            
            # 통합 파이프라인 실행
            result = run_full_pipeline(args.config, skip_training=args.skip_training)
            
            # 파이프라인 완료 메시지 출력
            print(f"\n🎉 PIPELINE COMPLETED!")                                          # 완료 메시지
            print(f"📄 Final submission: {result}")                                     # 최종 결과 파일 경로 출력
            
        # 고성능 학습 모드인 경우
        elif args.mode == "highperf":
            print("🏆 Running HIGH-PERFORMANCE training only (Target: F1 ~0.934)")      # 모드 안내 메시지
            
            # 고성능 학습 실행
            run_highperf_training(args.config)
            
        # 기본 학습 모드 처리
        else:
            print("📚 Running BASIC training (Original pipeline)")          # 모드 안내 메시지
            
            # 기본 학습 실행
            run_training(args.config)
        
        # 정상 완료 메시지 출력
        print("\n" + "=" * 50)                                              # 구분선 출력
        print("✅ [EXIT] Pipeline finished successfully")                   # 성공 완료 메시지
        
        # 추가 안내 메시지 출력 (통합 파이프라인이 아닌 경우)
        if args.mode != "full-pipeline":
            print("📊 Check experiments/ folder for trained models")       # 모델 저장 위치 안내
            print("💡 Use --mode full-pipeline for automatic inference")   # 추론 실행 안내
            
        # 정상 종료 처리
        sys.exit(0)

    # 사용자 인터럽트 예외 처리 (Ctrl+C 입력 시)
    except KeyboardInterrupt:
        # 인터럽트 메시지 출력
        print("\n⛔ [EXIT] Pipeline interrupted by user (KeyboardInterrupt)")  # 사용자 중단 메시지
        
        # 인터럽트 종료 코드 반환
        sys.exit(130)

    # 일반 예외 처리
    except Exception as e:
        # 에러 정보 출력
        print(f"[EXIT][ERROR] training failed: {type(e).__name__}: {e}")  # 에러 타입과 메시지 출력
        
        # 비정상 종료 처리
        sys.exit(1)


# ==================== 스크립트 실행 진입점 ==================== #
# 스크립트 직접 실행 시 메인 함수 호출
if __name__ == "__main__":  # 스크립트 직접 실행 시
    # 메인 함수 실행
    main()                  # main() 함수 호출
