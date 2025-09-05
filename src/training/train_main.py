# src/training/train_main.py
# 학습 실행 진입점 스크립트 (CLI에서 실행되는 메인 파일)
# 🚀 HIGH-PERFORMANCE VERSION with FULL PIPELINE

import argparse, sys                                    # argparse: CLI 인자 파싱 / sys: 프로그램 종료 코드 제어
from src.training.train import run_training             # 기존 학습 실행 함수
from src.training.train_highperf import run_highperf_training  # 🚀 고성능 학습 실행 함수
from src.pipeline.full_pipeline import run_full_pipeline      # 🎯 통합 파이프라인


# ---------------- 메인 함수 ---------------- #
def main():
    # ArgumentParser 객체 생성 (CLI 인자 정의)
    ap = argparse.ArgumentParser(description="Document Classification Training Pipeline")
    # --config 옵션 추가 (필수 인자, 학습 설정 YAML 파일 경로)
    ap.add_argument("--config", type=str, required=True, help="Path to training config YAML file")
    # --mode 옵션 추가 (실행 모드 선택)
    ap.add_argument("--mode", type=str, 
                   choices=["basic", "highperf", "full-pipeline"], 
                   default="full-pipeline", 
                   help="Execution mode: basic (original), highperf (training only), full-pipeline (train+inference)")
    # --skip-training 옵션 (full-pipeline 모드에서만 사용)
    ap.add_argument("--skip-training", action="store_true",
                   help="Skip training and run inference only (full-pipeline mode)")
    # 인자 파싱 → args.config, args.mode 속성 사용 가능
    args = ap.parse_args()

    try:
        print(f"🚀 Starting training pipeline...")
        print(f"📋 Config: {args.config}")
        print(f"🎯 Mode: {args.mode}")
        print("=" * 50)
        
        # 모드에 따라 실행 함수 선택
        if args.mode == "full-pipeline":
            print("🎯 Running FULL PIPELINE (Training + Inference)")
            print("🏆 Target: F1 ~0.934 with automatic submission file generation")
            # 통합 파이프라인 실행
            result = run_full_pipeline(args.config, skip_training=args.skip_training)
            print(f"\n🎉 PIPELINE COMPLETED!")
            print(f"📄 Final submission: {result}")
            
        elif args.mode == "highperf":
            print("🏆 Running HIGH-PERFORMANCE training only (Target: F1 ~0.934)")
            # 고성능 학습 실행 함수 호출
            run_highperf_training(args.config)
            
        else:
            print("📚 Running BASIC training (Original pipeline)")
            # 기존 학습 실행 함수 호출
            run_training(args.config)
        
        # 정상 종료 메시지 출력
        print("\n" + "=" * 50)
        print("✅ [EXIT] Pipeline finished successfully")
        if args.mode != "full-pipeline":
            print("📊 Check experiments/ folder for trained models")
            print("💡 Use --mode full-pipeline for automatic inference")
        # 프로세스 정상 종료 코드 반환 (0)
        sys.exit(0)

    # Ctrl+C 입력 시 처리
    except KeyboardInterrupt:
        # 사용자 인터럽트 메시지 출력
        print("\n⛔ [EXIT] Pipeline interrupted by user (KeyboardInterrupt)")
        # 종료 코드 130 (POSIX 신호 코드 SIGINT)
        sys.exit(130)

    # 그 외 모든 예외 처리
    except Exception as e:
        # 에러 유형과 메시지 출력
        print(f"[EXIT][ERROR] training failed: {type(e).__name__}: {e}")
        # 비정상 종료 코드 반환 (1)
        sys.exit(1)


# ---------------- 실행 진입점 ---------------- #
if __name__ == "__main__":
    # main() 호출 → CLI 실행 시에만 작동
    main()
