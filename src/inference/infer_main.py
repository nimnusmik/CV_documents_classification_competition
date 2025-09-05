# src/inference/infer_main.py
# 추론 실행 진입점 스크립트 (CLI에서 실행되는 메인 파일)
# 🚀 HIGH-PERFORMANCE VERSION

# argparse: CLI 인자 파싱
# sys: 프로그램 종료 제어
import argparse, sys
# run_inference: 추론 실행 함수 (별도 모듈에서 가져옴)
from src.inference.infer import run_inference
from src.inference.infer_highperf import run_highperf_inference  # 🚀 고성능 추론


# 메인 함수 정의
def main():
    # ArgumentParser 객체 생성
    ap = argparse.ArgumentParser(description="Document Classification Inference Pipeline")
    # 필수 config 인자 추가 (실행에 반드시 필요)
    ap.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    # 출력 경로 지정 (옵션, 없으면 기본 경로 사용)
    ap.add_argument("--out", type=str, default=None, help="Output CSV path")
    # 체크포인트 경로 지정 (옵션, 없으면 config 기반 기본값 사용)
    ap.add_argument("--ckpt", type=str, default=None, help="Model checkpoint path")
    # 🚀 모드 선택 추가
    ap.add_argument("--mode", type=str, choices=["basic", "highperf"], default="highperf",
                   help="Inference mode: basic (single model) or highperf (ensemble + TTA)")
    # 🚀 고성능 모드용 fold_results 경로
    ap.add_argument("--fold-results", type=str, default=None,
                   help="Path to fold_results.yaml (required for highperf mode)")
    # CLI 인자 파싱
    args = ap.parse_args()

    # 예외 처리 블록 시작
    try:
        print(f"🔮 Starting inference pipeline...")
        print(f"📋 Config: {args.config}")
        print(f"🎯 Mode: {args.mode}")
        print("=" * 50)
        
        if args.mode == "highperf":
            print("🏆 Running HIGH-PERFORMANCE inference (Ensemble + TTA)")
            # fold_results 경로 확인
            if not args.fold_results:
                print("❌ Error: --fold-results is required for highperf mode")
                print("💡 Example: --fold-results experiments/train/20250905/v094-swin-highperf/fold_results.yaml")
                sys.exit(1)
            
            # 고성능 추론 실행
            output_path = run_highperf_inference(args.config, args.fold_results, args.out)
            print(f"📄 High-performance prediction saved: {output_path}")
            
        else:
            print("📚 Running BASIC inference (Single model)")
            # run_inference 실행 (config 경로, out, ckpt 전달)
            run_inference(args.config, out=args.out, ckpt=args.ckpt)
        
        # 정상 종료 메시지 출력
        print("\n" + "=" * 50)
        print("✅ [EXIT] inference finished successfully (see logs/* for details)")
        print("📊 Check submissions/ folder for prediction files")
        # 종료 코드 0 (성공)
        sys.exit(0)

    # 사용자가 Ctrl+C 등으로 중단한 경우
    except KeyboardInterrupt:
        print("[EXIT] inference interrupted by user (KeyboardInterrupt)")
        # 종료 코드 130 (SIGINT)
        sys.exit(130)

    # 그 외 모든 예외 처리
    except Exception as e:
        print(f"[EXIT][ERROR] inference failed: {type(e).__name__}: {e}")
        # 종료 코드 1 (실패)
        sys.exit(1)


# 스크립트 직접 실행 시 main() 호출
if __name__ == "__main__":
    main()
