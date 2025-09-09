# src/inference/infer_main.py
# 추론 실행 진입점 스크립트 (CLI에서 실행되는 메인 파일)
# HIGH-PERFORMANCE VERSION

# argparse: CLI 인자 파싱
# sys: 프로그램 종료 제어
import argparse, sys                                                # 명령행 인자 처리 및 시스템 제어
from src.inference.infer import run_inference                       # 기본 추론 실행 함수
from src.inference.infer_highperf import run_highperf_inference     # 고성능 추론


# ---------------------- 메인 함수 ---------------------- #
def main():
    # ArgumentParser 객체 생성
    ap = argparse.ArgumentParser(description="문서 분류 추론 파이프라인")  # 인자 파서 생성
    # 필수 config 인자 추가 (실행에 반드시 필요)
    ap.add_argument("--config", type=str, required=True, help="설정 YAML 파일 경로")   # 설정 파일 인자
    # 출력 경로 지정 (옵션, 없으면 기본 경로 사용)
    ap.add_argument("--out", type=str, default=None, help="출력 CSV 파일 경로")                # 출력 파일 경로 인자
    # 체크포인트 경로 지정 (옵션, 없으면 config 기반 기본값 사용)
    ap.add_argument("--ckpt", type=str, default=None, help="모델 체크포인트 파일 경로")         # 모델 체크포인트 인자
    # 모드 선택 추가
    ap.add_argument("--mode", type=str, choices=["basic", "highperf"], default="highperf",  # 추론 모드 선택 인자
                   help="추론 모드: basic (단일 모델) 또는 highperf (앙상블 + TTA)")
    # 고성능 모드용 fold_results 경로
    ap.add_argument("--fold-results", type=str, default=None,                               # 폴드 결과 파일 인자
                   help="fold_results.yaml 파일 경로 (고성능 모드에서 필수)")
    # CLI 인자 파싱
    args = ap.parse_args()  # 명령행 인자 파싱

    # 예외 처리 블록 시작
    try:
        print(f"🔮 Starting inference pipeline...") # 추론 파이프라인 시작 메시지
        print(f"📋 Config: {args.config}")          # 설정 파일 경로 출력
        print(f"🎯 Mode: {args.mode}")              # 추론 모드 출력
        print("=" * 50)                             # 구분선 출력
        
        # 고성능 모드인 경우
        if args.mode == "highperf":
            # 고성능 추론 메시지
            print("🏆 Running HIGH-PERFORMANCE inference (Ensemble + TTA)")
            
            # 폴드 결과 파일이 없는 경우
            if not args.fold_results:
                print("❌ Error: --fold-results is required for highperf mode")  # 에러 메시지
                print("💡 Example: --fold-results experiments/train/lastest-train/fold_results.yaml")  # 예시 출력
                sys.exit(1) # 프로그램 종료 (에러 코드 1)
            
            # 고성능 추론 실행
            output_path = run_highperf_inference(args.config, args.fold_results, args.out)  # 고성능 추론 함수 호출
            print(f"📄 High-performance prediction saved: {output_path}")                   # 고성능 예측 결과 저장 메시지
        
        # 기본 모드인 경우
        else:
            print("📚 Running BASIC inference (Single model)")                              # 기본 추론 메시지
            # run_inference 실행 (config 경로, out, ckpt 전달)
            run_inference(args.config, out=args.out, ckpt=args.ckpt)                        # 기본 추론 함수 호출
        
        # 정상 종료 메시지 출력
        print("\n" + "=" * 50)                                                              # 구분선 출력
        print("✅ [EXIT] inference finished successfully (see logs/* for details)")         # 성공 메시지
        print("📊 Check submissions/ folder for prediction files")                          # 결과 파일 위치 안내
        sys.exit(0) # 정상 종료

    # 사용자가 Ctrl+C 등으로 중단한 경우
    except KeyboardInterrupt:
        print("[EXIT] inference interrupted by user (KeyboardInterrupt)")  # 사용자 중단 메시지
        sys.exit(130)   # 인터럽트 종료

    # 그 외 모든 예외 처리
    except Exception as e:
        print(f"[EXIT][ERROR] inference failed: {type(e).__name__}: {e}")  # 에러 메시지
        sys.exit(1)                             # 비정상 종료


# ---------------------- 메인 실행 블록 ---------------------- #
if __name__ == "__main__":  # 메인 실행 블록
    main()                  # 메인 함수 호출
