# argparse: CLI 인자 파싱
# sys: 프로그램 종료 제어
import argparse, sys
# run_inference: 추론 실행 함수 (별도 모듈에서 가져옴)
from src.inference.infer import run_inference


# 메인 함수 정의
def main():
    # ArgumentParser 객체 생성
    ap = argparse.ArgumentParser()
    # 필수 config 인자 추가 (실행에 반드시 필요)
    ap.add_argument("--config", type=str, required=True)
    # 출력 경로 지정 (옵션, 없으면 기본 경로 사용)
    ap.add_argument("--out", type=str, default=None)
    # 체크포인트 경로 지정 (옵션, 없으면 config 기반 기본값 사용)
    ap.add_argument("--ckpt", type=str, default=None)
    # CLI 인자 파싱
    args = ap.parse_args()

    # 예외 처리 블록 시작
    try:
        # run_inference 실행 (config 경로, out, ckpt 전달)
        # 로그 파일에는 [EXIT] INFERENCE 관련 메시지가 기록됨
        run_inference(args.config, out=args.out, ckpt=args.ckpt)
        # 정상 종료 메시지 출력
        print("[EXIT] inference finished successfully (see logs/* for details)")
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
