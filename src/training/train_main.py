# src/training/train_main.py
# 학습 실행 진입점 스크립트 (CLI에서 실행되는 메인 파일)

import argparse, sys                            # argparse: CLI 인자 파싱 / sys: 프로그램 종료 코드 제어
from src.training.train import run_training     # 실제 학습 실행 함수 (run_training) 불러오기


# ---------------- 메인 함수 ---------------- #
def main():
    # ArgumentParser 객체 생성 (CLI 인자 정의)
    ap = argparse.ArgumentParser()
    # --config 옵션 추가 (필수 인자, 학습 설정 YAML 파일 경로)
    ap.add_argument("--config", type=str, required=True)
    # 인자 파싱 → args.config 속성 사용 가능
    args = ap.parse_args()

    try:
        # run_training 함수 호출 (실제 학습 실행)
        run_training(args.config)
        # 정상 종료 메시지 출력
        print("[EXIT] training finished successfully (see logs/* for details)")
        # 프로세스 정상 종료 코드 반환 (0)
        sys.exit(0)

    # Ctrl+C 입력 시 처리
    except KeyboardInterrupt:
        # 사용자 인터럽트 메시지 출력
        print("[EXIT] training interrupted by user (KeyboardInterrupt)")
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
