# 표준 및 외부 라이브러리 임포트
import random, os, numpy as np, torch   # random : 파이썬 난수 발생
                                        # os     : 환경 변수 등 시스템 제어
                                        # numpy  : 수치 연산 난수 제어
                                        # torch  : PyTorch 난수 제어


# ---------------------- 시드 고정 함수 ---------------------- #
def set_seed(seed: int = 42, deterministic: bool = True):
    """
    전체 학습 파이프라인에서 재현성을 확보하기 위해
    파이썬, NumPy, PyTorch의 난수를 고정하는 함수
    """

    # Python 내장 random 시드 고정
    random.seed(seed)

    # NumPy 난수 시드 고정
    np.random.seed(seed)

    # PyTorch CPU 연산 난수 시드 고정
    torch.manual_seed(seed)

    # PyTorch CUDA 연산(멀티 GPU 포함) 시드 고정
    torch.cuda.manual_seed_all(seed)

    # deterministic 모드 활성화 시
    if deterministic:
        # cudnn의 연산을 결정론적으로 설정 (속도↓, 일관성↑)
        torch.backends.cudnn.deterministic = True
        # cudnn 자동 최적화 비활성화 (결과 변동 방지)
        torch.backends.cudnn.benchmark = False