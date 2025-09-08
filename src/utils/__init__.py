# ------------------------- utils 패키지 초기화 모듈 ------------------------- #
# 유틸리티 함수들을 모아놓은 패키지입니다.
# 공통 함수(common.py), 시드 설정(seed.py), GPU 체크(team_gpu_check.py) 등이 포함됩니다.

# ------------------------- 공통 유틸리티 함수 Import ------------------------- #
from .common import (                                       # 공통 함수들
    ensure_dir,                                             # 디렉토리 생성 보장
    load_yaml,                                              # YAML 파일 로드
    dump_yaml,                                              # YAML 파일 저장
    jsonl_append,                                           # JSONL 파일 추가
    now,                                                    # 현재 시간 문자열
    short_uid,                                              # 짧은 고유 ID 생성
    resolve_path,                                           # 경로 해석
    require_file,                                           # 파일 존재 확인
    require_dir                                             # 디렉토리 존재 확인
)

# ------------------------- 시드 설정 함수 Import ------------------------- #
from .seed import set_seed                                  # 랜덤 시드 고정 함수

# ------------------------- GPU 관련 함수 Import ------------------------- #
from .team_gpu_check import check_gpu_compatibility         # GPU 호환성 체크 함수

# ------------------------- 외부 노출 함수 정의 ------------------------- #
__all__ = [                                                 # 패키지에서 외부로 노출할 함수들
    # 공통 유틸리티
    'ensure_dir',                                           # 디렉토리 생성 보장
    'load_yaml',                                            # YAML 파일 로드
    'dump_yaml',                                            # YAML 파일 저장
    'jsonl_append',                                         # JSONL 파일 추가
    'now',                                                  # 현재 시간 문자열
    'short_uid',                                            # 짧은 고유 ID 생성
    'resolve_path',                                         # 경로 해석
    'require_file',                                         # 파일 존재 확인
    'require_dir',                                          # 디렉토리 존재 확인
    
    # 시드 설정
    'set_seed',                                             # 랜덤 시드 고정
    
    # GPU 관련
    'check_gpu_compatibility',                              # GPU 호환성 체크
]
