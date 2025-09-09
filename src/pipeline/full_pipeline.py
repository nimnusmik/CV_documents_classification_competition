# src/pipeline/full_pipeline.py
"""
전체 파이프라인 (학습 + 추론 통합)
한 번의 명령으로 학습 완료 후 자동으로 추론 실행하여 제출 파일 생성
"""

import os                                            # 운영체제 파일/디렉터리 조작
import time                                          # 시간 처리 함수
from typing import Optional                          # 타입 힌트 Optional
from pathlib import Path                             # 경로 처리 라이브러리

# ------------------------- 프로젝트 모듈 Import ------------------------- #
from src.training.train_highperf import run_highperf_training   # 고성능 학습 실행 함수
from src.inference.infer_highperf import run_highperf_inference # 고성능 추론 실행 함수
from src.utils import load_yaml, create_log_path               # 핵심 유틸리티 함수
from src.logging.logger import Logger                 # 로그 기록 클래스


def get_model_name(cfg, fold=None):
    """
    단일 모델/다중 모델 앙상블 config에서 fold별 모델명을 자동으로 반환
    - 단일 모델: cfg['model']['name']
    - 다중 모델: cfg['models'][f'fold_{fold}']['name']
    fold 인자가 없으면 단일 모델로 간주
    """
    # 다중 모델 앙상블 여부 판단
    if "models" in cfg and fold is not None and f"fold_{fold}" in cfg["models"]:
        return cfg["models"][f"fold_{fold}"]["name"]
    # 단일 모델
    elif "model" in cfg and "name" in cfg["model"]:
        return cfg["model"]["name"]
    else:
        raise KeyError("모델 이름을 찾을 수 없습니다. config 구조를 확인하세요.")

# ---------------------- 전체 파이프라인 실행 함수 ---------------------- #
# 전체 파이프라인 함수 정의
def run_full_pipeline(config_path: str, skip_training: bool = False, output_dir: Optional[str] = None):
    """
    전체 파이프라인 실행
    
    Args:
        config_path: 설정 파일 경로
        skip_training: True시 학습 건너뛰고 추론만 실행
        output_dir: 결과 저장 디렉터리 (None시 자동 생성)
    """
    
    # 설정 로드
    cfg = load_yaml(config_path)    # YAML 설정 파일 로드

    model_name = get_model_name(cfg, fold=0)  # 모델 이름 확인 (예외 발생 시 조기 종료)


    # 로거 설정
    timestamp = time.strftime("%Y%m%d_%H%M")                    # 타임스탬프 생성
    log_path = create_log_path("pipeline", f"full_pipeline_{timestamp}.log")  # 날짜별 로그 파일 경로 설정
    os.makedirs(os.path.dirname(log_path), exist_ok=True)           # 로그 디렉터리 생성
    
    logger = Logger(log_path=log_path)                              # 로거 인스턴스 생성
    logger.write("🚀 [PIPELINE] Full pipeline started")            # 파이프라인 시작 로그
    logger.write(f"📋 Config: {config_path}")                      # 설정 파일 경로 로그
    logger.write(f"⚙️ Skip training: {skip_training}")             # 학습 건너뛰기 여부 로그
    
    # 예외 처리 시작
    try:
        # ==================== 1단계: 학습 ====================
        # 학습을 건너뛰지 않는 경우
        if not skip_training:
            logger.write("\n" + "="*60)                             # 구분선 로그
            logger.write("🎯 [STAGE 1] HIGH-PERFORMANCE TRAINING")  # 1단계 시작 로그
            logger.write("="*60)                                    # 구분선 로그
            
            # 고성능 학습 실행
            run_highperf_training(config_path)
            
            logger.write("✅ [STAGE 1] Training completed successfully")  # 학습 완료 로그
        # 학습을 건너뛰는 경우
        else:
            logger.write("⏭️ [STAGE 1] Training skipped")          # 학습 건너뛰기 로그
        
        # ==================== 2단계: 결과 파일 찾기 ====================
        logger.write("\n" + "="*60)                                 # 구분선 로그
        logger.write("🔍 [STAGE 2] FINDING TRAINING RESULTS")       # 2단계 시작 로그
        logger.write("="*60)                                        # 구분선 로그
        
        # fold_results.yaml 파일 찾기
        day = time.strftime(cfg["project"]["date_format"])                              # 날짜 포맷 생성
        folder_name = f"{day}_{time.strftime(cfg['project']['time_format'])}_{cfg['project']['run_name']}"  # 폴더명 생성
        exp_base = Path(cfg["output"]["exp_dir"]) / day / folder_name                   # 실험 기본 경로

        fold_results_path = None    # 폴드 결과 파일 경로 초기화
        
        # 실험 기본 경로가 존재하는 경우
        if exp_base.exists():
            # 먼저 직접 경로에서 찾기
            direct_candidate = exp_base / "fold_results.yaml"
            
            # 직접 경로에 파일이 있는 경우
            if direct_candidate.exists():
                fold_results_path = str(direct_candidate)        # 경로 설정
            # 직접 경로에 파일이 없는 경우
            else:
                # 하위 디렉터리 순회 (역순) 하위 폴더에서 찾기
                for exp_dir in sorted(exp_base.iterdir(), reverse=True):
                    # 디렉터리인 경우
                    if exp_dir.is_dir():
                        candidate = exp_dir / "fold_results.yaml"   # 후보 파일 경로
                        
                        # 파일이 존재하는 경우
                        if candidate.exists():
                            fold_results_path = str(candidate)      # 경로 설정
                            break                                   # 반복문 종료
        
        # 폴드 결과 파일을 찾지 못한 경우
        if not fold_results_path:                    
            raise FileNotFoundError(                                # 파일 없음 예외 발생
                f"fold_results.yaml not found in {exp_base}. "      # 경로 정보
                "Make sure training completed successfully."        # 안내 메시지
            )
        
        logger.write(f"📁 Found fold results: {fold_results_path}") # 폴드 결과 파일 발견 로그
        
        # ==================== 3단계: 추론 ====================
        logger.write("\n" + "="*60)                                 # 구분선 로그
        logger.write("🔮 [STAGE 3] HIGH-PERFORMANCE INFERENCE")     # 3단계 시작 로그
        logger.write("="*60)                                        # 구분선 로그
        
        # 출력 경로 설정
        # 출력 디렉터리가 지정되지 않은 경우
        if output_dir is None:
            output_dir = f"submissions/{day}"                       # 기본 출력 디렉터리 설정
        
        # 증강 타입 결정 (학습 설정과 동일한 로직 사용)
        aug_type = "advanced_augmentation" if cfg["train"].get("use_advanced_augmentation", False) else "basic_augmentation"
        
        output_path = os.path.join(                                 # 출력 파일 경로 생성
            output_dir,                                             # 출력 디렉터리
            f"{cfg['project']['run_name']}_ensemble_{timestamp}_{aug_type}.csv"  # 파일명
        )
        
        # 고성능 추론 실행
        final_output = run_highperf_inference(config_path, fold_results_path, output_path)
        
        logger.write("✅ [STAGE 3] Inference completed successfully")  # 추론 완료 로그
        
        # ==================== 4단계: 결과 요약 ====================
        logger.write("\n" + "="*60)                                 # 구분선 로그
        logger.write("🎉 [PIPELINE] COMPLETION SUMMARY")            # 4단계 시작 로그
        logger.write("="*60)                                        # 구분선 로그
        
        logger.write(f"📊 Final submission file: {final_output}")   # 최종 제출 파일 로그
        logger.write(f"📈 Model config: {cfg['model']['name']}")    # 모델 설정 로그
        logger.write(f"🎯 Target F1 score: ~0.934")                 # 목표 F1 점수 로그
        logger.write(f"💾 Experiment results: {exp_base}")          # 실험 결과 경로 로그
        
        # 최종 출력 파일 경로 반환
        return final_output
    
    # 예외 발생 시
    except Exception as e:
        logger.write(f"❌ [PIPELINE] Failed: {str(e)}")     # 에러 로그
        raise                                               # 예외 재발생
    # 최종적으로 실행
    finally:
        logger.write("🏁 [PIPELINE] Full pipeline ended")   # 파이프라인 종료 로그


# ---------------------- CLI 진입점 ---------------------- #
def main():
    import argparse # 명령행 인자 파싱 모듈
    
    parser = argparse.ArgumentParser(description="전체 파이프라인 (학습 + 추론)")    # 인자 파서 생성
    parser.add_argument("--config", type=str, required=True,                                # 설정 파일 인자 추가
                       help="설정 YAML 파일 경로")                                     # 설정 파일 경로 도움말
    parser.add_argument("--skip-training", action="store_true",                             # 학습 건너뛰기 인자 추가
                       help="학습을 건너뛰고 추론만 실행")                         # 학습 건너뛰기 도움말
    parser.add_argument("--output-dir", type=str, default=None,                             # 출력 디렉터리 인자 추가
                       help="제출 파일을 위한 출력 디렉터리")                         # 출력 디렉터리 도움말
    
    args = parser.parse_args()  # 인자 파싱
    
    # 예외 처리 시작
    try:
        print("🚀 Starting Full Pipeline...")               # 파이프라인 시작 메시지
        print(f"📋 Config: {args.config}")                  # 설정 파일 출력
        print(f"⚙️ Skip training: {args.skip_training}")    # 학습 건너뛰기 여부 출력
        print("=" * 50)                                     # 구분선 출력
        
        # 전체 파이프라인 실행
        result = run_full_pipeline(
            args.config,                                    # 설정 파일 경로
            skip_training=args.skip_training,               # 학습 건너뛰기 여부
            output_dir=args.output_dir                      # 출력 디렉터리
        )                                                   # 파이프라인 실행 완료
        
        print("\n" + "=" * 50)                              # 구분선 출력
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")        # 완료 메시지
        print(f"📄 Final submission: {result}")             # 최종 결과 파일 출력
        print("🏆 Ready for competition submission!")       # 제출 준비 완료 메시지

    # 예외 발생 시
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")            # 에러 메시지 출력
        exit(1)                                             # 프로그램 종료 (에러 코드 1)


# ---------------------- 메인 실행 블록 ---------------------- #
if __name__ == "__main__":  # 메인 실행 블록
    main()                  # 메인 함수 호출
