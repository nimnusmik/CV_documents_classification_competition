# src/pipeline/full_pipeline.py
"""
전체 파이프라인 (학습 + 추론 통합)
한 번의 명령으로 학습 완료 후 자동으로 추론 실행하여 제출 파일 생성
"""

import os
import time
from typing import Optional
from pathlib import Path

from src.training.train_highperf import run_highperf_training
from src.inference.infer_highperf import run_highperf_inference
from src.utils.common import load_yaml
from src.utils.logger import Logger


def run_full_pipeline(config_path: str, skip_training: bool = False, output_dir: Optional[str] = None):
    """
    전체 파이프라인 실행
    
    Args:
        config_path: 설정 파일 경로
        skip_training: True시 학습 건너뛰고 추론만 실행
        output_dir: 결과 저장 디렉터리 (None시 자동 생성)
    """
    
    # 설정 로드
    cfg = load_yaml(config_path)
    
    # 로거 설정
    timestamp = time.strftime("%Y%m%d_%H%M")
    log_path = f"logs/pipeline/full_pipeline_{timestamp}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = Logger(log_path=log_path)
    logger.write("🚀 [PIPELINE] Full pipeline started")
    logger.write(f"📋 Config: {config_path}")
    logger.write(f"⚙️ Skip training: {skip_training}")
    
    try:
        # ==================== 1단계: 학습 ====================
        if not skip_training:
            logger.write("\n" + "="*60)
            logger.write("🎯 [STAGE 1] HIGH-PERFORMANCE TRAINING")
            logger.write("="*60)
            
            # 고성능 학습 실행
            run_highperf_training(config_path)
            
            logger.write("✅ [STAGE 1] Training completed successfully")
        else:
            logger.write("⏭️ [STAGE 1] Training skipped")
        
        # ==================== 2단계: 결과 파일 찾기 ====================
        logger.write("\n" + "="*60)
        logger.write("🔍 [STAGE 2] FINDING TRAINING RESULTS")
        logger.write("="*60)
        
        # fold_results.yaml 파일 찾기
        day = time.strftime(cfg["project"]["date_format"])
        exp_base = Path(cfg["output"]["exp_dir"]) / day / cfg["project"]["run_name"]
        
        fold_results_path = None
        if exp_base.exists():
            # 가장 최근 실험 폴더에서 fold_results.yaml 찾기
            for exp_dir in sorted(exp_base.iterdir(), reverse=True):
                candidate = exp_dir / "fold_results.yaml"
                if candidate.exists():
                    fold_results_path = str(candidate)
                    break
        
        if not fold_results_path:
            raise FileNotFoundError(
                f"fold_results.yaml not found in {exp_base}. "
                "Make sure training completed successfully."
            )
        
        logger.write(f"📁 Found fold results: {fold_results_path}")
        
        # ==================== 3단계: 추론 ====================
        logger.write("\n" + "="*60)
        logger.write("🔮 [STAGE 3] HIGH-PERFORMANCE INFERENCE")
        logger.write("="*60)
        
        # 출력 경로 설정
        if output_dir is None:
            output_dir = f"submissions/{day}"
        
        output_path = os.path.join(
            output_dir, 
            f"{cfg['project']['run_name']}_ensemble_{timestamp}.csv"
        )
        
        # 고성능 추론 실행
        final_output = run_highperf_inference(config_path, fold_results_path, output_path)
        
        logger.write("✅ [STAGE 3] Inference completed successfully")
        
        # ==================== 4단계: 결과 요약 ====================
        logger.write("\n" + "="*60)
        logger.write("🎉 [PIPELINE] COMPLETION SUMMARY")
        logger.write("="*60)
        
        logger.write(f"📊 Final submission file: {final_output}")
        logger.write(f"📈 Model config: {cfg['model']['name']}")
        logger.write(f"🎯 Target F1 score: ~0.934")
        logger.write(f"💾 Experiment results: {exp_base}")
        
        return final_output
        
    except Exception as e:
        logger.write(f"❌ [PIPELINE] Failed: {str(e)}")
        raise
    finally:
        logger.write("🏁 [PIPELINE] Full pipeline ended")


def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Full Pipeline (Training + Inference)")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to config YAML file")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and run inference only")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for submission file")
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting Full Pipeline...")
        print(f"📋 Config: {args.config}")
        print(f"⚙️ Skip training: {args.skip_training}")
        print("=" * 50)
        
        result = run_full_pipeline(
            args.config, 
            skip_training=args.skip_training,
            output_dir=args.output_dir
        )
        
        print("\n" + "=" * 50)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📄 Final submission: {result}")
        print("🏆 Ready for competition submission!")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
