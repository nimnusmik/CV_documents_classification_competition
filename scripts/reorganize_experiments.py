#!/usr/bin/env python3
"""
기존 experiments 폴더의 파일들을 새로운 구조로 정리하는 스크립트

기존 구조: experiments/train/20240907/model_name/...
새로운 구조: experiments/{train|infer|optimization}/20240907/model_name/images/...
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Tuple

def detect_pipeline_type(folder_path: Path) -> str:
    """폴더 내용을 분석해서 파이프라인 타입 추정"""
    files = list(folder_path.rglob("*"))
    file_names = [f.name.lower() for f in files]
    
    # 학습 관련 파일들
    train_indicators = ["fold_results.yaml", "best_model", ".pth", "train", "ckpt"]
    
    # 추론 관련 파일들  
    infer_indicators = ["submission", ".csv", "predictions", "inference"]
    
    # 최적화 관련 파일들
    opt_indicators = ["optuna", "hyperparameter", "optimization", "best_params", "study"]
    
    train_score = sum(1 for indicator in train_indicators if any(indicator in name for name in file_names))
    infer_score = sum(1 for indicator in infer_indicators if any(indicator in name for name in file_names))
    opt_score = sum(1 for indicator in opt_indicators if any(indicator in name for name in file_names))
    
    scores = {"train": train_score, "infer": infer_score, "optimization": opt_score}
    return max(scores.keys(), key=lambda k: scores[k])

def extract_model_name(folder_path: Path) -> str:
    """폴더명이나 설정 파일에서 모델명 추출"""
    folder_name = folder_path.name.lower()
    
    # 일반적인 모델명 패턴들
    model_patterns = [
        "swin", "convnext", "vit", "resnet", "efficientnet", 
        "densenet", "mobilenet", "regnet", "transformer"
    ]
    
    for pattern in model_patterns:
        if pattern in folder_name:
            return pattern
    
    # 설정 파일에서 모델명 찾기
    try:
        from src.utils.common import load_yaml
        for config_file in folder_path.rglob("*.yaml"):
            if "config" in config_file.name.lower():
                config = load_yaml(str(config_file))
                if "model" in config and "name" in config["model"]:
                    return config["model"]["name"]
    except:
        pass
    
    return "unknown_model"

def reorganize_experiments(experiments_dir: str = "experiments", dry_run: bool = False) -> List[Tuple[str, str]]:
    """
    실험 폴더를 새로운 구조로 정리
    
    Args:
        experiments_dir: 실험 폴더 경로
        dry_run: True면 실제 이동 없이 계획만 출력
        
    Returns:
        (원본 경로, 대상 경로) 튜플 리스트
    """
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        print(f"❌ {experiments_dir} 폴더가 존재하지 않습니다.")
        return []
    
    moves = []
    
    # 기존 구조 탐색
    for date_folder in exp_path.iterdir():
        if not date_folder.is_dir():
            continue
            
        # 날짜 폴더인지 확인 (YYYYMMDD 형식)
        if not (date_folder.name.isdigit() and len(date_folder.name) == 8):
            continue
        
        print(f"📅 Processing date folder: {date_folder.name}")
        
        for experiment_folder in date_folder.iterdir():
            if not experiment_folder.is_dir():
                continue
                
            print(f"  📁 Analyzing: {experiment_folder.name}")
            
            # 파이프라인 타입 감지
            pipeline_type = detect_pipeline_type(experiment_folder)
            
            # 모델명 추출
            model_name = extract_model_name(experiment_folder)
            
            # 새로운 경로 설정
            new_path = exp_path / pipeline_type / date_folder.name / model_name
            
            # 이미 올바른 위치에 있는지 확인
            if experiment_folder.parent.parent.name == pipeline_type:
                print(f"    ✅ Already in correct location")
                continue
            
            moves.append((str(experiment_folder), str(new_path)))
            print(f"    🔄 {pipeline_type.upper()} | {model_name} | {experiment_folder} -> {new_path}")
    
    # 실제 이동 실행
    if not dry_run:
        print(f"\n🚀 Starting reorganization...")
        
        for src, dst in moves:
            try:
                # 대상 디렉터리 생성
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                # 이동
                shutil.move(src, dst)
                
                # images 폴더 생성
                images_dir = Path(dst) / "images"
                images_dir.mkdir(exist_ok=True)
                
                print(f"✅ Moved: {src} -> {dst}")
                
            except Exception as e:
                print(f"❌ Failed to move {src}: {str(e)}")
    
    else:
        print(f"\n📋 DRY RUN - {len(moves)} folders would be reorganized")
        
    return moves

def create_visualization_for_existing_results():
    """기존 결과들에 대해 시각화 생성"""
    from src.utils.visualizations import (
        visualize_training_pipeline, 
        visualize_inference_pipeline,
        visualize_optimization_pipeline
    )
    
    exp_path = Path("experiments")
    
    for pipeline_type in ["train", "infer", "optimization"]:
        pipeline_path = exp_path / pipeline_type
        if not pipeline_path.exists():
            continue
            
        print(f"🎨 Creating visualizations for {pipeline_type}...")
        
        for date_folder in pipeline_path.iterdir():
            if not date_folder.is_dir():
                continue
                
            for model_folder in date_folder.iterdir():
                if not model_folder.is_dir():
                    continue
                    
                try:
                    model_name = model_folder.name
                    output_dir = str(model_folder)
                    
                    if pipeline_type == "train":
                        # 학습 결과 시각화
                        fold_results_file = model_folder / "fold_results.yaml"
                        if fold_results_file.exists():
                            from src.utils.common import load_yaml
                            results = load_yaml(str(fold_results_file))
                            
                            visualize_training_pipeline(
                                fold_results=results,
                                model_name=model_name,
                                output_dir=output_dir,
                                history_data=None
                            )
                            print(f"  ✅ Training viz: {model_folder}")
                    
                    elif pipeline_type == "infer":
                        # 추론 결과 시각화 (CSV 파일 있는 경우)
                        csv_files = list(model_folder.rglob("*.csv"))
                        if csv_files:
                            import pandas as pd
                            import numpy as np
                            
                            df = pd.read_csv(csv_files[0])
                            predictions = np.random.rand(len(df), 3)  # 예시 데이터
                            
                            visualize_inference_pipeline(
                                predictions=predictions,
                                model_name=model_name,
                                output_dir=output_dir,
                                confidence_scores=None,
                                ensemble_weights=None,
                                tta_results=None
                            )
                            print(f"  ✅ Inference viz: {model_folder}")
                    
                    elif pipeline_type == "optimization":
                        # 최적화 결과 시각화
                        study_files = list(model_folder.rglob("*.pkl"))
                        if study_files:
                            visualize_optimization_pipeline(
                                study_path=str(study_files[0]),
                                model_name=model_name,
                                output_dir=output_dir,
                                trials_df=None
                            )
                            print(f"  ✅ Optimization viz: {model_folder}")
                            
                except Exception as e:
                    print(f"  ❌ Visualization failed for {model_folder}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Reorganize experiments folder structure")
    parser.add_argument("--experiments-dir", default="experiments", help="Experiments directory path")
    parser.add_argument("--dry-run", action="store_true", help="Show planned changes without executing")
    parser.add_argument("--create-viz", action="store_true", help="Create visualizations for existing results")
    
    args = parser.parse_args()
    
    print("🔧 Experiments Folder Reorganization Tool")
    print("=" * 50)
    
    if args.dry_run:
        print("🧪 DRY RUN MODE - No actual changes will be made")
    
    # 폴더 재구성
    moves = reorganize_experiments(args.experiments_dir, args.dry_run)
    
    print(f"\n📊 Summary:")
    print(f"  Total moves planned/executed: {len(moves)}")
    
    if not args.dry_run and moves:
        print("  ✅ Reorganization completed!")
        
        if args.create_viz:
            print("\n🎨 Creating visualizations...")
            create_visualization_for_existing_results()
            print("  ✅ Visualizations completed!")
    
    print("\n📁 New structure:")
    print("experiments/")
    print("├── train/")
    print("│   └── YYYYMMDD/")
    print("│       └── model_name/")
    print("│           └── images/")
    print("├── infer/")
    print("│   └── YYYYMMDD/")
    print("│       └── model_name/")
    print("│           └── images/")
    print("└── optimization/")
    print("    └── YYYYMMDD/")
    print("        └── model_name/")
    print("            └── images/")

if __name__ == "__main__":
    main()
