# src/utils/wandb_logger.py
"""
WandB 로깅 유틸리티
팀 프로젝트용 WandB 통합 로깅 시스템
"""

import os
import wandb
from datetime import datetime
from typing import Dict, Any, Optional
import torch


class WandBLogger:
    """WandB 로깅 클래스"""
    
    def __init__(
        self,
        project_name: str = "document-classification-team",
        entity: Optional[str] = None,
        experiment_name: str = "experiment",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        self.project_name = project_name
        self.entity = entity
        self.experiment_name = experiment_name
        self.config = config or {}
        self.tags = tags or []
        
        # 타임스탬프 추가
        timestamp = datetime.now().strftime("%m%d-%H%M")
        self.run_name = f"{experiment_name}-{timestamp}"
        
        self.run = None
        self.is_initialized = False
    
    def login(self):
        """WandB 로그인"""
        try:
            if wandb.api.api_key is None:
                print("WandB에 로그인이 필요합니다.")
                wandb.login()
            else:
                print(f"WandB 로그인 상태: {wandb.api.viewer()['username']}")
        except:
            print("WandB 로그인을 진행합니다...")
            wandb.login()
    
    def init_run(self, fold: Optional[int] = None):
        """WandB 실행 초기화"""
        if self.is_initialized:
            return
        
        self.login()
        
        # fold가 지정된 경우 run name에 추가
        run_name = self.run_name
        if fold is not None:
            run_name = f"fold-{fold}-{run_name}"
        
        # WandB run 초기화
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=self.config,
            tags=self.tags,
            reinit=True
        )
        
        self.is_initialized = True
        print(f"📋 실험명: {run_name}")
        print(f"🔗 WandB URL: {self.run.url}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """메트릭 로깅"""
        if not self.is_initialized:
            return
        
        wandb.log(metrics, step=step)
    
    def log_model(self, model_path: str, name: str = "model"):
        """모델 아티팩트 로깅"""
        if not self.is_initialized:
            return
        
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    def log_confusion_matrix(self, y_true, y_pred, class_names=None):
        """Confusion Matrix 로깅"""
        if not self.is_initialized:
            return
        
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
    
    def log_predictions(self, images, predictions, targets, class_names=None):
        """예측 결과 시각화 로깅"""
        if not self.is_initialized:
            return
        
        # 최대 100개 샘플만 로깅
        max_samples = min(100, len(images))
        
        data = []
        for i in range(max_samples):
            img = images[i]
            pred = predictions[i]
            target = targets[i]
            
            # 이미지를 wandb Image로 변환
            if torch.is_tensor(img):
                img = img.cpu().numpy().transpose(1, 2, 0)
            
            pred_class = class_names[pred] if class_names else str(pred)
            target_class = class_names[target] if class_names else str(target)
            
            data.append([
                wandb.Image(img),
                pred_class,
                target_class,
                pred == target
            ])
        
        table = wandb.Table(
            data=data,
            columns=["Image", "Prediction", "Target", "Correct"]
        )
        
        wandb.log({"predictions": table})
    
    def finish(self):
        """WandB 실행 종료"""
        if self.run is not None:
            wandb.finish()
            self.is_initialized = False
    
    def __enter__(self):
        """Context manager 진입"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.finish()


# 편의 함수들
def create_wandb_config(
    model_name: str,
    img_size: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    **kwargs
) -> Dict[str, Any]:
    """WandB config 생성 함수"""
    config = {
        "architecture": model_name,
        "image_size": img_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "framework": "PyTorch",
        "dataset": "Document Classification",
    }
    config.update(kwargs)
    return config


def log_fold_results(logger: WandBLogger, fold: int, metrics: Dict[str, float]):
    """Fold 결과 로깅"""
    logger.log_metrics({
        f"fold_{fold}_train_f1": metrics.get("train_f1", 0),
        f"fold_{fold}_val_f1": metrics.get("val_f1", 0),
        f"fold_{fold}_train_loss": metrics.get("train_loss", 0),
        f"fold_{fold}_val_loss": metrics.get("val_loss", 0),
    })
