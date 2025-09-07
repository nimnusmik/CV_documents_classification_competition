"""
학습 루프 관리
"""

import random
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict, Any, Optional

from ..utils.metrics import calculate_metrics
from ..config import Config


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> tuple:
    """
    Mixup 데이터 증강 적용
    
    Args:
        x: 입력 이미지 배치
        y: 레이블 배치  
        alpha: Mixup 강도 파라미터
        
    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
    """
    
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


class Trainer:
    """
    훈련 루프를 관리하는 클래스
    
    Features:
        - Mixed Precision Training
        - Mixup/Cutmix 지원
        - Gradient Clipping
        - 진행률 표시
        - 메트릭 계산
    """
    
    def __init__(self, config: Config):
        """
        Args:
            config: 설정 객체
        """
        
        self.config = config
        self.scaler = GradScaler()  # Mixed Precision용
        
        print("✅ Trainer initialized")
        print(f"   - Mixed Precision: Enabled")
        print(f"   - Mixup probability: {config.mixup_prob}")
        print(f"   - Gradient clipping: {config.grad_clip_norm}")
    
    def train_one_epoch(
        self, 
        model: nn.Module, 
        dataloader, 
        optimizer, 
        loss_fn, 
        device: Optional[str] = None
    ) -> Dict[str, float]:
        """
        한 에포크 훈련 수행
        
        Args:
            model: 훈련할 모델
            dataloader: 데이터로더
            optimizer: 옵티마이저
            loss_fn: 손실 함수
            device: 디바이스 (None이면 config.device 사용)
            
        Returns:
            Dict[str, float]: 훈련 결과 메트릭
        """
        
        if device is None:
            device = self.config.device
        
        model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            
            # Mixup 적용 (확률적)
            if random.random() < self.config.mixup_prob:
                mixed_images, y_a, y_b, lam = mixup_data(
                    images, targets, alpha=self.config.mixup_alpha
                )
                
                # Mixed Precision Forward
                with autocast():
                    outputs = model(mixed_images)
                    loss = lam * loss_fn(outputs, y_a) + (1 - lam) * loss_fn(outputs, y_b)
            else:
                # 일반 학습
                with autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # 메모리 효율적
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config.grad_clip_norm
                )
            
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # 메트릭 수집
            total_loss += loss.item()
            
            # 예측값 수집 (Mixup이 아닌 경우만)
            if random.random() >= self.config.mixup_prob:
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())
            
            # 진행률 업데이트
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # 에포크 결과 계산
        avg_loss = total_loss / len(dataloader)
        
        # 메트릭 계산 (Mixup으로 인해 데이터가 적을 수 있음)
        if len(all_preds) > 0:
            metrics = calculate_metrics(all_targets, all_preds)
        else:
            # Mixup만 사용된 경우 메트릭 계산 불가
            metrics = {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0
            }
        
        result = {
            "train_loss": avg_loss,
            "train_accuracy": metrics["accuracy"],
            "train_f1": metrics["macro_f1"],
            "train_weighted_f1": metrics["weighted_f1"]
        }
        
        return result
    
    def get_training_info(self) -> Dict[str, Any]:
        """훈련 설정 정보 반환"""
        
        return {
            "mixed_precision": True,
            "mixup_enabled": self.config.mixup_prob > 0,
            "mixup_probability": self.config.mixup_prob,
            "mixup_alpha": self.config.mixup_alpha,
            "gradient_clipping": self.config.grad_clip_norm,
            "scaler_state": {
                "scale": self.scaler.get_scale(),
                "growth_interval": self.scaler.get_growth_interval()
            }
        }


if __name__ == "__main__":
    # 테스트 코드
    from ..config import Config
    from ..models.model import create_model
    import torch.optim as optim
    
    config = Config()
    config.device = "cpu"  # 테스트용
    config.model_name = "efficientnet_b0"
    config.batch_size = 4
    config.mixup_prob = 0.3
    
    print("=== Trainer Test ===")
    
    # 모델 및 옵티마이저 생성
    model = create_model(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 가상 데이터 생성
    from torch.utils.data import TensorDataset, DataLoader
    
    fake_images = torch.randn(20, 3, 224, 224)
    fake_targets = torch.randint(0, 17, (20,))
    fake_dataset = TensorDataset(fake_images, fake_targets)
    fake_loader = DataLoader(fake_dataset, batch_size=4, shuffle=True)
    
    # 트레이너 생성 및 훈련 테스트
    trainer = Trainer(config)
    
    print("Training info:", trainer.get_training_info())
    
    # 한 에포크 훈련
    train_results = trainer.train_one_epoch(model, fake_loader, optimizer, loss_fn)
    
    print("Train results:", train_results)
    
    print("✅ Trainer test completed successfully")
