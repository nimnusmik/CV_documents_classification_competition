"""
모델 생성 및 관리
"""

import timm
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ..config import Config


def create_model(config: Config, device: Optional[str] = None) -> nn.Module:
    """
    설정에 따른 모델 생성
    
    Args:
        config: 설정 객체
        device: 디바이스 (None이면 config.device 사용)
        
    Returns:
        nn.Module: 생성된 모델
    """
    
    if device is None:
        device = config.device
    
    # timm 모델 생성
    model = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes
    )
    
    # 디바이스로 이동
    model = model.to(device)
    
    print(f"✅ Model created: {config.model_name}")
    print(f"   - Pretrained: {config.pretrained}")
    print(f"   - Num classes: {config.num_classes}")
    print(f"   - Device: {device}")
    
    return model


def load_pretrained_model(
    model: nn.Module, 
    state_dict_path: str, 
    strict: bool = True
) -> nn.Module:
    """
    사전 훈련된 가중치 로드
    
    Args:
        model: 모델 객체
        state_dict_path: state_dict 파일 경로
        strict: 엄격한 키 매칭 여부
        
    Returns:
        nn.Module: 가중치가 로드된 모델
    """
    
    # state_dict 로드
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    # 모델에 로드
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"⚠️ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️ Unexpected keys: {unexpected_keys}")
    
    if not missing_keys and not unexpected_keys:
        print(f"✅ Model weights loaded successfully from {state_dict_path}")
    else:
        print(f"⚠️ Model weights loaded with warnings from {state_dict_path}")
    
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    모델 정보 조회
    
    Args:
        model: 모델 객체
        
    Returns:
        Dict[str, Any]: 모델 정보
    """
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 메모리 사용량 추정 (MB)
    param_size = total_params * 4  # float32 기준
    model_size_mb = param_size / (1024 * 1024)
    
    # 모델 타입 정보
    model_type = type(model).__name__
    
    info = {
        "model_type": model_type,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "model_size_mb": model_size_mb,
        "device": next(model.parameters()).device.type,
        "dtype": str(next(model.parameters()).dtype)
    }
    
    # GPU 메모리 정보 (CUDA 사용시)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    
    return info


def freeze_backbone(model: nn.Module, freeze_layers: List[str] = None) -> nn.Module:
    """
    백본 레이어 동결 (Transfer Learning)
    
    Args:
        model: 모델 객체
        freeze_layers: 동결할 레이어 이름 리스트 (None이면 classifier만 남기고 전체 동결)
        
    Returns:
        nn.Module: 백본이 동결된 모델
    """
    
    if freeze_layers is None:
        # 기본적으로 classifier를 제외한 모든 레이어 동결
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'head' not in name and 'fc' not in name:
                param.requires_grad = False
                
        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
        print(f"✅ Backbone frozen: {frozen_params} parameters")
    else:
        # 지정된 레이어들만 동결
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(freeze_layer in name for freeze_layer in freeze_layers):
                param.requires_grad = False
                frozen_count += 1
        
        print(f"✅ Specified layers frozen: {frozen_count} parameters")
    
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """
    모델 전체 동결 해제
    
    Args:
        model: 모델 객체
        
    Returns:
        nn.Module: 동결 해제된 모델
    """
    
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"✅ Model unfrozen: {trainable_params} trainable parameters")
    
    return model


class ModelEMA:
    """
    Exponential Moving Average를 적용한 모델 (선택적 사용)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: 원본 모델
            decay: EMA 감쇠율
        """
        
        self.model = model
        self.decay = decay
        self.ema_model = type(model)(model.config) if hasattr(model, 'config') else None
        
        if self.ema_model is None:
            # 모델 복사
            import copy
            self.ema_model = copy.deepcopy(model)
        
        # EMA 모델을 평가 모드로 설정
        self.ema_model.eval()
        
        # 그래디언트 비활성화
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self):
        """EMA 가중치 업데이트"""
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def get_model(self) -> nn.Module:
        """EMA 모델 반환"""
        return self.ema_model


def create_model_with_config(
    model_name: str,
    num_classes: int = 17,
    pretrained: bool = True,
    device: str = "cuda",
    freeze_backbone: bool = False,
    use_ema: bool = False,
    ema_decay: float = 0.9999
) -> Dict[str, Any]:
    """
    고급 모델 생성 함수 (다양한 옵션 지원)
    
    Args:
        model_name: timm 모델명
        num_classes: 클래스 수
        pretrained: 사전훈련 가중치 사용 여부
        device: 디바이스
        freeze_backbone: 백본 동결 여부
        use_ema: EMA 사용 여부
        ema_decay: EMA 감쇠율
        
    Returns:
        Dict[str, Any]: 모델 및 관련 객체들
    """
    
    # 기본 모델 생성
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    ).to(device)
    
    result = {"model": model}
    
    # 백본 동결
    if freeze_backbone:
        model = freeze_backbone(model)
        result["frozen_backbone"] = True
    
    # EMA 모델 생성
    if use_ema:
        ema = ModelEMA(model, decay=ema_decay)
        result["ema"] = ema
    
    # 모델 정보
    result["info"] = get_model_info(model)
    
    return result


def print_model_summary(model: nn.Module):
    """모델 정보를 보기 좋게 출력"""
    
    info = get_model_info(model)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Model Type: {info['model_type']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Frozen Parameters: {info['frozen_parameters']:,}")
    print(f"Model Size: {info['model_size_mb']:.2f} MB")
    print(f"Device: {info['device']}")
    print(f"Data Type: {info['dtype']}")
    
    if 'gpu_memory_allocated_mb' in info:
        print(f"GPU Memory Allocated: {info['gpu_memory_allocated_mb']:.2f} MB")
        print(f"GPU Memory Reserved: {info['gpu_memory_reserved_mb']:.2f} MB")
    
    print("=" * 50)


if __name__ == "__main__":
    # 테스트 코드
    from ..config import Config
    
    config = Config()
    config.model_name = "efficientnet_b0"  # 테스트용으로 작은 모델
    config.device = "cpu"  # 테스트용으로 CPU
    
    print("=== Model Creation Test ===")
    
    # 기본 모델 생성
    model = create_model(config)
    
    # 모델 정보 조회
    print_model_summary(model)
    
    print("\n=== Advanced Model Creation Test ===")
    
    # 고급 모델 생성
    advanced_result = create_model_with_config(
        model_name="efficientnet_b0",
        num_classes=17,
        device="cpu",
        freeze_backbone=True,
        use_ema=True
    )
    
    advanced_model = advanced_result["model"]
    ema = advanced_result.get("ema")
    
    print("✅ Advanced model created")
    
    if ema:
        print("✅ EMA model created")
    
    # 백본 동결 해제 테스트
    print("\n=== Unfreeze Test ===")
    unfrozen_model = unfreeze_model(advanced_model)
    
    print("✅ Model creation tests completed successfully")
