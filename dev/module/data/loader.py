"""
DataLoader 생성 및 관리
"""

import pandas as pd
from torch.utils.data import DataLoader
from typing import Union, Optional

from .dataset import ImageDataset, TTAImageDataset, MemoryEfficientImageDataset
from .transforms import get_train_transforms, get_test_transforms, get_tta_transforms
from ..config import Config


def create_train_loader(
    data: Union[str, pd.DataFrame],
    config: Config,
    shuffle: bool = True,
    drop_last: bool = False,
    use_cache: bool = False
) -> DataLoader:
    """
    훈련용 DataLoader 생성
    
    Args:
        data: CSV 경로 또는 DataFrame
        config: 설정 객체
        shuffle: 데이터 셔플 여부
        drop_last: 마지막 불완전한 배치 제거 여부
        use_cache: 이미지 캐싱 사용 여부
        
    Returns:
        DataLoader: 훈련용 데이터로더
    """
    
    # Transform 생성
    transform = get_train_transforms(config)
    
    # Dataset 생성 (캐시 옵션 고려)
    if use_cache:
        dataset = MemoryEfficientImageDataset(
            data=data,
            path=config.train_path,
            transform=transform,
            cache_images=True,
            max_cache_size=1000  # 적당한 크기로 제한
        )
    else:
        dataset = ImageDataset(
            data=data,
            path=config.train_path,
            transform=transform
        )
    
    # DataLoader 생성
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,  # GPU 전송 속도 향상
        drop_last=drop_last,
        persistent_workers=True if config.num_workers > 0 else False  # 워커 재사용
    )
    
    return loader


def create_test_loader(
    data: Union[str, pd.DataFrame],
    config: Config,
    shuffle: bool = False
) -> DataLoader:
    """
    테스트/검증용 DataLoader 생성
    
    Args:
        data: CSV 경로 또는 DataFrame
        config: 설정 객체  
        shuffle: 데이터 셔플 여부 (일반적으로 False)
        
    Returns:
        DataLoader: 테스트용 데이터로더
    """
    
    # Transform 생성 (증강 없음)
    transform = get_test_transforms(config)
    
    # Dataset 생성
    dataset = ImageDataset(
        data=data,
        path=config.train_path,  # 검증시에는 train 폴더 사용
        transform=transform
    )
    
    # DataLoader 생성
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    return loader


def create_inference_loader(
    config: Config,
    shuffle: bool = False
) -> DataLoader:
    """
    최종 추론용 DataLoader 생성 (test 폴더 사용)
    
    Args:
        config: 설정 객체
        shuffle: 데이터 셔플 여부
        
    Returns:
        DataLoader: 추론용 데이터로더
    """
    
    # Transform 생성
    transform = get_test_transforms(config)
    
    # Dataset 생성 (test 폴더 + sample_submission.csv)
    dataset = ImageDataset(
        data=config.get_test_csv_path(),
        path=config.test_path,
        transform=transform
    )
    
    # DataLoader 생성
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    return loader


def create_tta_loader(
    config: Config,
    use_heavy_tta: bool = False,
    shuffle: bool = False
) -> DataLoader:
    """
    TTA 추론용 DataLoader 생성
    
    Args:
        config: 설정 객체
        use_heavy_tta: 확장된 TTA 사용 여부
        shuffle: 데이터 셔플 여부
        
    Returns:
        DataLoader: TTA용 데이터로더
    """
    
    # TTA Transform 생성
    if use_heavy_tta:
        from .transforms import get_heavy_tta_transforms
        transforms = get_heavy_tta_transforms(config)
    else:
        transforms = get_tta_transforms(config)
    
    # TTA Dataset 생성
    dataset = TTAImageDataset(
        data=config.get_test_csv_path(),
        path=config.test_path,
        transforms=transforms
    )
    
    # DataLoader 생성 (TTA는 메모리를 많이 사용하므로 작은 배치 크기)
    loader = DataLoader(
        dataset,
        batch_size=config.tta_batch_size,  # 보통 일반 배치보다 작음
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    return loader


def create_fold_loaders(
    train_df: pd.DataFrame,
    train_idx: list,
    val_idx: list,
    config: Config,
    use_cache: bool = False
) -> tuple:
    """
    K-Fold용 train/validation DataLoader 쌍 생성
    
    Args:
        train_df: 전체 훈련 DataFrame
        train_idx: 훈련 인덱스 리스트
        val_idx: 검증 인덱스 리스트
        config: 설정 객체
        use_cache: 캐싱 사용 여부
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # DataFrame 분할
    train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
    
    # 훈련용 DataLoader
    train_loader = create_train_loader(
        data=train_fold_df,
        config=config,
        shuffle=True,
        drop_last=False,
        use_cache=use_cache
    )
    
    # 검증용 DataLoader
    val_loader = create_test_loader(
        data=val_fold_df,
        config=config,
        shuffle=False
    )
    
    return train_loader, val_loader


def get_loader_info(loader: DataLoader) -> dict:
    """
    DataLoader 정보 조회
    
    Args:
        loader: DataLoader 객체
        
    Returns:
        dict: DataLoader 정보
    """
    
    dataset = loader.dataset
    
    info = {
        "dataset_size": len(dataset),
        "batch_size": loader.batch_size,
        "num_batches": len(loader),
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "shuffle": hasattr(loader, 'sampler') and hasattr(loader.sampler, 'shuffle')
    }
    
    # Dataset별 추가 정보
    if hasattr(dataset, 'get_class_distribution'):
        info["class_distribution"] = dataset.get_class_distribution()
    
    if hasattr(dataset, 'get_num_transforms'):
        info["num_tta_transforms"] = dataset.get_num_transforms()
    
    if hasattr(dataset, 'get_cache_stats'):
        info["cache_stats"] = dataset.get_cache_stats()
    
    return info


def estimate_loader_memory(loader: DataLoader, config: Config) -> str:
    """
    DataLoader 메모리 사용량 추정
    
    Args:
        loader: DataLoader 객체
        config: 설정 객체
        
    Returns:
        str: 포맷팅된 메모리 사용량
    """
    
    dataset = loader.dataset
    
    # 기본 계산: img_size x img_size x 3 x 4 (float32)
    single_img_bytes = config.img_size * config.img_size * 3 * 4
    
    # TTA 배수 고려
    tta_multiplier = 1
    if hasattr(dataset, 'get_num_transforms'):
        tta_multiplier = dataset.get_num_transforms()
    
    # 배치당 메모리
    batch_memory = single_img_bytes * loader.batch_size * tta_multiplier
    
    # GPU 오버헤드 고려 (약 2-3배)
    gpu_overhead = 2.5
    total_memory = batch_memory * gpu_overhead
    
    # 포맷팅
    from ..utils.misc import format_memory
    return format_memory(int(total_memory))


def create_optimized_loader(
    data: Union[str, pd.DataFrame],
    config: Config,
    mode: str = "train",
    **kwargs
) -> DataLoader:
    """
    최적화된 DataLoader 생성 (설정 기반 자동 튜닝)
    
    Args:
        data: 데이터 소스
        config: 설정 객체
        mode: 모드 ("train", "test", "tta")
        **kwargs: 추가 인자들
        
    Returns:
        DataLoader: 최적화된 데이터로더
    """
    
    # 모드별 최적화 설정
    optimized_config = config.__class__(**config.__dict__)
    
    if mode == "train":
        # 훈련시 최적화
        optimized_config.num_workers = min(config.num_workers, 16)  # 너무 많으면 오버헤드
        return create_train_loader(data, optimized_config, **kwargs)
    
    elif mode == "test":
        # 테스트시 최적화
        optimized_config.batch_size = min(config.batch_size * 2, 128)  # 큰 배치로 빠른 추론
        return create_test_loader(data, optimized_config, **kwargs)
    
    elif mode == "tta":
        # TTA시 메모리 고려
        optimized_config.tta_batch_size = max(config.tta_batch_size // 2, 16)  # 안전한 배치 크기
        optimized_config.num_workers = min(config.num_workers, 8)  # 메모리 절약
        return create_tta_loader(optimized_config, **kwargs)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # 테스트 코드
    import tempfile
    import os
    from PIL import Image
    import numpy as np
    
    # 설정 생성
    config = Config()
    
    # 임시 테스트 환경 구성
    with tempfile.TemporaryDirectory() as temp_dir:
        # 가상 이미지와 CSV 생성
        img_dir = os.path.join(temp_dir, "train")
        os.makedirs(img_dir)
        
        # 가상 이미지 생성
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
            img.save(os.path.join(img_dir, f"img_{i:03d}.jpg"))
        
        # CSV 파일 생성
        csv_path = os.path.join(temp_dir, "train.csv")
        test_df = pd.DataFrame({
            'ID': [f'img_{i:03d}.jpg' for i in range(10)],
            'target': [i % 3 for i in range(10)]
        })
        test_df.to_csv(csv_path, index=False)
        
        # 설정 업데이트
        config.train_path = img_dir
        config.batch_size = 4
        config.num_workers = 0  # 테스트 환경에서는 0
        
        print("=== DataLoader Creation Test ===")
        
        # 각종 DataLoader 생성 테스트
        train_loader = create_train_loader(csv_path, config)
        test_loader = create_test_loader(csv_path, config)
        
        print("✅ Train and Test loaders created successfully")
        
        # 정보 조회 테스트
        train_info = get_loader_info(train_loader)
        test_info = get_loader_info(test_loader)
        
        print(f"Train loader info: {train_info}")
        print(f"Test loader info: {test_info}")
        
        # 메모리 사용량 추정
        train_memory = estimate_loader_memory(train_loader, config)
        test_memory = estimate_loader_memory(test_loader, config)
        
        print(f"Estimated train memory: {train_memory}")
        print(f"Estimated test memory: {test_memory}")
        
        # 실제 데이터 로딩 테스트
        print("\n=== Data Loading Test ===")
        
        # 한 배치 로딩 테스트
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}: images shape={images.shape}, targets shape={targets.shape}")
            if batch_idx == 0:  # 첫 배치만 테스트
                break
        
        print("✅ Data loading test completed successfully")
        
        # K-Fold 테스트 (간단히)
        print("\n=== K-Fold Test ===")
        
        train_idx = [0, 1, 2, 3, 4, 5, 6, 7]
        val_idx = [8, 9]
        
        fold_train_loader, fold_val_loader = create_fold_loaders(
            test_df, train_idx, val_idx, config
        )
        
        fold_train_info = get_loader_info(fold_train_loader)
        fold_val_info = get_loader_info(fold_val_loader)
        
        print(f"Fold train size: {fold_train_info['dataset_size']}")
        print(f"Fold validation size: {fold_val_info['dataset_size']}")
        
        print("✅ K-Fold loader test completed successfully")
