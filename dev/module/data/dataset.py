"""
데이터셋 클래스 정의
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Union, List, Any


class ImageDataset(Dataset):
    """
    이미지 분류를 위한 기본 데이터셋 클래스
    
    Features:
        - CSV 파일 또는 DataFrame 모두 지원
        - 유연한 transform 적용
        - 메모리 효율적인 이미지 로딩
    """
    
    def __init__(self, data: Union[str, pd.DataFrame], path: str, transform=None):
        """
        Args:
            data (Union[str, pd.DataFrame]): CSV 경로 또는 DataFrame
            path (str): 이미지 폴더 경로
            transform: albumentations transform 객체
        """
        
        if isinstance(data, str):
            # CSV 파일 경로인 경우 로드
            self.df = pd.read_csv(data).values
        else:
            # DataFrame인 경우 numpy array로 변환
            self.df = data.values
        
        self.path = path
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        인덱스에 해당하는 이미지와 레이블 반환
        
        Args:
            idx (int): 데이터 인덱스
            
        Returns:
            tuple: (image, target) 또는 (image, dummy_target)
        """
        
        # CSV 구조: [filename, target] 또는 [filename, dummy]
        if len(self.df[idx]) >= 2:
            name, target = self.df[idx][:2]  # 처음 2개 컬럼만 사용
        else:
            name = self.df[idx][0]
            target = 0  # dummy target
        
        # 이미지 로드 (RGB 모드로 변환)
        img_path = os.path.join(self.path, name)
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Transform 적용
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        return img, target
    
    def get_class_distribution(self) -> dict:
        """클래스별 샘플 수 분포 반환"""
        
        if len(self.df[0]) < 2:
            return {}  # target이 없는 경우
        
        targets = [row[1] for row in self.df]
        unique, counts = np.unique(targets, return_counts=True)
        
        return dict(zip(unique.astype(int), counts.astype(int)))
    
    def get_sample_image_path(self, idx: int = 0) -> str:
        """특정 인덱스의 이미지 경로 반환"""
        return os.path.join(self.path, self.df[idx][0])


class TTAImageDataset(Dataset):
    """
    Test Time Augmentation을 위한 특별한 데이터셋 클래스
    
    Features:
        - 단일 이미지에 여러 transform 동시 적용
        - 메모리 효율적인 배치 처리
        - 다양한 TTA 전략 지원
    """
    
    def __init__(self, data: Union[str, pd.DataFrame], path: str, transforms: List[Any]):
        """
        Args:
            data (Union[str, pd.DataFrame]): CSV 경로 또는 DataFrame
            path (str): 이미지 폴더 경로  
            transforms (List[Any]): TTA용 transform 리스트
        """
        
        if isinstance(data, str):
            self.df = pd.read_csv(data).values
        else:
            self.df = data.values
        
        self.path = path
        self.transforms = transforms
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        인덱스에 해당하는 이미지에 모든 TTA transform 적용
        
        Args:
            idx (int): 데이터 인덱스
            
        Returns:
            tuple: (List[augmented_images], target)
        """
        
        # 파일명과 target 추출
        if len(self.df[idx]) >= 2:
            name, target = self.df[idx][:2]
        else:
            name = self.df[idx][0]
            target = 0
        
        # 이미지 로드
        img_path = os.path.join(self.path, name)
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # 모든 transform을 적용한 결과를 리스트로 저장
        augmented_images = []
        for transform in self.transforms:
            aug_img = transform(image=img)['image']
            augmented_images.append(aug_img)
        
        return augmented_images, target
    
    def get_num_transforms(self) -> int:
        """적용되는 transform 개수 반환"""
        return len(self.transforms)
    
    def get_expected_memory_usage(self, img_size: int = 384, batch_size: int = 64) -> str:
        """
        예상 메모리 사용량 계산 (추정치)
        
        Args:
            img_size (int): 이미지 크기
            batch_size (int): 배치 크기
            
        Returns:
            str: 포맷팅된 메모리 사용량
        """
        
        # 단일 이미지 메모리: img_size x img_size x 3 channels x 4 bytes (float32)
        single_img_bytes = img_size * img_size * 3 * 4
        
        # TTA로 인한 배수
        tta_multiplier = len(self.transforms)
        
        # 배치당 총 메모리
        batch_memory = single_img_bytes * tta_multiplier * batch_size
        
        # GPU에서는 추가적인 메모리 오버헤드 고려 (약 2배)
        gpu_memory = batch_memory * 2
        
        # 포맷팅
        from ..utils.misc import format_memory
        return format_memory(gpu_memory)


class MemoryEfficientImageDataset(ImageDataset):
    """
    메모리 효율적인 이미지 데이터셋
    
    Features:
        - 이미지 캐싱 옵션
        - 지연 로딩 (lazy loading)
        - 메모리 사용량 모니터링
    """
    
    def __init__(self, data: Union[str, pd.DataFrame], path: str, 
                 transform=None, cache_images: bool = False, max_cache_size: int = 1000):
        """
        Args:
            data: 데이터 소스
            path: 이미지 경로
            transform: 이미지 변환
            cache_images (bool): 이미지 캐싱 여부
            max_cache_size (int): 최대 캐시 크기
        """
        
        super().__init__(data, path, transform)
        
        self.cache_images = cache_images
        self.max_cache_size = max_cache_size
        self._image_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __getitem__(self, idx: int) -> tuple:
        """캐싱 기능이 포함된 이미지 로딩"""
        
        name, target = self.df[idx][:2] if len(self.df[idx]) >= 2 else (self.df[idx][0], 0)
        
        # 캐시 확인
        if self.cache_images and name in self._image_cache:
            img = self._image_cache[name].copy()
            self._cache_hits += 1
        else:
            # 이미지 로드
            img_path = os.path.join(self.path, name)
            img = np.array(Image.open(img_path).convert('RGB'))
            self._cache_misses += 1
            
            # 캐시에 저장 (크기 제한 확인)
            if self.cache_images and len(self._image_cache) < self.max_cache_size:
                self._image_cache[name] = img.copy()
        
        # Transform 적용
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed['image']
        
        return img, target
    
    def get_cache_stats(self) -> dict:
        """캐시 통계 반환"""
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_enabled": self.cache_images,
            "cache_size": len(self._image_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate
        }
    
    def clear_cache(self):
        """캐시 초기화"""
        self._image_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


if __name__ == "__main__":
    # 테스트 코드
    import tempfile
    import os
    from PIL import Image
    
    # 임시 테스트 데이터 생성
    with tempfile.TemporaryDirectory() as temp_dir:
        # 가상 이미지 생성
        img_dir = os.path.join(temp_dir, "images")
        os.makedirs(img_dir)
        
        for i in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(os.path.join(img_dir, f"test_{i}.jpg"))
        
        # CSV 파일 생성
        csv_path = os.path.join(temp_dir, "test.csv")
        test_df = pd.DataFrame({
            'filename': [f'test_{i}.jpg' for i in range(5)],
            'target': [i % 3 for i in range(5)]
        })
        test_df.to_csv(csv_path, index=False)
        
        print("=== Basic Dataset Test ===")
        
        # 기본 데이터셋 테스트
        dataset = ImageDataset(csv_path, img_dir)
        print(f"Dataset length: {len(dataset)}")
        print(f"Class distribution: {dataset.get_class_distribution()}")
        
        # 샘플 로드 테스트
        img, target = dataset[0]
        print(f"Image shape: {img.shape}, Target: {target}")
        
        print("\n=== Memory Efficient Dataset Test ===")
        
        # 메모리 효율적 데이터셋 테스트
        mem_dataset = MemoryEfficientImageDataset(csv_path, img_dir, cache_images=True)
        
        # 여러 번 접근해서 캐시 효과 확인
        for i in range(10):
            _ = mem_dataset[i % len(mem_dataset)]
        
        cache_stats = mem_dataset.get_cache_stats()
        print(f"Cache stats: {cache_stats}")
