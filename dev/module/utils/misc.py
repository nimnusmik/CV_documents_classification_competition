"""
기타 유틸리티 함수들
"""

import time
import torch
from typing import Dict, Any


def get_device_info() -> Dict[str, Any]:
    """
    현재 디바이스 정보 조회
    
    Returns:
        Dict[str, Any]: 디바이스 및 CUDA 정보
    """
    
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info.update({
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "device_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "device_memory_allocated": torch.cuda.memory_allocated(),
            "device_memory_cached": torch.cuda.memory_reserved(),
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_enabled": torch.backends.cudnn.enabled
        })
    
    return info


def format_time(seconds: float) -> str:
    """
    초를 시:분:초 형태로 포맷팅
    
    Args:
        seconds (float): 초 단위 시간
        
    Returns:
        str: 포맷팅된 시간 문자열
    """
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m {int(secs)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        mins, secs = divmod(remainder, 60)
        return f"{int(hours)}h {int(mins)}m {int(secs)}s"


def format_memory(bytes_size: int) -> str:
    """
    바이트를 읽기 쉬운 형태로 변환
    
    Args:
        bytes_size (int): 바이트 크기
        
    Returns:
        str: 포맷팅된 메모리 크기
    """
    
    if bytes_size == 0:
        return "0B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(bytes_size)
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    return f"{size:.1f}{units[unit_index]}"


def print_device_info():
    """디바이스 정보를 보기 좋게 출력"""
    
    info = get_device_info()
    
    print("=== Device Information ===")
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"Device Count: {info['device_count']}")
        print(f"Current Device: {info['current_device']}")
        print(f"Device Name: {info['device_name']}")
        print(f"Total Memory: {format_memory(info['device_memory_total'])}")
        print(f"Allocated Memory: {format_memory(info['device_memory_allocated'])}")
        print(f"Cached Memory: {format_memory(info['device_memory_cached'])}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"cuDNN Enabled: {info['cudnn_enabled']}")
    
    print("=" * 30)


class Timer:
    """
    간단한 타이머 클래스
    
    Usage:
        timer = Timer()
        timer.start()
        # ... some code ...
        elapsed = timer.stop()
        print(f"Elapsed: {timer.format()}")
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """타이머 시작"""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self) -> float:
        """타이머 종료 및 경과 시간 반환"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """경과 시간 계산"""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time
    
    def format(self) -> str:
        """경과 시간을 포맷팅해서 반환"""
        return format_time(self.elapsed())
    
    def __enter__(self):
        """Context manager 시작"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop()


if __name__ == "__main__":
    # 테스트 코드
    print_device_info()
    
    print("\n=== Timer Test ===")
    
    # 일반 사용법
    timer = Timer()
    timer.start()
    time.sleep(1.5)
    elapsed = timer.stop()
    print(f"Method 1 - Elapsed: {timer.format()}")
    
    # Context manager 사용법
    with Timer() as timer:
        time.sleep(0.5)
    print(f"Method 2 - Elapsed: {timer.format()}")
    
    # 포맷 테스트
    print(f"\nTime formatting test:")
    print(f"30 seconds: {format_time(30)}")
    print(f"90 seconds: {format_time(90)}")
    print(f"3900 seconds: {format_time(3900)}")
    
    print(f"\nMemory formatting test:")
    print(f"1024 bytes: {format_memory(1024)}")
    print(f"1536 MB: {format_memory(1536 * 1024 * 1024)}")
    print(f"2.5 GB: {format_memory(int(2.5 * 1024 * 1024 * 1024))}")
