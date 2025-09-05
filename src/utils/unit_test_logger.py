"""
단위 테스트용 로깅 및 결과 저장 유틸리티

이 모듈은 Jupyter 노트북에서 실행되는 단위 테스트의 모든 출력을 
체계적으로 로깅하고 결과를 저장하는 기능을 제공합니다.
"""

import os
import sys
import io
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Union, Dict, Any
import json


class UnitTestLogger:
    """단위 테스트용 로거 클래스"""
    
    def __init__(self, test_name: str, base_log_dir: str = "logs"):
        """
        Args:
            test_name: 테스트 이름 (예: "highperf_dataset", "mixup_augmentation")
            base_log_dir: 기본 로그 디렉토리
        """
        self.test_name = test_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 로그 디렉토리 구조 생성
        self.base_dir = Path(base_log_dir) / "notebooks" / "modular" / "unit_test" / test_name / self.timestamp
        self.log_dir = self.base_dir / "logs"
        self.image_dir = self.base_dir / "images"
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        
        # 디렉토리 생성
        for dir_path in [self.log_dir, self.image_dir, self.data_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        self.logger = self._setup_logger()
        self.test_results = {}
        self.test_start_time = datetime.now()
        
        # 출력 캡처를 위한 설정
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        self.log_info(f"단위 테스트 시작: {test_name}")
        self.log_info(f"로그 디렉토리: {self.base_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """로거 초기화"""
        logger = logging.getLogger(f"unit_test_{self.test_name}")
        logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 파일 핸들러 추가
        log_file = self.log_dir / f"{self.test_name}_{self.timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_info(self, message: str):
        """정보 로그 기록"""
        self.logger.info(message)
        print(f"📝 {message}")
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """에러 로그 기록"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True)
            print(f"❌ {message}: {str(exception)}")
        else:
            self.logger.error(message)
            print(f"❌ {message}")
    
    def log_warning(self, message: str):
        """경고 로그 기록"""
        self.logger.warning(message)
        print(f"⚠️ {message}")
    
    def log_success(self, message: str):
        """성공 로그 기록"""
        self.logger.info(f"SUCCESS: {message}")
        print(f"✅ {message}")
    
    @contextmanager
    def capture_output(self, section_name: str):
        """출력 캡처 컨텍스트 매니저"""
        captured_output = io.StringIO()
        captured_error = io.StringIO()
        
        # stdout, stderr 리다이렉트
        sys.stdout = captured_output
        sys.stderr = captured_error
        
        try:
            yield captured_output, captured_error
        finally:
            # 원래 출력으로 복원
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            # 캡처된 내용 저장
            output_content = captured_output.getvalue()
            error_content = captured_error.getvalue()
            
            if output_content:
                output_file = self.log_dir / f"{section_name}_output.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output_content)
                self.log_info(f"출력 저장: {output_file}")
            
            if error_content:
                error_file = self.log_dir / f"{section_name}_error.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(error_content)
                self.log_warning(f"에러 출력 저장: {error_file}")
            
            # 콘솔에도 출력
            if output_content:
                print(output_content)
            if error_content:
                print(error_content, file=sys.stderr)
    
    def save_figure(self, fig, filename: str, title: Optional[str] = None, dpi: int = 300):
        """matplotlib 그림 저장"""
        try:
            # 파일 확장자가 없으면 .png 추가
            if not filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
                filename += '.png'
            
            filepath = self.image_dir / filename
            
            # 제목 설정
            if title:
                fig.suptitle(f"{title} - {self.timestamp}", fontsize=12)
            
            # 저장
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
            self.log_success(f"그림 저장: {filepath}")
            
            # 메타데이터 저장
            meta_file = self.image_dir / f"{filename.split('.')[0]}_meta.json"
            metadata = {
                "filename": filename,
                "title": title,
                "timestamp": self.timestamp,
                "dpi": dpi,
                "test_name": self.test_name
            }
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return filepath
            
        except Exception as e:
            self.log_error(f"그림 저장 실패: {filename}", e)
            return None
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, description: Optional[str] = None):
        """DataFrame 저장"""
        try:
            # 파일 확장자가 없으면 .csv 추가
            if not filename.endswith(('.csv', '.xlsx', '.json')):
                filename += '.csv'
            
            filepath = self.data_dir / filename
            
            # 확장자에 따라 저장
            if filename.endswith('.csv'):
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
            elif filename.endswith('.xlsx'):
                df.to_excel(filepath, index=False)
            elif filename.endswith('.json'):
                df.to_json(filepath, orient='records', indent=2, force_ascii=False)
            
            self.log_success(f"데이터프레임 저장: {filepath} ({len(df)} 행)")
            
            # 메타데이터 저장
            meta_file = self.data_dir / f"{filename.split('.')[0]}_meta.json"
            metadata = {
                "filename": filename,
                "description": description,
                "shape": list(df.shape),
                "columns": list(df.columns),
                "timestamp": self.timestamp,
                "test_name": self.test_name
            }
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return filepath
            
        except Exception as e:
            self.log_error(f"데이터프레임 저장 실패: {filename}", e)
            return None
    
    def save_numpy_array(self, arr: np.ndarray, filename: str, description: Optional[str] = None):
        """NumPy 배열 저장"""
        try:
            if not filename.endswith('.npy'):
                filename += '.npy'
            
            filepath = self.data_dir / filename
            np.save(filepath, arr)
            
            self.log_success(f"NumPy 배열 저장: {filepath} {arr.shape}")
            
            # 메타데이터 저장
            meta_file = self.data_dir / f"{filename.split('.')[0]}_meta.json"
            metadata = {
                "filename": filename,
                "description": description,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "timestamp": self.timestamp,
                "test_name": self.test_name
            }
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return filepath
            
        except Exception as e:
            self.log_error(f"NumPy 배열 저장 실패: {filename}", e)
            return None
    
    def save_test_result(self, test_section: str, result: dict):
        """테스트 결과 저장"""
        self.test_results[test_section] = {
            **result,
            "timestamp": datetime.now().isoformat(),
            "section": test_section
        }
        
        # 개별 결과 파일 저장
        result_file = self.results_dir / f"{test_section}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results[test_section], f, indent=2, ensure_ascii=False)
        
        self.log_info(f"테스트 결과 저장: {test_section}")
    
    def save_performance_metrics(self, metrics: dict, section: str = "performance"):
        """성능 메트릭 저장"""
        metrics_with_meta = {
            "metrics": metrics,
            "test_name": self.test_name,
            "section": section,
            "timestamp": datetime.now().isoformat()
        }
        
        metrics_file = self.results_dir / f"{section}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_with_meta, f, indent=2, ensure_ascii=False)
        
        self.log_success(f"성능 메트릭 저장: {metrics_file}")
    
    def finalize_test(self):
        """테스트 완료 및 최종 결과 저장"""
        test_end_time = datetime.now()
        test_duration = (test_end_time - self.test_start_time).total_seconds()
        
        # 전체 테스트 요약
        test_summary = {
            "test_name": self.test_name,
            "start_time": self.test_start_time.isoformat(),
            "end_time": test_end_time.isoformat(),
            "duration_seconds": test_duration,
            "total_sections": len(self.test_results),
            "results": self.test_results,
            "log_directory": str(self.base_dir)
        }
        
        # 최종 요약 파일 저장
        summary_file = self.base_dir / "test_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)
        
        self.log_success(f"테스트 완료! 총 소요 시간: {test_duration:.2f}초")
        self.log_success(f"결과 요약: {summary_file}")
        
        # 최종 요약 출력
        print("\n" + "="*50)
        print(f"🏁 단위 테스트 완료: {self.test_name}")
        print("="*50)
        print(f"📁 결과 디렉토리: {self.base_dir}")
        print(f"⏱️ 소요 시간: {test_duration:.2f}초")
        print(f"📊 테스트 섹션 수: {len(self.test_results)}")
        print("="*50)
        
        return test_summary


# 편의 함수들
def create_test_logger(test_name: str) -> UnitTestLogger:
    """테스트 로거 생성 편의 함수"""
    return UnitTestLogger(test_name)


def log_test_section(logger: UnitTestLogger, section_name: str):
    """테스트 섹션 시작 로깅 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log_info(f"=== {section_name} 시작 ===")
            try:
                result = func(*args, **kwargs)
                logger.log_success(f"=== {section_name} 완료 ===")
                return result
            except Exception as e:
                logger.log_error(f"=== {section_name} 실패 ===", e)
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # 테스트 예제
    logger = create_test_logger("example_test")
    
    # 기본 로깅 테스트
    logger.log_info("테스트 로거 예제 시작")
    logger.log_success("성공 메시지 테스트")
    logger.log_warning("경고 메시지 테스트")
    
    # 그림 저장 테스트
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title("예제 그래프")
    logger.save_figure(fig, "example_plot", "예제 플롯")
    plt.close(fig)
    
    # 데이터 저장 테스트
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    logger.save_dataframe(df, "example_data", "예제 데이터프레임")
    
    # 테스트 결과 저장
    logger.save_test_result("example_section", {
        "status": "success",
        "score": 0.95,
        "details": "예제 테스트 완료"
    })
    
    # 테스트 완료
    logger.finalize_test()
