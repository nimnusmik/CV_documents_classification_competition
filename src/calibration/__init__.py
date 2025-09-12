# src/calibration/__init__.py
"""
모델 캘리브레이션 모듈

Temperature Scaling을 통한 확률 보정 기능을 제공합니다.
"""

from .temperature_scaling import TemperatureScaling, CalibrationTrainer
from .calibration_utils import (
    calculate_ece,
    calibrate_model_ensemble,
    apply_calibration_to_predictions
)

__all__ = [
    'TemperatureScaling',
    'CalibrationTrainer', 
    'calculate_ece',
    'calibrate_model_ensemble',
    'apply_calibration_to_predictions'
]
