"""
평가 지표 계산 유틸리티
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, List, Any


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    분류 성능 지표들을 종합적으로 계산
    
    Args:
        y_true (List[int]): 실제 레이블
        y_pred (List[int]): 예측 레이블
    
    Returns:
        Dict[str, float]: 각종 성능 지표들
    """
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),  # 대회 메인 지표
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
        "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def get_class_wise_f1(y_true: List[int], y_pred: List[int], num_classes: int = 17) -> Dict[str, float]:
    """
    클래스별 F1 스코어 계산
    
    Args:
        y_true (List[int]): 실제 레이블
        y_pred (List[int]): 예측 레이블  
        num_classes (int): 총 클래스 수
    
    Returns:
        Dict[str, float]: 클래스별 F1 스코어
    """
    
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    class_f1_dict = {}
    for class_idx in range(num_classes):
        if class_idx < len(f1_scores):
            class_f1_dict[f"class_{class_idx}_f1"] = f1_scores[class_idx]
        else:
            class_f1_dict[f"class_{class_idx}_f1"] = 0.0
    
    return class_f1_dict


def get_confusion_matrix_stats(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """
    혼동 행렬 기반 통계 정보
    
    Args:
        y_true (List[int]): 실제 레이블
        y_pred (List[int]): 예측 레이블
    
    Returns:
        Dict[str, Any]: 혼동 행렬 및 관련 통계
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 대각선 원소들 (정확히 분류된 샘플 수)
    correct_per_class = np.diag(cm)
    
    # 각 클래스별 전체 샘플 수
    total_per_class = np.sum(cm, axis=1)
    
    # 클래스별 정확도
    class_accuracy = correct_per_class / (total_per_class + 1e-8)
    
    return {
        "confusion_matrix": cm.tolist(),
        "correct_per_class": correct_per_class.tolist(),
        "total_per_class": total_per_class.tolist(), 
        "class_accuracy": class_accuracy.tolist(),
        "worst_class": int(np.argmin(class_accuracy)),
        "best_class": int(np.argmax(class_accuracy))
    }


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    지표들을 보기 좋게 포맷팅
    
    Args:
        metrics (Dict[str, float]): 계산된 지표들
        precision (int): 소수점 자릿수
    
    Returns:
        str: 포맷팅된 지표 문자열
    """
    
    formatted_lines = []
    
    # 주요 지표들을 먼저 표시
    main_metrics = ["accuracy", "macro_f1", "weighted_f1"]
    
    for key in main_metrics:
        if key in metrics:
            formatted_lines.append(f"{key.upper()}: {metrics[key]:.{precision}f}")
    
    # 나머지 지표들 표시
    for key, value in metrics.items():
        if key not in main_metrics:
            formatted_lines.append(f"{key.upper()}: {value:.{precision}f}")
    
    return " | ".join(formatted_lines)


if __name__ == "__main__":
    # 테스트 코드
    import random
    
    # 가상 데이터 생성
    n_samples = 1000
    n_classes = 17
    
    y_true = [random.randint(0, n_classes-1) for _ in range(n_samples)]
    y_pred = [random.randint(0, n_classes-1) for _ in range(n_samples)]
    
    # 지표 계산
    metrics = calculate_metrics(y_true, y_pred)
    class_f1 = get_class_wise_f1(y_true, y_pred)
    cm_stats = get_confusion_matrix_stats(y_true, y_pred)
    
    print("=== Overall Metrics ===")
    print(format_metrics(metrics))
    
    print("\n=== Class-wise F1 Scores ===")
    for i in range(5):  # 처음 5개 클래스만 표시
        print(f"Class {i}: {class_f1[f'class_{i}_f1']:.4f}")
    
    print(f"\n=== Confusion Matrix Stats ===")
    print(f"Best performing class: {cm_stats['best_class']}")
    print(f"Worst performing class: {cm_stats['worst_class']}")
    print(f"Average class accuracy: {np.mean(cm_stats['class_accuracy']):.4f}")
