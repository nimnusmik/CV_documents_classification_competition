from __future__ import annotations                  # 최신 타입 힌트(전방 참조) 사용
import json                                         # JSON 파일 로드·저장
import numpy as np                                  # 수치 연산 라이브러리
import pandas as pd                                 # CSV 로드 및 데이터 처리
import torch                                        # 텐서 연산 및 학습 프레임워크
from torch.utils.data import WeightedRandomSampler  # 불균형 데이터 보정 샘플러

# ---------------------- 클래스 가중치 계산 ---------------------- #
# 클래스별 샘플 개수로부터 가중치 계산
def compute_class_weights_from_counts(counts: dict[int,int], clip_min=0.5, clip_max=3.0) -> dict[int,float]:
    med = np.median(list(counts.values()))          # 클래스 샘플 수의 중앙값 계산
    
    # 클래스별 가중치 계산: 중앙값 대비 비율, 최소/최대 값은 클리핑
    return {
        # count=0 방지 위해 max(1, count)
        c: float(np.clip(med / max(1, counts.get(c, 0)), clip_min, clip_max))
        # 총 17개 클래스(0~16)에 대해 반복
        for c in range(17)
    }

# ---------------------- Weighted Sampler 생성 ---------------------- #
# WeightedRandomSampler 생성 함수
def make_weighted_sampler(train_csv: str, weights_json: str | None = None,
                          clip_min=0.5, clip_max=3.0) -> WeightedRandomSampler:
    # 학습 CSV 로드
    df = pd.read_csv(train_csv)

    # 외부 JSON 가중치 파일이 제공된 경우
    if weights_json:
        with open(weights_json, "r") as f:                  # 가중치 JSON 파일 열기
            w = json.load(f)                                # JSON 내용 로드
        class_w = {int(k): float(v) for k,v in w.items()}   # 문자열 키를 int로 변환 후 float 값으로 저장
    
    # JSON 가중치 파일이 없는 경우 CSV 기반으로 가중치 계산
    else:
        # target 컬럼에서 클래스별 샘플 개수 집계
        counts = dict(df["target"].value_counts())
        
        # 가중치 계산
        class_w = compute_class_weights_from_counts(counts, clip_min, clip_max)

    # 각 샘플별 가중치 배열 생성
    sample_w = df["target"].map(class_w).astype(float).values

    # WeightedRandomSampler 객체 반환 (치환 샘플링: replacement=True)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),  # 가중치를 Torch Tensor(double)로 변환
        num_samples=len(sample_w),                              # 샘플 개수 지정
        replacement=True                                        # 동일 샘플 중복 선택 허용
    )
