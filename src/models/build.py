# src/models/build.py
from __future__ import annotations              # 최신 타입 힌트(전방 참조) 지원
import timm                                     # timm: 다양한 pretrained 모델 제공
import torch.nn as nn                           # PyTorch 신경망 모듈

# timm의 global_pool 허용 값 집합
# - 일부 timm 모델은 'avg'가 디폴트
_VALID_POOLS = {"avg", "avgmax", "catavgmax", "max", "gem", "token"}


# ----------------------------- 모델 생성 함수 ----------------------------- #
def build_model(
    name: str,                                  # timm 모델 이름 (예: 'efficientnet_b3')
    num_classes: int,                           # 분류 클래스 수 (0이면 head 제거)
    pretrained: bool = True,                    # pretrained 가중치 사용 여부
    drop_rate: float = 0.0,                     # dropout 비율
    drop_path_rate: float = 0.0,                # stochastic depth 비율
    pooling: str | None = "avg",                # 전역 풀링 방식 (None이면 비활성화)
):
    """
    모델 생성 유틸 함수
    - name: timm 모델 이름
    - num_classes: 분류 클래스 수 (0이면 classification head 제거)
    - pretrained: 사전학습 가중치 사용 여부
    - drop_rate: dropout 비율
    - drop_path_rate: stochastic depth 비율
    - pooling: timm global_pool 옵션 (avg 권장).
               None/'none'이면 전역 풀링 비활성화됨.
               단, num_classes > 0 인 경우 pooling을 반드시 지정해야 함.
    """

    # ---------------- pooling 정규화 ---------------- #
    if pooling is None:                         # None 명시 시
        pooling_norm = None
    else:
        p = str(pooling).strip().lower()        # 문자열 소문자 정규화
        if p in ("none", "disable", "disabled", "off", ""):
            # 비활성화 키워드 입력 시 None 처리
            pooling_norm = None
        else:
            if p not in _VALID_POOLS:           # 지원하지 않는 값이면
                pooling_norm = "avg"            # 안전하게 'avg'로 강제
            else:
                pooling_norm = p                # 정상 값이면 그대로 사용

    # ---------------- num_classes & pooling 조합 검사 ---------------- #
    if num_classes and num_classes > 0 and pooling_norm is None:
        # 분류 헤드를 쓰려면 pooling이 필요
        # pooling이 없으면 conv classifier 직접 구현 필요
        raise ValueError(
            "pooling(None/none) + num_classes>0 조합은 timm에서 허용되지 않습니다. "
            "config의 model.pooling을 'avg'(권장)로 설정하거나 num_classes=0(헤드 제거) 후 "
            "외부 분류기를 붙이세요."
        )

    # ---------------- timm create_model 인자 구성 ---------------- #
    # - global_pool: pooling_norm이 None이면 전달하지 않음 (timm 기본값 사용)
    create_kwargs = dict(
        pretrained=pretrained,                                      # 사전학습 가중치 사용 여부
        num_classes=num_classes if num_classes is not None else 0,  # 분류 클래스 수
        drop_rate=drop_rate,                                        # dropout 비율
        drop_path_rate=drop_path_rate,                              # stochastic depth 비율
    )

    # pooling_norm이 None이 아닐 때만 global_pool 인자 추가
    if pooling_norm is not None:
        create_kwargs["global_pool"] = pooling_norm  # timm 표준 인자

    # timm 모델 생성
    model = timm.create_model(name, **create_kwargs)

    # 필요 시 head 교체/동결 등 추가 확장 가능
    return model