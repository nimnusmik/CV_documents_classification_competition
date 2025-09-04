from __future__ import annotations                  # 최신 타입 힌트(전방 참조) 사용
import os, json                                     # 경로/디렉토리 처리, JSON 유틸
import pandas as pd                                 # CSV 로드·저장 및 데이터 처리
from sklearn.model_selection import StratifiedKFold # 계층적 K-Fold 분할기

# ---------------------- Stratified K-Fold 분할 ---------------------- #
# 계층적 K-Fold 분할 생성 후 CSV 저장
def make_stratified_folds(train_csv: str, out_csv: str, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(train_csv) # 학습 CSV 로드

    # 계층 샘플링 분할기 구성
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    
    # 초기 폴드 값을 미할당 상태로 표시
    df["fold"] = -1

    # 각 분할의 검증 인덱스에 폴드 번호 할당
    for i, (_, val_idx) in enumerate(skf.split(df["ID"], df["target"])):
        # 해당 검증 세트 행에 폴드 번호 기록
        df.loc[val_idx, "fold"] = i

    assert (df["fold"] >= 0).all(), "fold assignment failed"    # 모든 행이 폴드 할당 완료 검증
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)        # 출력 디렉토리 보장
    df[["ID", "fold"]].to_csv(out_csv, index=False)             # ID와 fold만 저장
    
    # 폴드 컬럼이 추가된 전체 데이터프레임 반환
    return df
