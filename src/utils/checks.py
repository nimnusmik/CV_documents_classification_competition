from __future__ import annotations                  # 최신 타입 힌트(전방 참조) 사용

import json, os, csv, sys, argparse                 # 설정 로드, 파일/경로, CSV, 종료 제어, CLI 파싱
from typing import Dict, List, Tuple                # 타입 명시로 코드 가독성·안정성 확보
import pandas as pd                                 # 데이터프레임 처리 표준 라이브러리
from collections import Counter                     # 클래스 분포 집계를 위한 카운터
from datetime import datetime                       # 로그 타임스탬프 생성
from .image_io import compute_md5, is_broken_image  # 이미지 MD5 해시/손상 여부 검사 유틸

REQUIRED_TRAIN_COLS = ["ID", "target"]              # 학습 CSV 필수 컬럼 정의
REQUIRED_SUB_COLS = ["ID", "target"]                # 제출 샘플 CSV 필수 컬럼 정의
LABELS = list(range(17))                            # 유효 라벨 범위 0~16

# 파일 기반 단순 로거
def _log(path: str, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시각 포맷팅
    with open(path, "a", encoding="utf-8") as f:    # 로그 파일 append 모드 오픈
        f.write(f"[{ts}] {msg}\n")                  # 타임스탬프 포함 메시지 기록

# JSON 설정 파일 로드
def load_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:  # 설정 파일 읽기
        return json.load(f)                         # 딕셔너리 형태로 반환

# CSV 스키마 검증 진입점 (주의: 실제 반환값은 4개)
def validate_csv_schema(cfg: dict, log_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = cfg["paths"]                            # 경로 설정 추출
    train = pd.read_csv(paths["train_csv"])         # 학습 CSV 로드
    sub   = pd.read_csv(paths["test_csv"])          # (주의) 여기서는 test_csv를 읽음. sample_submission 의도일 수 있음
    meta  = pd.read_csv(paths["meta_csv"])          # 메타 매핑 CSV 로드
    issues = []                                     # 검출 이슈 누적 리스트

    # train.csv 검증
    # 필수 컬럼 존재 확인
    for c in REQUIRED_TRAIN_COLS:
        if c not in train.columns:  # 컬럼 부재 체크
            issues.append(("train.csv", f"missing column: {c}"))    # 이슈 등록
            
    # 타겟 결측치 존재 여부
    if train["target"].isna().any():  
        issues.append(("train.csv", "target has NaN"))              # 이슈 등록
    
    # 타겟 값 범위 검증
    if not set(train["target"].unique()).issubset(set(LABELS)):
        issues.append(("train.csv", "target out of range [0..16]"))  # 이슈 등록

    # meta.csv 검증
    # 필수 매핑 컬럼 확인
    if not {"target","class_name"}.issubset(set(meta.columns)):  
        issues.append(("meta.csv", "required columns: target,class_name"))  # 이슈 등록
        
    # 0~16 전 라벨 커버 여부
    if set(meta["target"].tolist()) != set(LABELS):
        issues.append(("meta.csv", "mapping must cover 0..16"))     # 이슈 등록

    # sample_submission.csv 검증 가정 (현재 sub는 test_csv를 읽는 점 유의)
    for c in REQUIRED_SUB_COLS:     # 제출 샘플 필수 컬럼 확인
        if c not in sub.columns:    # 컬럼 부재 체크
            issues.append(("sample_submission.csv", f"missing column: {c}"))# 이슈 등록

    # 이슈 존재 시
    if issues:
        _log(log_path, f"[SCHEMA] FAIL: {issues}")  # 스키마 실패 로그
    # 이슈 없을 시
    else:
        _log(log_path, "[SCHEMA] OK")               # 스키마 통과 로그

    # 데이터프레임 3개와 이슈 리스트 반환(타입힌트와 불일치)
    return train, sub, meta, issues

# 파일 존재/손상/MD5 스캔
def scan_files_and_hashes(root_dir: str, ids: List[str], log_path: str) -> Dict[str, str]:
    md5s = {}                   # ID→MD5 매핑 결과
    missing, broken = [], []    # 누락/손상 ID 수집
    
    # 각 이미지 ID 순회
    for _id in ids:
        p = os.path.join(root_dir, f"{_id}.jpg")  # 파일 경로 구성
        
        # 파일 존재 여부 확인
        if not os.path.exists(p):
            missing.append(_id); continue  # 누락 수집 후 다음
        # 손상 이미지 여부 검사
        if is_broken_image(p):
            broken.append(_id); continue  # 손상 수집 후 다음
        
        # 정상 파일의 MD5 계산·저장
        md5s[_id] = compute_md5(p)
    
    # 누락 요약 로그
    if missing: _log(log_path, f"[FILES] missing={len(missing)} -> {missing[:10]} ...")
    
    # 손상 요약 로그
    if broken:  _log(log_path, f"[FILES] broken={len(broken)} -> {broken[:10]} ...")
    
    # ID별 해시 결과 반환
    return md5s

# 중복/리크 탐지
def detect_duplicates(train_md5: Dict[str,str], test_md5: Dict[str,str], log_path: str) -> Tuple[List[str], List[str]]:
    md5_to_train = {}   # MD5→train ID 역매핑
    dup_in_train = []   # 학습 내부 중복 ID 목록
    
    # 학습 MD5 순회
    for k, v in train_md5.items():  
        if v in md5_to_train:       # 동일 해시 존재 시
            dup_in_train.append(k)  # 중복 ID 기록
        else:
            md5_to_train[v] = k     # 최초 해시 매핑 등록
    
    # train-test 교차 중복(리크) 기록
    cross = []
    
    # 테스트 MD5 순회
    for k, v in test_md5.items():
        if v in md5_to_train:       # 학습에 동일 해시 존재 시
            cross.append(f"{k} (test) == {md5_to_train[v]} (train)")  # 리크 상세 기록
            
    # 학습 내부 중복 로그
    if dup_in_train: _log(log_path, f"[DUP] train internal dups={len(dup_in_train)}")
    
    # 리크 로그
    if cross: _log(log_path, f"[LEAK] train-test overlaps={len(cross)}")
    
    # 중복 목록, 리크 목록 반환
    return dup_in_train, cross

# 데이터 스키마 요약 저장
def save_data_schema(cfg: dict, train: pd.DataFrame, sub: pd.DataFrame, out_json: str) -> None:
    # 산출 메타 정보 구성
    schema = {  
        "train_rows": len(train), "train_cols": list(train.columns),        # 학습 행수·컬럼명
        "target_labels": sorted(train["target"].unique().tolist()),         # 사용 라벨 집합
        "submission_rows": len(sub), "submission_cols": list(sub.columns)   # 제출(혹은 테스트) 구조
    }
    
    # 출력 파일 오픈
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)  # JSON 저장(한글 보존)

# 클래스 가중치 산출·저장
def save_class_weights(train: pd.DataFrame, out_json: str, clip_min=0.5, clip_max=3.0) -> Dict[int, float]:
    counts = Counter(train["target"].tolist())              # 라벨별 빈도 계산
    med = float(pd.Series(list(counts.values())).median())  # 빈도 중앙값 산출
    
    # 중앙값 비례 역가중(클리핑 적용)
    weights = {c: max(clip_min, min(clip_max, med / float(counts.get(c, 1)))) for c in range(17)}
    
    # 출력 파일 오픈
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)  # 가중치 JSON 저장
        
    # 가중치 딕셔너리 반환
    return weights

def main():  # CLI 엔트리포인트
    ap = argparse.ArgumentParser()              # 인자 파서 생성
    ap.add_argument("--cfg", required=True)     # 설정 파일 경로 필수 인자
    args = ap.parse_args()                      # 인자 파싱
    cfg = load_cfg(args.cfg)                    # 설정 로드
    paths = cfg["paths"]                        # 경로 섹션 참조
    
    os.makedirs(paths["processed_dir"], exist_ok=True)                          # 산출물 디렉터리 보장
    os.makedirs(paths["exp_dir"], exist_ok=True)                                # 실험 디렉터리 보장
    os.makedirs(paths["logs_dir"], exist_ok=True)                               # 로그 디렉터리 보장
    log_path = os.path.join(paths["logs_dir"], "data_checks.log")               # 로그 파일 경로 선언

    train, sub, meta, issues = validate_csv_schema(cfg, log_path)               # 스키마 검증 및 이슈 수집
    
    # 파일/해시 스캔
    train_ids = train["ID"].astype(str).tolist()                                # 학습 ID 리스트
    test_ids  = pd.read_csv(paths["test_csv"])["ID"].astype(str).tolist()       # 테스트 ID 리스트 로드
    tr_md5 = scan_files_and_hashes(paths["train_root"], train_ids, log_path)    # 학습 파일 MD5 스캔
    te_md5 = scan_files_and_hashes(paths["test_root"],  test_ids,  log_path)    # 테스트 파일 MD5 스캔
    dup_in_train, cross = detect_duplicates(tr_md5, te_md5, log_path)           # 중복/리크 탐지

    # 산출물 저장
    # 스키마 요약 저장
    save_data_schema(cfg, train, sub, os.path.join(paths["processed_dir"], "data_schema.json"))
    
    # 클래스 가중치 산출/저장
    weights = save_class_weights(
        train,                                                                  # 학습 데이터
        os.path.join(paths["processed_dir"], "class_weights.json"),             # 클래스 가중치 저장 경로
        cfg["sampler"]["clip_min"], cfg["sampler"]["clip_max"]                  # 클리핑 범위 설정에서 읽기
    )

    # 이슈 CSV 집계 저장
    issues_rows = []  # 이슈 레코드 누적 버퍼
    
    # 스키마 이슈 변환
    for f, reason in (issues or []):
        issues_rows.append({"file": f, "issue": reason})                        # 레코드 추가
        
    # 학습 내부 중복 변환
    for k in dup_in_train:
        issues_rows.append({"file": "train", "issue": f"duplicate md5, id={k}"})# 레코드 추가
    
    # 교차 리크 변환
    for rec in cross:
        issues_rows.append({"file": "train-test", "issue": f"overlap {rec}"})   # 레코드 추가
    
    # 이슈가 하나라도 있으면
    if issues_rows:
        # CSV로 저장
        pd.DataFrame(issues_rows).to_csv(os.path.join(paths["processed_dir"], "data_issues.csv"), index=False)

    # strict 모드 종료 조건 처리
    # 엄격 모드이며 이슈 존재 시
    if cfg.get("strict_mode", True) and (issues_rows or dup_in_train or cross):
        _log(log_path, "[RESULT] STRICT MODE: FAIL")    # 실패 로그 기록
        sys.exit(1)                                     # 비정상 종료 코드 반환
        
    # 이슈 없음 또는 엄격 모드 아님
    else:
        _log(log_path, "[RESULT] OK")                   # 성공 로그 기록

# 직접 실행 시에만 동작
if __name__ == "__main__":
    main()  # 메인 함수 호출
