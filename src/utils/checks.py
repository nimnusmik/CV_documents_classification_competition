from __future__ import annotations                  # 최신 타입 힌트(전방 참조) 사용

import argparse, csv, json, os, sys                 # CLI 인자 파싱, CSV/JSON 처리, 파일/경로, 시스템 종료
from collections import Counter                     # 클래스 빈도 카운트
from datetime import datetime                       # 현재 시간 기록용
from pathlib import Path                            # 경로 객체화
from typing import Dict, List, Tuple                # 타입 힌트용 자료형

import pandas as pd                                 # 데이터프레임 처리

from .image_io import compute_md5, is_broken_image  # 이미지 MD5 계산, 손상 여부 판별 함수

REQUIRED_TRAIN_COLS = ["ID", "target"]                      # train.csv 필수 컬럼
REQUIRED_SUB_COLS   = ["ID", "target"]                      # 제출 CSV 필수 컬럼
LABELS = list(range(17))                                    # 라벨 0~16
EXTS   = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG") # 허용 확장자


# ------------------------------- logging ------------------------------- #
# 로그 기록 함수
def _log(path: str, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")       # 현재 시각 문자열
    Path(path).parent.mkdir(parents=True, exist_ok=True)    # 로그 경로 디렉터리 생성
    
    # 로그 파일 append 모드
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")  # 메시지 기록


# ------------------------------- cfg load ----------------------------- #
# cfg 로드 함수
def load_cfg(cfg_path: str) -> dict:
    cfg_p = Path(cfg_path).resolve()                # cfg 절대경로 변환
    
    # 파일 열기
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)  # JSON 파싱

    # 기준 후보: [설정파일 디렉터리, 프로젝트 루트(src/utils/에서 두 단계↑), 현재 작업 디렉터리]
    cfg_dir = cfg_p.parent                              # 설정파일 디렉터리
    project_root = Path(__file__).resolve().parents[2]  # .../src/utils/checks.py -> repo root
    cwd = Path.cwd()                                    # 현재 작업 디렉터리
    base_candidates = [cfg_dir, project_root, cwd]      # 기준 후보 리스트

    # 경로 해석 함수
    def _resolve(p: str) -> str:
        p = os.path.expanduser(os.path.expandvars(p))   # ~, 환경변수 확장
        q = Path(p)                                     # Path 객체 변환

        # 절대경로화
        if q.is_absolute():
            return str(q)   # 절대경로인 경우 그대로 반환
        
        # 존재 확인하며 가장 먼저 존재하는 후보를 채택
        for base in base_candidates:
            cand = (base / q)   # 후보 경로 생성

            # 존재 확인
            if cand.exists() or cand.parent.exists():
                # 가장 먼저 존재하는 후보를 채택
                return str(cand.resolve())
        
        # 마지막 폴백: 프로젝트 루트 기준으로 해석
        return str((project_root / q).resolve())

    # paths 항목 존재 시
    if "paths" in cfg:                              
        for k, v in list(cfg["paths"].items()):     # paths 순회
            cfg["paths"][k] = _resolve(v)           # 절대경로로 변환

    return cfg  # cfg 반환


# ------------------------------- indexing ----------------------------- #
# 파일 인덱스 생성
def build_file_index(img_root: str) -> dict[str, str]:
    root = Path(img_root)       # 이미지 루트 Path 객체
    index: dict[str, str] = {}  # 결과 딕셔너리
    
    # 허용 확장자 순회
    for ext in EXTS:
        # 해당 확장자 파일 탐색
        for p in root.glob(f"*{ext}"):
            # stem=ID, 절대경로 매핑
            index[p.stem] = str(p.resolve())
            
    # 매핑 반환
    return index

# 인덱스에서 ID→경로 조회
def resolve_path_from_index(_id: str, index: dict[str, str]) -> str | None:
    # ID 문자열로 조회
    return index.get(str(_id))


# ------------------------------ validators ---------------------------- #
# CSV 검증
def validate_csv_schema(cfg: dict, log_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    paths = cfg["paths"]                           # 경로 설정 가져오기
    train = pd.read_csv(paths["train_csv"])        # train.csv 로드
    sub   = pd.read_csv(paths["test_csv"])         # test_csv 로드 (제출용)
    meta  = pd.read_csv(paths["meta_csv"])         # meta.csv 로드
    issues: list[tuple[str, str]] = []             # 이슈 리스트 초기화

    # train 필수 컬럼 확인
    for c in REQUIRED_TRAIN_COLS:
        # 컬럼 없음
        if c not in train.columns:
            issues.append(("train.csv", f"missing column: {c}"))            # 이슈 추가

    # target 컬럼 존재 시
    if "target" in train.columns:
        # NaN 확인
        if train["target"].isna().any():
            issues.append(("train.csv", "target has NaN"))                  # 이슈 추가
            
        # 라벨 범위 초과 확인
        if not set(train["target"].unique()).issubset(LABELS):
            issues.append(("train.csv", "target out of range [0..16]"))     # 이슈 추가

    # meta 필수 컬럼 확인
    if not {"target", "class_name"}.issubset(meta.columns):
        issues.append(("meta.csv", "required columns: target,class_name"))  # 이슈 추가
        
    # 라벨 매핑 전체 확인
    if set(meta["target"].tolist()) != set(LABELS):
        issues.append(("meta.csv", "mapping must cover 0..16"))             # 이슈 추가

    # 제출 CSV 필수 컬럼 확인
    for c in REQUIRED_SUB_COLS:
        # 컬럼 없음
        if c not in sub.columns:
            issues.append(("sample_submission.csv", f"missing column: {c}"))# 이슈 추가

    # 이슈 존재 시
    if issues:
        _log(log_path, f"[SCHEMA] FAIL: {issues}")  # 이슈 로그 기록
    # 이슈 없음
    else:
        _log(log_path, "[SCHEMA] OK")               # 이슈 없음 로그 기록

    # 이슈 리스트 반환
    return train, sub, meta, issues


# --------------------------- file scan / hashes ----------------------- #
# 파일/해시 스캔
def scan_files_and_hashes(ids: List[str], index: dict[str, str], log_path: str) -> Tuple[Dict[str, str], list, list]:
    md5s: dict[str, str] = {}   # ID→MD5 매핑
    missing: list[str] = []     # 누락 리스트
    broken: list[str] = []      # 손상 리스트
    
    # ID 순회
    for _id in ids:
        p = resolve_path_from_index(_id, index) # 인덱스에서 경로 조회
        
        # 파일 없음
        if not p or not os.path.exists(p):
            missing.append(_id)     # 누락 리스트에 추가
            continue                # 다음 ID로
        # 이미지 손상 확인
        if is_broken_image(p):
            broken.append(_id)      # 손상 리스트에 추가
            continue                # 다음 ID로
        
        md5s[_id] = compute_md5(p)  # 정상 이미지 해시 계산

    # 누락 로그 기록
    if missing:
        _log(log_path, f"[FILES] missing={len(missing)} -> {missing[:10]} ...") # 누락 로그 기록
    # 손상 로그 기록
    if broken:
        _log(log_path, f"[FILES] broken={len(broken)} -> {broken[:10]} ...")    # 손상 로그 기록
        
    # 결과 반환
    return md5s, missing, broken


# --------------------------- dup / leakage check ---------------------- #
# 중복/리크 체크
def detect_duplicates(train_md5: Dict[str, str], test_md5: Dict[str, str], log_path: str) -> Tuple[List[str], List[str]]:
    md5_to_train: dict[str, str] = {}   # 해시→train ID 매핑
    dup_in_train: list[str] = []        # 학습 내 중복 리스트
    
    # train 해시 순회
    for tid, h in train_md5.items():
        # 중복 존재 시
        if h in md5_to_train:
            # 중복 리스트에 추가
            dup_in_train.append(tid)
        # 중복 없을 시
        else:
            # 새로운 해시 등록
            md5_to_train[h] = tid

    cross: list[str] = []   # train-test 교차 중복 리스트
    
    # test 해시 순회
    for sid, h in test_md5.items():
        # 동일 해시 발견 시
        if h in md5_to_train:
            # 교차 중복 리스트에 추가
            cross.append(f"{sid} (test) == {md5_to_train[h]} (train)")

    # 중복 로그 기록
    if dup_in_train:
        _log(log_path, f"[DUP] train internal dups={len(dup_in_train)}")
    # 리크 로그 기록
    if cross:
        _log(log_path, f"[LEAK] train-test overlaps={len(cross)}")

    # 결과 반환
    return dup_in_train, cross


# ------------------------------- outputs ------------------------------ #
# 데이터 스키마 저장
def save_data_schema(cfg: dict, train: pd.DataFrame, sub: pd.DataFrame, out_json: str) -> None:
    schema = {
        "train_rows": len(train),                                       # train 행 수
        "train_cols": list(train.columns),                              # train 컬럼 목록
        "target_labels": sorted(train["target"].unique().tolist()),     # 라벨 리스트
        "submission_rows": len(sub),                                    # 제출 행 수
        "submission_cols": list(sub.columns),                           # 제출 컬럼 목록
    }
    
    # 파일 열기
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)  # JSON 저장

# 클래스 가중치 저장
def save_class_weights(train: pd.DataFrame, out_json: str, clip_min=0.5, clip_max=3.0) -> Dict[int, float]:
    counts = Counter(train["target"].tolist())              # 라벨별 개수 집계
    med = float(pd.Series(list(counts.values())).median())  # 중앙값 계산
    
    # 가중치 계산
    weights = {c: max(clip_min, min(clip_max, med / float(counts.get(c, 1)))) for c in range(17)}
    
    # 파일 열기
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2) # JSON 저장
        
    # 가중치 반환
    return weights


# -------------------------------- main -------------------------------- #
def main():                                     # 메인 실행 함수
    ap = argparse.ArgumentParser()              # 인자 파서
    ap.add_argument("--cfg", required=True)     # cfg 인자 추가
    args = ap.parse_args()                      # 인자 파싱

    cfg = load_cfg(args.cfg)                    # cfg 로드
    paths = cfg["paths"]                        # 경로 설정

    Path(paths["processed_dir"]).mkdir(parents=True, exist_ok=True) # 디렉터리 생성
    Path(paths["exp_dir"]).mkdir(parents=True, exist_ok=True)       # 실험 결과 디렉터리 생성
    Path(paths["logs_dir"]).mkdir(parents=True, exist_ok=True)      # 로그 디렉터리 생성
    log_path = str(Path(paths["logs_dir"]) / "data_checks.log")     # 로그 경로

    train, sub, meta, issues = validate_csv_schema(cfg, log_path)   # CSV 검증

    train_index = build_file_index(paths["train_root"]) # train 인덱스 생성
    test_index  = build_file_index(paths["test_root"])  # test 인덱스 생성

    # 파일 인덱스 저장
    with open(Path(paths["processed_dir"]) / "file_index_train.json", "w", encoding="utf-8") as f:
        # 학습 인덱스 저장
        json.dump(train_index, f, ensure_ascii=False, indent=2)
    # 테스트 인덱스 저장
    with open(Path(paths["processed_dir"]) / "file_index_test.json", "w", encoding="utf-8") as f:
        # 테스트 인덱스 저장
        json.dump(test_index, f, ensure_ascii=False, indent=2)
        
    # 인덱스 로그 기록
    _log(log_path, f"[INDEX] train={len(train_index)} test={len(test_index)}")

    train_ids = train["ID"].astype(str).tolist()    # train ID 추출
    test_ids  = sub["ID"].astype(str).tolist()      # test ID 추출
    tr_md5, miss_tr, brok_tr = scan_files_and_hashes(train_ids, train_index, log_path)  # train 스캔
    te_md5, miss_te, brok_te = scan_files_and_hashes(test_ids, test_index, log_path)    # test 스캔
    dup_in_train, cross = detect_duplicates(tr_md5, te_md5, log_path)                   # 중복/리크 검사

    # 스키마 저장
    save_data_schema(cfg, train, sub, str(Path(paths["processed_dir"]) / "data_schema.json"))

    # 클래스 가중치 저장
    weights = save_class_weights(
        train,                                                      # 학습 데이터
        str(Path(paths["processed_dir"]) / "class_weights.json"),   # 클래스 가중치 저장 경로
        cfg["sampler"]["clip_min"],                                 # 클립 최소값
        cfg["sampler"]["clip_max"],                                 # 클립 최대값
    )

    issues_rows: list[dict] = []        # 이슈 레코드 리스트
    
    for f, reason in (issues or []):    # 스키마 이슈 변환
        issues_rows.append({"file": f, "issue": reason})
    for k in dup_in_train:              # 학습 중복 변환
        issues_rows.append({"file": "train", "issue": f"duplicate md5, id={k}"})
    for rec in cross:                   # 교차 리크 변환
        issues_rows.append({"file": "train-test", "issue": f"overlap {rec}"})
    if miss_tr:                         # train 누락
        issues_rows.append({"file": "train_root", "issue": f"missing={len(miss_tr)}"})
    if miss_te:                         # test 누락
        issues_rows.append({"file": "test_root", "issue": f"missing={len(miss_te)}"})
    if brok_tr:                         # train 손상
        issues_rows.append({"file": "train_root", "issue": f"broken={len(brok_tr)}"})
    if brok_te:                         # test 손상
        issues_rows.append({"file": "test_root", "issue": f"broken={len(brok_te)}"})
        
    # 이슈 존재 시 CSV 저장
    if issues_rows:
        pd.DataFrame(issues_rows).to_csv(Path(paths["processed_dir"]) / "data_issues.csv", index=False)
    
    # strict 실패 여부
    strict_fail = bool(issues_rows)
    
    # strict 모드 + 이슈 존재
    if cfg.get("strict_mode", True) and strict_fail:
        _log(log_path, "[RESULT] STRICT MODE: FAIL")    # 실패 로그
        sys.exit(1)                                     # 비정상 종료
    # 성공 로그
    else:
        _log(log_path, "[RESULT] OK")

if __name__ == "__main__":
    main()
