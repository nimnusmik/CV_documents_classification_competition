# 📘 Training Pipeline 실행 가이드 (🚀 HIGH-PERFORMANCE VERSION)

## 1) 실행 명령어

### 🏆 고성능 모드 (권장 - F1 ~0.934 목표)
```bash
python -m src.training.train_main --config configs/train_highperf.yaml --mode highperf
```

### 📚 기본 모드 (기존 버전)
```bash
python -m src.training.train_main --config configs/train.yaml --mode basic
```

**무슨 뜻?**

- `m src.training.train_main` → `src/training/train_main.py` 모듈을 **엔트리포인트**로 실행합니다.
- `--config configs/train_highperf.yaml` → 고성능 학습용 **설정 파일 경로**를 전달합니다.
- `--mode highperf` → **고성능 모드** 활성화 (Swin Transformer + Hard Aug + Mixup + WandB)

> 🎯 **성능 비교**
> - **기본 모드**: F1 ~0.87 (EfficientNet-B3 기반)
> - **고성능 모드**: F1 ~0.934 (Swin Transformer + 고급 기법들)

> ⚙️ 환경 변수
> 
> 프로젝트 루트에서 실행하면 보통 자동으로 OK입니다. 필요 시 다음을 먼저 실행하세요.
> 
> ```bash
> export PYTHONPATH="$(pwd):$PYTHONPATH"
> ```

---

## 2) 실행 전 체크리스트

- 🗂️ **데이터 배치**
    - `data/raw/train.csv` : 학습 메타 (필수 컬럼: `ID`, `target`)
    - `data/raw/sample_submission.csv` : 제출 포맷 참고용
    - `data/raw/train/` : **학습 이미지 폴더**
    - `data/raw/test/` : (추론용 이미지 폴더 — *학습 가이드에서는 사용하지 않음*)
- 🔤 **파일명/확장자**
    - CSV의 `ID`가 `abc123.jpg`처럼 **확장자를 포함**하든, `abc123`처럼 **없든** 모두 안전 처리됩니다. (`.jpg.jpg` 방지)
- 🧱 **디렉토리**
    - `logs/`, `experiments/` 폴더는 실행 중 **자동 생성**됩니다.
- 🧬 **재현성**
    - 시드(`project.seed`) 고정, 설정 스냅샷(`experiments/.../config.yaml`) 저장으로 **완전 재현**을 지원합니다.

---

## 3) 실행 시 내부 흐름 (모듈 & 함수 호출 순서)

### (1) 엔트리포인트: `src/training/train_main.py`

1. 🧭 `argparse`로 CLI 인자 파싱 (`-config`)
2. 🚀 `run_training(config_path)` 호출
3. 🏁 종료 상태를 **콘솔**에 출력
    - 정상: `[EXIT] training finished successfully (see logs/* for details)`
    - 에러: `[EXIT][ERROR] training failed: ...`

### (2) 핵심 파이프라인: `src/training/train.py` → `run_training(cfg_path)`

1. 📄 **Config 로드 & 표시**
    - `load_yaml`로 YAML 로드 → `[CFG] ...` 섹션 로그 출력
    - 예: 모델/학습/데이터/출력 경로 등 설정값을 전부 로그에 남김
2. 🎲 **Seed 고정**
    - `set_seed` 호출 → PyTorch/NumPy/랜덤 시드 고정
3. 📝 **Logger 시작 + 표준 입·출력 리디렉션**
    - `logs/run_YYYYMMDD-HHMM_<run_id>.log` 생성
    - 모든 `print`/오류가 **로그 파일에도** 기록
    - 시작: `>> Logger started: ...` / 종료: `>> Stopping logger and restoring stdio`
4. 📂 **경로 검증 & 데이터 로드**
    - `require_file/require_dir`로 `train.csv`, `image_dir_train` 등 **필수 경로 확인**
    - `[PATH] OK | train_csv=... | sample_csv=... | image_dir_train=...`
    - `pd.read_csv(train_csv)` → 필수 컬럼(`ID`, `target`) 점검
5. 🔀 **K-Fold 분할**
    - `StratifiedKFold(n_splits=folds)` 또는 기존 `fold` 열 검증
    - `[FOLD] distribution={0:..., 1:..., ...}` 로 분배 로그
6. 📦 **아티팩트 디렉토리 생성**
    - `experiments/YYYYMMDD/<run_id>/` 생성
    - 포함:
        - `ckpt/` (체크포인트)
        - `metrics.jsonl` (에폭별 지표: loss/F1/lr/시간/메모리)
        - `config.yaml` (실행 스냅샷)
    - `[ARTIFACTS] ...`로 경로를 명확히 로깅
7. 🧰 **데이터로더 & 모델 빌드**
    - `DocClsDataset` + `DataLoader` 구성 (`_build_loaders`)
    - `build_model`로 timm 기반 백본 생성 (`global_pool` 정석 매핑)
    - `Adam/AdamW`, `CosineAnnealingLR` 등 옵티마/스케줄러 초기화 로그
8. 🔁 **학습/검증 루프**
    - `train_one_epoch` : 스텝별 loss, lr, 메모리 등 촘촘 로그
    - `validate` : `macro_f1`/loss 계산, 요약 로그
    - Best 갱신 시 `ckpt/best_fold{n}.pth` 저장 + `NEW_BEST` 로그
    - 각 에폭 결과는 `metrics.jsonl`에 **JSON Lines**로 누적
9. 📚 **모드 분기**
    - `data.valid_fold: int` → **단일 폴드 학습**
    - `data.valid_fold: "all"` → **전 폴드 순회(K-Fold 학습)**
        - OOF 결과(`oof_logits.npy`, `oof_targets.npy`) 저장
10. ✅ **종료 마커**
    - 파이프라인 정상 종료: `[BOOT] training pipeline finished successfully`
    - **항상 기록되는 최종 마커**:
        - 정상: `[EXIT] TRAINING SUCCESS code=0`
        - 에러: `[EXIT] TRAINING ERROR code=1` (Traceback 포함)

---

## 4) 결과물 & 디렉토리 구조

### 🧾 로그

- 위치: `logs/run_YYYYMMDD-HHMM_<run_id>.log`
- 주요 태그:
    
    `[BOOT]`, `[CFG]`, `[PATH]`, `[DATA]`, `[MODEL]`, `[EPOCH]`, `NEW_BEST`, `[DONE]`, `[EXIT]`
    

### 🧪 실험 아티팩트

- 위치: `experiments/YYYYMMDD/<run_id>/`

```
experiments/
└── 20250904/
    └── v087-8c206e/
        ├── ckpt/
        │   ├── best_fold0.pth
        │   ├── best_fold1.pth
        │   ├── best_fold2.pth
        │   ├── best_fold3.pth
        │   └── best_fold4.pth
        ├── oof/
        │   ├── oof_logits.npy
        │   └── oof_targets.npy
        ├── config.yaml
        └── metrics.jsonl
```

**파일 설명**

- `ckpt/best_fold*.pth` : 폴드별 **최고 성능** 시점의 가중치
- `metrics.jsonl` : 각 에폭의 `train_loss`, `valid_loss`, `macro_f1`, `lr`, `time_s`, `mem_MiB` 기록
- `config.yaml` : 실행 당시 설정 스냅샷 (재현성)
- `oof/*.npy` : 전 폴드 OOF 검증 결과(선택 저장)

---

## 5) 파일 간 관계(의존 다이어그램)

```mermaid
flowchart TD

A[train_main.py] -->|run_training| B[train.py]

B -->|config load| C[utils/common.py]
B -->|set_seed| D[utils/seed.py]
B -->|Logger| E[utils/logger.py]
B -->|Dataset| F[data/dataset.py]
B -->|Transforms| G[data/transforms.py]
B -->|Model build| H[models/build.py]
B -->|Metrics| I[metrics/f1.py]

```

---

## 6) 설정(`train_v087.yaml`) 핵심 키 설명

- 🌳 `data.*`
    - `train_csv` : 학습 CSV 경로 (예: `../data/raw/train.csv`)
    - `sample_csv` : 샘플 제출 CSV (경로 검증용)
    - `image_dir_train` : **학습 이미지 폴더** (예: `../data/raw/train`)
    - `image_ext` : 확장자 기본값 (`.jpg` 등).
        
        → CSV `ID`에 이미 확장자가 있으면 **추가하지 않음**. 없으면 붙임. 그래도 못 찾으면 `.jpg/.png/...` **후보 확장자 자동 탐색**
        
    - `id_col`, `target_col` : 컬럼명 지정 (기본 `ID`, `target`)
    - `folds` / `valid_fold` : 폴드 수 / **`int`(단일 폴드) 또는 `"all"`(전 폴드)**
    - `stratify` : 층화 여부
- 🧠 `model.*`
    - `name` : timm 모델명 (예: `efficientnet_b3`)
    - `pretrained` : 사전학습 가중치 사용
    - `pooling` : `avg`/`gem`/`max`/… → timm `global_pool`로 매핑
- 🏋️ `train.*`
    - `img_size`, `batch_size`, `epochs`, `lr`, `weight_decay`, `optimizer`, `scheduler`, `amp`, `grad_clip_norm`, `label_smoothing`
    - `log_interval` : 미니배치 로그 간격
- 🗃️ `output.*`
    - `logs_dir` : 로그 디렉토리
    - `exp_dir` : 실험(아티팩트) 디렉토리
    - `snapshots` : 설정 스냅샷 저장 여부

> 📌 경로 해석 규칙
> 
> 
> 현재 예시는 **config 파일 기준 상대경로**(`../data/...`)를 사용합니다.
> 
> → `configs/train_v087.yaml`에서 `../data/...`는 레포 루트의 `data/...`를 가리킵니다.
> 

---

## 7) 로그 판독 요령 (현업용 포인트)

- ✅ **정상 시작 지표**
    - `[PATH] OK ...` : CSV/이미지 경로 검증 통과
    - `[DATA] dataset sizes | train=... valid=...` : 샘플 수 확인
    - `[MODEL] name=... pooling=... params(total/trainable)=...` : 모델/파라미터 확인
- 📈 **에폭별 핵심**
    - `[EPOCH n] ... loss=... lr=...`
    - `validate` 후 `macro_f1=...`
    - `NEW_BEST F1=... -> ckpt/best_foldX.pth` : 최고 성능 저장 포인트
- 🏁 **마지막 줄(가장 중요)**
    - 정상 종료: `[EXIT] TRAINING SUCCESS code=0`
    - 에러 종료: `[EXIT] TRAINING ERROR code=1` (+ Traceback)

---

## 8) 트러블슈팅 (증상 → 조치)

- ❌ `FileNotFoundError: .../data/raw/train.csv`
    - `train_v087.yaml`의 경로가 **config 기준 상대경로**로 맞는지 확인
    - `../data/raw/train.csv` 형태인지 검증
- ❌ 이미지 로드 실패(`.../train/xxx.jpg.jpg`)
    - CSV `ID`에 확장자가 이미 들어있음
    - **현재 데이터셋 로직이 중복 확장 방지**하므로, 혹시 최신 코드가 아니라면 `src/data/dataset.py` 업데이트
- ❌ `AssertionError: Pooling can only be disabled ...`
    - `model.pooling`이 timm `global_pool`로 올바르게 매핑되어야 함
    - `pooling: "avg"` 권장 (커스텀 풀링을 쓰려면 외부 head 구성 필요)
- ❌ `No module named src.training.train_main`
    - `PYTHONPATH` 확인: `export PYTHONPATH="$(pwd):$PYTHONPATH"`

---

## 9) 로그 & 메트릭 활용 팁

- 🔎 `metrics.jsonl`은 **JSON Lines** 형식 → 손쉽게 집계/시각화 가능
- 예) 마지막 에폭만 추려보기(Python)
    
    ```python
    import json
    with open("experiments/20250904/v087-xxxxxx/metrics.jsonl") as f:
        rows = [json.loads(l) for l in f]
    last = [r for r in rows if isinstance(r.get("epoch"), int)]
    print(sorted(last, key=lambda x:(x["fold"], x["epoch"]))[-1])
    ```