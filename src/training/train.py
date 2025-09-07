# ------------------------- 표준 라이브러리 ------------------------- #
import os, time, numpy as np, torch, torch.nn as nn, pandas as pd, psutil
# os       : 파일/디렉터리 경로, 시스템 유틸
# time     : 시간 측정, 로깅
# numpy    : 수치 계산, 배열 연산
# torch    : PyTorch 메인 모듈
# torch.nn : 신경망 계층/손실 함수 모듈
# pandas   : 데이터프레임 처리
# psutil   : 시스템 메모리 사용량 추적

# ------------------------- PyTorch 유틸 ------------------------- #
from torch.utils.data import DataLoader                             # 데이터 로더
from sklearn.model_selection import StratifiedKFold                 # 계층적 K-폴드 분할
from torch.cuda.amp import autocast, GradScaler                     # AMP (자동 혼합 정밀도) 지원
from torch.optim import Adam, AdamW                                 # 옵티마이저 (Adam, AdamW)
from torch.optim.lr_scheduler import CosineAnnealingLR              # 학습률 스케줄러 (코사인 감쇠)
from tqdm import tqdm                                # 진행바 시각화

# ------------------------- 프로젝트 유틸 ------------------------- #
from src.utils.seed import set_seed                                 # 랜덤 시드 고정
from src.logging.logger import Logger                               # 로그 기록 클래스
from src.utils.common import (                                      # 공통 유틸 함수들
    load_yaml, ensure_dir, dump_yaml, jsonl_append, short_uid,
    resolve_path, require_file, require_dir, create_log_path
)

# ------------------------- 데이터/모델 관련 ------------------------- #
from src.data.dataset import DocClsDataset                          # 문서 분류 Dataset 클래스
from src.data.transforms import (                                    # 학습/검증 변환 함수들
    build_train_tfms, build_valid_tfms, build_advanced_train_tfms   # 기본/고급 변환 파이프라인
)
from src.models.build import build_model                            # 모델 생성기
from src.metrics.f1 import macro_f1_from_logits                     # 매크로 F1 스코어 계산 함수


# ---------------------------
# helpers
# ---------------------------

# ---------------------- 실행 디렉토리/아티팩트 생성 ---------------------- #
def _make_run_dirs(cfg, run_id, logger):
    # 날짜 문자열 포맷팅 (예: 20250101)
    day = time.strftime(cfg["project"]["date_format"])
    # 시간 문자열 포맷팅 (예: 1530)
    time_str = time.strftime(cfg["project"]["time_format"])
    # 타임스탬프 포함된 폴더명 생성 (예: swin-highperf_20250907_1530)
    folder_name = f"{cfg['project']['run_name']}_{day}_{time_str}"
    # 실험 루트 디렉터리 생성
    exp_root = ensure_dir(os.path.join(cfg["output"]["exp_dir"], day, folder_name))
    # 체크포인트 저장 디렉터리 생성
    ckpt_dir = ensure_dir(os.path.join(exp_root, "ckpt"))
    # 메트릭 기록 파일 경로
    metrics_path = os.path.join(exp_root, "metrics.jsonl")
    # 설정 스냅샷 저장 경로
    cfg_path = os.path.join(exp_root, "config.yaml")
    # 현재 설정을 YAML로 저장
    dump_yaml(cfg, cfg_path)
    
    # 로그 기록
    logger.write(f"[ARTIFACTS] exp_root={exp_root}")            # 실험 루트 디렉터리
    logger.write(f"[ARTIFACTS] ckpt_dir={ckpt_dir}")            # 체크포인트 디렉터리
    logger.write(f"[ARTIFACTS] metrics_path={metrics_path}")    # 메트릭 기록 파일 경로
    logger.write(f"[ARTIFACTS] cfg_snapshot={cfg_path}")        # 설정 스냅샷 저장 경로
    # 경로 반환
    return exp_root, ckpt_dir, metrics_path, cfg_path

# ---------------------- 로거 생성 ---------------------- #
def _make_logger(cfg, run_id):
    # 증강 타입에 따른 로그 파일명 생성
    aug_type = "advanced_augmentation" if cfg["train"].get("use_advanced_augmentation", False) else "basic_augmentation"
    log_name = f"train_{time.strftime('%Y%m%d-%H%M')}_{cfg['project']['run_name']}_{aug_type}.log"
    # 날짜별 로그 파일 전체 경로
    log_path = create_log_path("train", log_name)
    # Logger 객체 생성
    logger = Logger(log_path)
    # 표준 입출력 리다이렉트 시작
    logger.start_redirect()
    
    # tqdm 출력도 로거에 리다이렉트할 수 있는 경우
    if hasattr(logger, "tqdm_redirect"):
        logger.tqdm_redirect()  # tqdm 출력 리다이렉트
        
    # 로그 시작 메시지 기록
    logger.write(f">> Logger started: {log_path}")
    
    # logger 반환
    return logger

# ---------------------- 디바이스 선택 ---------------------- #
def _device(cfg):
    # cfg에 'cuda' 지정 && CUDA 사용 가능 시 'cuda', 아니면 'cpu'
    return "cuda" if (cfg["project"]["device"]=="cuda" and torch.cuda.is_available()) else "cpu"

# ---------------------- 모델 파라미터 수 계산 ---------------------- #
def _count_params(model):
    # 전체 파라미터 수
    total = sum(p.numel() for p in model.parameters())
    # 학습 가능한 파라미터 수
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 튜플로 반환
    return total, trainable

# ---------------------- 옵티마이저/스케줄러 설정 ---------------------- #
def _opt_and_sch(params, cfg, steps_per_epoch, logger):
    # 옵티마이저 이름 소문자 변환
    opt_name = cfg["train"]["optimizer"].lower()
    # 학습률과 weight decay 가져오기
    lr = cfg["train"]["lr"]; wd = cfg["train"]["weight_decay"]
    # AdamW 또는 Adam 옵티마이저 생성
    opt = AdamW(params, lr=lr, weight_decay=wd) if opt_name=="adamw" else Adam(params, lr=lr, weight_decay=wd)
    # CosineAnnealingLR 스케줄러 생성 (epochs * steps 기준)
    sch = CosineAnnealingLR(opt, T_max=max(1, cfg["train"]["epochs"]*max(1, steps_per_epoch))) \
          if cfg["train"]["scheduler"]=="cosine" else None
          
    # 로그 기록
    logger.write(
        f"[OPTIM] optimizer={opt.__class__.__name__}, lr={lr}, weight_decay={wd}, " # 옵티마이저
        f"scheduler={sch.__class__.__name__ if sch else 'none'}"                    # 스케줄러
    )
    
    # 옵티마이저와 스케줄러 반환
    return opt, sch

# ---------------------- 한 에폭 학습 ---------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    logger, epoch, max_grad_norm=None, log_interval=50):
    # 모델을 학습 모드로 전환
    model.train()
    # 손실값 누적 초기화
    running_loss = 0.0
    # 현재 프로세스 핸들 (메모리 사용량 추적용)
    p = psutil.Process(os.getpid())
    # 에폭 시작 로그 기록
    logger.write(f"[EPOCH {epoch}] >>> TRAIN start | steps={len(loader)}")

    # 배치 단위 학습 루프
    for step, (imgs, labels) in enumerate(loader, 1):
        # 데이터를 GPU/CPU 디바이스로 이동
        imgs, labels = imgs.to(device), labels.to(device)
        # 옵티마이저 gradient 초기화
        optimizer.zero_grad(set_to_none=True)

        # 자동 혼합정밀(AMP) 학습 영역
        with autocast(enabled=scaler is not None):
            # 모델 forward → 예측 로짓
            logits = model(imgs)
            # 손실 계산
            loss = criterion(logits, labels)

        # AMP 활성화된 경우
        if scaler:
            # 손실 스케일링 후 backward
            scaler.scale(loss).backward()
            # gradient clipping 필요할 경우
            if max_grad_norm:
                scaler.unscale_(optimizer)  # clip 전에 unscale 필요
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # 옵티마이저 스텝 및 스케일러 업데이트
            scaler.step(optimizer)
            scaler.update()
        # AMP 비활성화된 경우 (일반 학습)
        else:
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # 누적 손실 업데이트 (배치 크기 반영)
        running_loss += loss.item() * imgs.size(0)

        # 로그 출력 간격마다 기록
        if (step % max(1, log_interval)) == 0 or step == 1 or step == len(loader):
            lr = optimizer.param_groups[0]["lr"]
            logger.write(
                f"[EPOCH {epoch}][TRAIN step {step}/{len(loader)}] "
                f"loss={loss.item():.5f} lr={lr:.6f} bs={imgs.size(0)}"
            )

    # 에폭 종료 후 메모리 사용량(MB)
    mem = p.memory_info().rss / (1024*1024)
    # 평균 손실 계산
    epoch_loss = running_loss / len(loader.dataset)
    # 에폭 종료 로그 기록
    logger.write(f"[EPOCH {epoch}] <<< TRAIN end | loss={epoch_loss:.5f} mem={mem:.0f}MiB")
    # 에폭 손실과 메모리 사용량 반환
    return epoch_loss, mem


# ---------------------- 검증 ---------------------- #
@torch.no_grad()  # 검증은 gradient 계산 비활성화
def validate(model, loader, criterion, device, logger, epoch=None):
    # phase 이름 정의 (EVAL 또는 EPOCH n)
    phase = f"EPOCH {epoch}" if epoch is not None else "EVAL"
    # 검증 시작 로그
    logger.write(f"[{phase}] >>> VALID start | steps={len(loader)}")
    # 모델을 평가 모드로 전환
    model.eval()
    # 누적 손실 초기화
    running_loss = 0.0
    # 로짓과 타깃 저장 리스트
    all_logits = []
    all_targets = []

    # 배치 단위 검증 루프
    for step, (imgs, labels) in enumerate(loader, 1):
        # 데이터를 GPU/CPU 디바이스로 이동
        imgs, labels = imgs.to(device), labels.to(device)
        # forward → 로짓 계산
        logits = model(imgs)
        # 손실 계산
        loss = criterion(logits, labels)
        # 누적 손실 업데이트
        running_loss += loss.item() * imgs.size(0)
        # 결과 저장 (CPU로 이동)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    # 전체 배치 로짓/타깃 결합
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    # macro-F1 계산
    f1 = macro_f1_from_logits(logits, targets)
    # 평균 손실 계산
    epoch_loss = running_loss / len(loader.dataset)
    # 검증 종료 로그 기록
    logger.write(f"[{phase}] <<< VALID end | loss={epoch_loss:.5f} macro_f1={f1:.5f}")
    # 손실, F1, 로짓, 타깃 반환
    return epoch_loss, f1, logits, targets


# ---------------------- DataLoader 빌드 ---------------------- #
def _build_loaders(cfg, trn_df, val_df, image_dir, logger):
    # 데이터셋/로더 생성 로그 기록
    logger.write(
        f"[DATA] build loaders | img_dir={image_dir} | "
        f"img_size={cfg['train']['img_size']} | bs={cfg['train']['batch_size']}"
    )

    # 변환 함수 선택 (고급 증강 vs 기본 증강)
    use_advanced = cfg["train"].get("use_advanced_augmentation", False)  # 기본값: False
    train_transform_fn = build_advanced_train_tfms if use_advanced else build_train_tfms
    
    logger.write(f"[DATA] augmentation type: {'advanced' if use_advanced else 'basic'}")

    # 학습용 데이터셋 생성
    train_ds = DocClsDataset(
        trn_df,                                   # 학습 데이터프레임
        image_dir,                                # 이미지 디렉터리
        cfg["data"]["image_ext"],                 # 이미지 확장자
        cfg["data"]["id_col"],                    # ID 컬럼명
        cfg["data"]["target_col"],                # 타깃 컬럼명
        train_transform_fn(cfg["train"]["img_size"])  # 선택된 학습용 변환 파이프라인
    )

    # 검증용 데이터셋 생성
    valid_ds = DocClsDataset(
        val_df,                                   # 검증 데이터프레임
        image_dir,                                # 이미지 디렉터리
        cfg["data"]["image_ext"],                 # 이미지 확장자
        cfg["data"]["id_col"],                    # ID 컬럼명
        cfg["data"]["target_col"],                # 타깃 컬럼명
        build_valid_tfms(cfg["train"]["img_size"])# 검증용 변환 파이프라인
    )

    # 데이터셋 크기 로그 기록
    logger.write(f"[DATA] dataset sizes | train={len(train_ds)} valid={len(valid_ds)}")

    # 학습용 DataLoader 생성
    train_ld = DataLoader(
        train_ds,                                 # 학습 데이터셋
        batch_size=cfg["train"]["batch_size"],    # 배치 크기
        shuffle=True,                             # 무작위 섞기
        num_workers=cfg["project"]["num_workers"],# 워커 수
        pin_memory=True,                          # GPU 메모리 핀ning
        drop_last=False                           # 마지막 배치 유지
    )

    # 검증용 DataLoader 생성
    valid_ld = DataLoader(
        valid_ds,                                 # 검증 데이터셋
        batch_size=cfg["train"]["batch_size"],    # 배치 크기
        shuffle=False,                            # 순서 유지
        num_workers=cfg["project"]["num_workers"],# 워커 수
        pin_memory=True,                          # GPU 메모리 핀ning
        drop_last=False                           # 마지막 배치 유지
    )

    # 학습/검증 로더 반환
    return train_ld, valid_ld


# ---------------------- 모델 빌드 ---------------------- #
def _build_model(cfg, device, logger):
    # 모델 생성
    model = build_model(
        cfg["model"]["name"],                     # 모델 이름
        cfg["data"]["num_classes"],               # 클래스 개수
        cfg["model"]["pretrained"],               # 사전학습 여부
        cfg["model"]["drop_rate"],                # Dropout 비율
        cfg["model"]["drop_path_rate"],           # DropPath 비율
        cfg["model"]["pooling"]                   # 풀링 방식
    ).to(device)                                  # 디바이스에 로드

    # 파라미터 개수 계산
    total, trainable = _count_params(model)

    # 모델 관련 로그 출력
    logger.write(
        f"[MODEL] name={cfg['model']['name']} "         # 모델 이름
        f"pretrained={cfg['model']['pretrained']} "     # 사전학습 여부
        f"pooling={cfg['model']['pooling']} "           # 풀링 방식
        f"params(total/trainable)={total}/{trainable}"  # 파라미터 개수
    )

    # 모델 반환
    return model


# ---------------------- 데이터 분할 (폴드) ---------------------- #
def _split_folds(df, cfg, logger):
    # 폴드 준비 로그 기록
    logger.write(
        f"[FOLD] preparing | folds={cfg['data']['folds']} " # 폴드 수
        f"stratify={cfg['data'].get('stratify', True)}"     # 계층화 여부
    )

    # 폴드 수
    folds = cfg["data"]["folds"]

    # fold 컬럼이 없거나, 값이 유효하지 않을 경우 새로 생성
    if "fold" not in df.columns or (df["fold"] < 0).any() or (df["fold"] >= folds).any():
        # stratify 모드인 경우
        if cfg["data"].get("stratify", True):
            # StratifiedKFold 객체 생성
            skf = StratifiedKFold(
                n_splits=folds,                     # 폴드 수
                shuffle=True,                       # 데이터 섞기 여부
                random_state=cfg["project"]["seed"] # 랜덤 시드
            )
            
            # fold 초기화
            df["fold"] = -1
            
            # fold 분리 및 할당
            for f, (_, v_idx) in enumerate(skf.split(df, df[cfg["data"]["target_col"]])):
                # 각 검증 인덱스에 폴드 번호 할당
                df.loc[df.index[v_idx], "fold"] = f

        # 비-stratify 모드인 경우
        else:
            # 전체 인덱스
            idx = np.arange(len(df))

            # fold 분리 및 할당
            for f in range(folds):
                # 각 폴드에 해당하는 인덱스 할당
                df.loc[idx[f::folds], "fold"] = f

    # 분포 계산
    dist = df["fold"].value_counts().sort_index().to_dict()
    # 로그 출력
    logger.write(f"[FOLD] distribution={dist}")

    # 폴드가 할당된 데이터프레임 반환
    return df

# =========================
# 공개 진입 함수
# =========================

# 학습 파이프라인 실행
def run_training(cfg_path: str):
    # ---------------------- 설정 로드 ---------------------- #
    cfg = load_yaml(cfg_path)                                         # YAML 설정 로드
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))              # cfg 파일이 위치한 절대 경로
    cfg.setdefault("train", {}).setdefault("log_interval", 50)        # 로그 간격 기본값 설정

    # ---------------------- 시드 및 실행 ID ---------------------- #
    set_seed(cfg["project"]["seed"])                                  # 랜덤 시드 고정
    run_id = f'{cfg["project"]["run_name"]}-{short_uid()}'            # 실행 ID 생성
    logger = _make_logger(cfg, run_id)                                # 로거 생성
    logger.write("[BOOT] training pipeline started")                  # 파이프라인 시작 로그 기록

    exit_status = "SUCCESS"                                           # 기본 종료 상태
    exit_code = 0                                                     # 기본 종료 코드

    try:
        # ---------------------- 디바이스 설정 ---------------------- #
        device = _device(cfg)                                         # GPU/CPU 결정
        logger.write(f"[BOOT] device={device}")                       # 디바이스 로그 출력
        logger.write(f"[CFG] loaded config from {cfg_path}")          # 설정 로드 로그
        logger.write(f"[CFG] data section: {cfg['data']}")            # 데이터 설정 출력
        logger.write(f"[CFG] train section: {cfg['train']}")          # 학습 설정 출력
        logger.write(f"[CFG] model section: {cfg['model']}")          # 모델 설정 출력
        logger.write(f"[CFG] output section: {cfg['output']}")        # 출력 설정 출력

        # ---------------------- 경로 확인 ---------------------- #
        train_csv = resolve_path(cfg_dir, cfg["data"]["train_csv"])   # 학습 CSV 경로 확인
        sample_csv = resolve_path(cfg_dir, cfg["data"]["sample_csv"]) # 제출 CSV 경로 확인
        image_dir = resolve_path(cfg_dir,                             # 이미지 디렉터리 확인
                                 cfg["data"].get("image_dir_train",
                                 cfg["data"].get("image_dir", "data/raw/train")))
        require_file(train_csv,  "data.train_csv 확인")                # 학습 CSV 존재 확인
        require_file(sample_csv, "data.sample_csv 확인")               # 제출 CSV 존재 확인
        require_dir(image_dir,   "data.image_dir_train 확인")          # 이미지 디렉터리 존재 확인

        # 경로 확인 로그 출력
        logger.write(f"[PATH] OK | train_csv={train_csv} | sample_csv={sample_csv} | image_dir_train={image_dir}")

        # ---------------------- 데이터 로드 및 검증 ---------------------- #
        # 학습 데이터 로드
        df = pd.read_csv(train_csv)
        # 필수 컬럼 정의 (ID, target)
        need_cols = (cfg["data"]["id_col"], cfg["data"]["target_col"])
        # CSV 컬럼 로그 출력
        logger.write(f"[DATA] columns={list(df.columns)} | required={need_cols}")
        # 필수 컬럼 누락 시 예외 발생
        if cfg["data"]["id_col"] not in df.columns or cfg["data"]["target_col"] not in df.columns:
            raise KeyError(f"CSV 열({cfg['data']['id_col']}, {cfg['data']['target_col']}) 누락")
        # 데이터 프레임 폴드 분할 (StratifiedKFold)
        df = _split_folds(df, cfg, logger)

        # ---------------------- 아티팩트 디렉토리 ---------------------- #
        # # 산출물 경로 생성
        exp_root, ckpt_dir, metrics_path, _ = _make_run_dirs(cfg, run_id, logger)

        # ---------------------- 학습 모드 확인 ---------------------- #
        valid_fold = cfg["data"]["valid_fold"]          # valid_fold 값 확인
        logger.write(f"[MODE] valid_fold={valid_fold}") # 모드 로그 출력

        # ---------------------- 단일 폴드 학습 ---------------------- #
        if isinstance(valid_fold, int):
            trn = df[df["fold"]!=valid_fold].reset_index(drop=True)                 # 학습 데이터
            val = df[df["fold"]==valid_fold].reset_index(drop=True)                 # 검증 데이터
            logger.write(f"[FOLD {valid_fold}] train={len(trn)} valid={len(val)}")  # 데이터 크기 로그

            train_ld, valid_ld = _build_loaders(cfg, trn, val, image_dir, logger)   # DataLoader 생성
            model = _build_model(cfg, device, logger)                               # 모델 빌드
            criterion = nn.CrossEntropyLoss()                                       # 손실 함수
            scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))                  # AMP 스케일러
            
            # 옵티마이저+스케줄러
            optimizer, scheduler = _opt_and_sch(model.parameters(), cfg, len(train_ld), logger)

            best_f1 = -1.0                                                          # 최고 F1 초기값
            best_path = os.path.join(ckpt_dir, f"best_fold{valid_fold}.pth")        # 체크포인트 경로

            # ---------------------- 에폭 루프 ---------------------- #
            for epoch in range(1, cfg["train"]["epochs"]+1):
                logger.write(f"[EPOCH {epoch}] ---------- start ----------")        # 에폭 시작 로그
                t0 = time.time()                                                    # 시작 시간 기록
                
                # 한 에폭 학습 실행
                tr_loss, tr_mem = train_one_epoch(
                    model, train_ld, criterion, optimizer, scaler,                  # 모델, 데이터로더, 손실함수, 옵티마이저, 스케일러
                    device, logger, epoch,                                          # 디바이스, 로거, 에폭
                    cfg["train"]["grad_clip_norm"], cfg["train"]["log_interval"]    # 그래디언트 클리핑, 로그 간격
                )
                
                # 검증 실행
                val_loss, val_f1, *_ = validate(
                    model, valid_ld, criterion, device, logger, epoch               # 모델, 데이터로더, 손실함수, 디바이스, 로거, 에폭
                )
                
                # 스케줄러 업데이트
                if scheduler: 
                    scheduler.step()

                # 메트릭 딕셔너리 생성
                rec = {
                    "fold": valid_fold,                         # 폴드 번호
                    "epoch": epoch,                             # 에폭 번호
                    "train_loss": tr_loss,                      # 학습 손실
                    "valid_loss": val_loss,                     # 검증 손실
                    "macro_f1": float(val_f1),                  # 매크로 F1 점수
                    "lr": optimizer.param_groups[0]["lr"],      # 학습률
                    "time_s": time.time() - t0,                 # 경과 시간
                    "mem_MiB": tr_mem,                          # GPU 메모리 사용량
                }
                jsonl_append(metrics_path, rec)                 # 메트릭 저장
                logger.write(f"[EPOCH {epoch}] metrics={rec}")  # 메트릭 로그

                # 최고 F1 갱신 시 최고 성능 모델 체크포인트 저장
                if val_f1 > best_f1:
                    # 최고 성능 모델 업데이트
                    best_f1 = float(val_f1)
                    # 체크포인트 저장
                    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "fold": valid_fold}, best_path)
                    # 최고 성능 모델 저장 로그
                    logger.write(f"[EPOCH {epoch}] NEW_BEST F1={best_f1:.5f} -> {best_path}")
                    
                # 에폭 종료 로그
                logger.write(f"[EPOCH {epoch}] ----------- end -----------")

            # 단일 폴드 학습 종료 로그
            logger.write(f"[DONE] single-fold training finished | best_f1={best_f1:.5f}")

        # ---------------------- 전체 폴드 학습 ---------------------- #
        elif isinstance(valid_fold, str) and valid_fold.lower() == "all":
            folds = cfg["data"]["folds"]                            # 전체 폴드 수
            oof_logits, oof_targets, per_fold = [], [], []          # fold별 저장 변수

            # fold 루프
            for fold in range(folds):
                trn = df[df["fold"]!=fold].reset_index(drop=True)   # 학습 데이터
                val = df[df["fold"]==fold].reset_index(drop=True)   # 검증 데이터
                # 폴드 시작 로그
                logger.write(f"[FOLD {fold}] >>> start | train={len(trn)} valid={len(val)}")

                # DataLoader 빌드
                train_ld, valid_ld = _build_loaders(cfg, trn, val, image_dir, logger)
                # 모델 빌드
                model = _build_model(cfg, device, logger)
                # 손실 함수
                criterion = nn.CrossEntropyLoss()
                # AMP 스케일러
                scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))
                # 옵티마이저/스케줄러
                optimizer, scheduler = _opt_and_sch(model.parameters(), cfg, len(train_ld), logger)

                best_f1 = -1.0                                              # 최고 f1 초기값
                best_path = os.path.join(ckpt_dir, f"best_fold{fold}.pth")  # fold별 체크포인트 경로
                v_best_logits, v_best_targets = None, None                  # 최고 logits/targets 저장 변수

                # 에폭 반복
                for epoch in range(1, cfg["train"]["epochs"] + 1):
                    # 에폭 시작 로그
                    logger.write(f"[FOLD {fold}][EPOCH {epoch}] ---------- start ----------")
                    # 시작 시간
                    t0 = time.time()
                    # 학습
                    tr_loss, tr_mem = train_one_epoch(
                        model, train_ld, criterion, optimizer, scaler,                  # 모델, 학습 데이터로더, 손실 함수, 옵티마이저, 스케일러
                        device, logger, epoch,                                          # 디바이스, 로거, 에폭
                        cfg["train"]["grad_clip_norm"], cfg["train"]["log_interval"]    # 그래디언트 클리핑, 로그 간격
                    )
                    
                    # 모델 검증 실행
                    # - model       : 현재 학습 중인 신경망 모델
                    # - valid_ld    : 검증 데이터셋 DataLoader
                    # - criterion   : 손실 함수 (예: CrossEntropyLoss)
                    # - device      : 실행 디바이스 (CPU 또는 GPU)
                    # - logger      : 학습 로그 기록 객체
                    # - epoch       : 현재 에폭 번호 (로그 출력용)
                    #
                    # 반환값:
                    # - val_loss    : 검증 데이터셋 전체 평균 손실
                    # - val_f1      : 검증 데이터셋 매크로 F1 점수
                    # - v_logits    : 모델 출력 로짓(logits) 전체 (torch.Tensor)
                    # - v_targets   : 검증 데이터셋의 실제 라벨 전체 (torch.Tensor)
                    val_loss, val_f1, v_logits, v_targets = validate(
                        model, valid_ld, criterion, device, logger, epoch
                    )
                    
                    # 스케줄러 업데이트
                    if scheduler:
                        scheduler.step()

                    # 메트릭 기록
                    rec = {
                        "fold": fold,                           # 폴드 번호
                        "epoch": epoch,                         # 에폭 번호
                        "train_loss": tr_loss,                  # 학습 손실
                        "valid_loss": val_loss,                 # 검증 손실
                        "macro_f1": float(val_f1),              # 매크로 F1 점수
                        "lr": optimizer.param_groups[0]["lr"],  # 학습률
                        "time_s": time.time() - t0,             # 경과 시간
                        "mem_MiB": tr_mem,                      # GPU 메모리 사용량
                    }
                    # jsonl에 메트릭 추가
                    jsonl_append(metrics_path, rec)
                    # 로그 출력
                    logger.write(f"[FOLD {fold}][EPOCH {epoch}] metrics={rec}")

                                        # 최고 F1 갱신 시 체크포인트 저장
                    if val_f1 > best_f1:
                        # 최고 성능 갱신 → best_f1 업데이트
                        best_f1 = float(val_f1)
                        # 현재 fold에서의 최고 logits/targets 저장
                        v_best_logits, v_best_targets = v_logits.clone(), v_targets.clone()
                        # 모델 파라미터 + 설정 + 에폭/폴드 정보 저장
                        torch.save(
                            {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "fold": fold},
                            best_path
                        )
                        # 새로운 최고 F1 스코어 로그 출력
                        logger.write(f"[FOLD {fold}][EPOCH {epoch}] NEW_BEST F1={best_f1:.5f} -> {best_path}")
                    # 에폭 종료 로그 출력
                    logger.write(f"[FOLD {fold}][EPOCH {epoch}] ----------- end -----------")

                # fold별 최고 F1 기록 누적
                per_fold.append(best_f1)
                # fold별 최고 logits 저장
                if v_best_logits is not None:
                    oof_logits.append(v_best_logits)
                # fold별 최고 targets 저장
                if v_best_targets is not None:
                    oof_targets.append(v_best_targets)
                # fold 학습 종료 로그 출력
                logger.write(f"[FOLD {fold}] <<< end | best_f1={best_f1:.5f}")

            # OOF(Out-Of-Fold) 결과 계산 시작
            import torch as _t
            # fold별 logits 합치기 (있을 경우)
            oof_logits_cat = _t.cat(oof_logits, 0) if len(oof_logits) else None
            # fold별 targets 합치기 (있을 경우)
            oof_targets_cat = _t.cat(oof_targets, 0) if len(oof_targets) else None

            # OOF macro F1 계산 (두 텐서 모두 존재 시)
            if oof_logits_cat is not None and oof_targets_cat is not None:
                # macro F1 점수 계산
                oof_macro = macro_f1_from_logits(oof_logits_cat, oof_targets_cat)
                # 메트릭 파일에 기록
                jsonl_append(metrics_path, {"fold": "all", "epoch": -1, "oof_macro_f1": float(oof_macro)})
                # fold별 점수와 함께 로그 출력
                logger.write(f"[OOF] macro-F1={oof_macro:.5f}; per-fold={['%.5f'%s for s in per_fold]}")
                try:
                    # OOF 결과 배열 저장
                    oof_dir = ensure_dir(os.path.join(exp_root, "oof"))
                    # logits 저장
                    np.save(os.path.join(oof_dir, "oof_logits.npy"), oof_logits_cat.numpy())
                    # targets 저장
                    np.save(os.path.join(oof_dir, "oof_targets.npy"), oof_targets_cat.numpy())
                    # 저장 성공 로그 출력
                    logger.write(f"[OOF] saved arrays -> {oof_dir}")
                # 저장 실패 시 예외 처리
                except Exception as e:
                    logger.write(f"[OOF][WARN] save failed: {e}")
            # 전체 폴드 학습 종료 로그 출력
            logger.write(f"[DONE] all-fold training finished")


        # ---------------------- 잘못된 valid_fold 값 ---------------------- #
        else:
            raise ValueError("data.valid_fold 는 정수 또는 'all' 이어야 합니다.")

        # 학습 파이프라인 정상 종료 로그
        logger.write("[BOOT] training pipeline finished successfully")

    # ---------------------- 예외 처리 ---------------------- #
    except Exception as e:
        exit_status = "ERROR"                                           # 종료 상태 ERROR
        exit_code = 1                                                   # 종료 코드 1
        logger.write(f"[ERROR] {type(e).__name__}: {e}", print_error=True)  # 에러 로그 출력
        raise                                                           # 예외 다시 발생

    # ---------------------- 종료 처리 ---------------------- #
    finally:
        logger.write(f"[EXIT] TRAINING {exit_status} code={exit_code}") # 종료 상태 로그 기록
        logger.write(">> Stopping logger and restoring stdio")          # 로거 종료 로그
        logger.stop_redirect()                                          # 리다이렉트 해제
        logger.close()                                                  # 로거 닫기