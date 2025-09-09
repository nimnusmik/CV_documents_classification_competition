# src/training/train_highperf.py
"""
고성능 학습 파이프라인
- Mixup 지원: Mixup 데이터 증강 지원
- Hard Augmentation: 강력한 데이터 증강
- WandB 로깅: WandB 실험 추적
- Swin Transformer & ConvNext 지원: 최신 모델 아키텍처 지원
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import os, time, numpy as np, torch, torch.nn as nn, pandas as pd, psutil  # 기본 라이브러리들
import shutil                                                       # 파일/폴더 복사 유틸
# os       : 파일/디렉터리 경로, 시스템 유틸
# time     : 시간 측정, 로깅
# numpy    : 수치 계산, 배열 연산
# torch    : PyTorch 메인 모듈
# torch.nn : 신경망 계층/손실 함수 모듈
# pandas   : 데이터프레임 처리
# psutil   : 시스템 메모리 사용량 추적
# shutil   : 파일/폴더 복사, 이동

# ------------------------- PyTorch 유틸 ------------------------- #
from torch.utils.data import DataLoader                             # 데이터 로더 클래스
from sklearn.model_selection import StratifiedKFold                 # 계층적 K-폴드 분할
from torch.cuda.amp import autocast, GradScaler                     # AMP (자동 혼합 정밀도) 지원
from torch.optim import Adam, AdamW                                 # 옵티마이저 (Adam, AdamW)
from torch.optim.lr_scheduler import CosineAnnealingLR              # 코사인 감쇠 스케줄러
from tqdm import tqdm                                               # 진행률 표시바

# ------------------------- 프로젝트 유틸 ------------------------- #
from src.utils.config import set_seed                               # 랜덤 시드 고정
from src.logging.logger import Logger                               # 기본 로거 클래스
from src.logging.wandb_logger import WandbLogger, create_wandb_config # WandB 로거 및 설정 생성
from src.utils.core.common import (                                             # 핵심 유틸리티 모듈
    load_yaml, ensure_dir, dump_yaml, short_uid, resolve_path, require_file, require_dir, create_log_path
)

# ------------------------- 시각화 및 출력 관리 ------------------------- #
from src.utils.visualizations import visualize_training_pipeline, create_organized_output_structure
from src.utils.visualizations import ExperimentOutputManager

# ------------------------- 데이터/모델 관련 ------------------------- #
from src.data.dataset import HighPerfDocClsDataset, mixup_data      # 고성능 데이터셋/믹스업 함수
from src.models.build import build_model, get_recommended_model, build_model_for_fold, is_multi_model_config, get_model_for_fold     # 모델 빌드/추천 함수
from src.metrics.f1 import macro_f1_from_logits                     # 매크로 F1 스코어 계산


# ---------------------- Mixup 학습 함수 ---------------------- #
# Mixup 손실 함수 정의
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # 가중 평균 손실 반환
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Mixup 학습 함수 정의
def train_one_epoch_mixup(model, loader, criterion, optimizer, scaler, device,  # 모델, 데이터로더, 손실함수, 옵티마이저, 스케일러, 디바이스
                         logger, wandb_logger, epoch, max_grad_norm=None,       # 로거/에폭/그래디언트 클리핑 파라미터
                         mixup_alpha=1.0, use_mixup=True):                      # Mixup 알파값과 사용 여부
    
    model.train()           # 모델을 학습 모드로 설정
    running_loss = 0.0      # 누적 손실 초기화
    total_samples = 0       # 총 샘플 수 초기화

    # 데이터셋 epoch 업데이트 (Hard Augmentation 강도 조절)
    # 데이터셋이 에폭 업데이트 메서드를 가지고 있는지 확인
    if hasattr(loader.dataset, 'update_epoch'):
        # 에폭에 따른 증강 강도 업데이트
        loader.dataset.update_epoch(epoch)
    
    # 학습 시작 로그
    logger.write(f"[EPOCH {epoch}] >>> TRAIN start | steps={len(loader)} mixup={use_mixup}")
    
    # 배치별 학습 시작
    for step, (imgs, labels) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}"), 1):
        imgs, labels = imgs.to(device), labels.to(device)   # 데이터를 GPU로 이동
        optimizer.zero_grad(set_to_none=True)               # 그래디언트 초기화 (메모리 효율)
        
        # --------------------- Mixup 적용 여부 --------------------- #
        # Mixup 적용
        if use_mixup and np.random.random() > 0.5:          # 50% 확률로 Mixup 적용
            # Mixup 데이터 생성
            mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, mixup_alpha)
            
            # AMP 자동 캐스팅 적용
            with autocast(enabled=scaler is not None):
                # 믹스된 이미지로 순전파
                logits = model(mixed_imgs)
                # Mixup 손실 계산
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                
        # Mixup 미적용 경우
        else:
            with autocast(enabled=scaler is not None):      # AMP 자동 캐스팅 적용
                logits = model(imgs)                        # 원본 이미지로 순전파
                loss = criterion(logits, labels)            # 일반 손실 계산
        
        # --------------------- 역전파 및 옵티마이저 스텝 -------------------- #
        # AMP 스케일러 사용 시
        if scaler:                                      
            scaler.scale(loss).backward()                   # 스케일된 손실로 역전파
            if max_grad_norm:                               # 그래디언트 클리핑 적용 여부
                scaler.unscale_(optimizer)                  # 그래디언트 언스케일링
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 그래디언트 클리핑
            scaler.step(optimizer)                          # 스케일된 옵티마이저 스텝
            scaler.update()                                 # 스케일러 업데이트
            
        # 일반 옵티마이저 사용 시
        else:
            loss.backward()                                 # 일반 역전파
            if max_grad_norm:                               # 그래디언트 클리핑 적용 여부
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 그래디언트 클리핑
            optimizer.step()                                # 옵티마이저 스텝

        # 메트릭 누적
        running_loss += loss.item() * imgs.size(0)          # 배치 크기로 가중된 손실 누적
        total_samples += imgs.size(0)                       # 총 샘플 수 누적
        
        # -------------------- WandB 로깅 -------------------- #
        if wandb_logger and step % 10 == 0:                 # 10 스텝마다 WandB 로깅
            wandb_logger.log_metrics({                      # 메트릭 딕셔너리 로깅
                "train/batch_loss": loss.item(),            # 배치 손실값
                "train/learning_rate": optimizer.param_groups[0]["lr"],  # 현재 학습률
                "train/epoch": epoch,                       # 현재 에폭 번호
                "train/step": step                          # 현재 스텝 번호
            })                                              # 메트릭 딕셔너리 종료
        
        # -------------------- 로그 출력 ---------------------- #
        # 50스텝마다 또는 첫/마지막 스텝
        if step % 50 == 0 or step == 1 or step == len(loader):
            lr = optimizer.param_groups[0]["lr"]            # 현재 학습률 추출
            
            # 로그 메시지 작성
            logger.write(
                f"[EPOCH {epoch}][TRAIN step {step}/{len(loader)}] "        # 에폭/스텝 정보
                f"loss={loss.item():.5f} lr={lr:.6f} bs={imgs.size(0)}"     # 손실/학습률/배치크기
            )
    
    # 에폭 평균 손실 계산
    epoch_loss = running_loss / total_samples
    
    # 학습 종료 로그
    logger.write(f"[EPOCH {epoch}] <<< TRAIN end | loss={epoch_loss:.5f}")
    
    # 에폭 손실 반환
    return epoch_loss


# ---------------------- 고성능 검증 함수 ---------------------- #
@torch.no_grad()            # 그래디언트 계산 비활성화 데코레이터
# 고성능 검증 함수 정의
def validate_highperf(model, loader, criterion, device, logger, wandb_logger, epoch=None):
    phase = f"EPOCH {epoch}" if epoch is not None else "EVAL"           # 에폭 정보 또는 EVAL 설정
    logger.write(f"[{phase}] >>> VALID start | steps={len(loader)}")    # 검증 시작 로그
    
    model.eval()            # 모델을 평가 모드로 설정
    running_loss = 0.0      # 누적 손실 초기화
    total_samples = 0       # 총 샘플 수 초기화
    all_logits = []         # 모든 로짓 저장 리스트
    all_targets = []        # 모든 타겟 저장 리스트

    # ---------------------- 배치별 검증 시작 --------------------- #
    for step, (imgs, labels) in enumerate(tqdm(loader, desc=f"Valid Epoch {epoch}"), 1):
        imgs, labels = imgs.to(device), labels.to(device)   # 데이터를 GPU로 이동
        
        logits = model(imgs)                                # 모델 순전파
        loss = criterion(logits, labels)                    # 손실 계산
        
        running_loss += loss.item() * imgs.size(0)          # 배치 크기로 가중된 손실 누적
        total_samples += imgs.size(0)                       # 총 샘플 수 누적
        all_logits.append(logits.cpu())                     # 로짓을 CPU로 이동하여 저장
        all_targets.append(labels.cpu())                    # 라벨을 CPU로 이동하여 저장

    # ---------------------- 에폭 메트릭 계산 --------------------- #
    logits = torch.cat(all_logits, dim=0)               # 모든 로짓 연결
    targets = torch.cat(all_targets, dim=0)             # 모든 타겟 연결
    f1 = macro_f1_from_logits(logits, targets)          # 매크로 F1 스코어 계산
    epoch_loss = running_loss / total_samples           # 에폭 평균 손실 계산

    # ------------------------ WandB 로깅 ----------------------- #
    # WandB 로거가 있는 경우
    if wandb_logger:
        wandb_logger.log_metrics({                      # 메트릭 딕셔너리 로깅
            "val/loss": epoch_loss,                     # 검증 손실값
            "val/f1": f1,                               # 검증 F1 스코어
            "val/epoch": epoch                          # 에폭 번호
        })

    # 검증 종료 로그
    logger.write(f"[{phase}] <<< VALID end | loss={epoch_loss:.5f} macro_f1={f1:.5f}")
    
    # 손실, F1, 로짓, 타겟 반환
    return epoch_loss, f1, logits, targets


# ---------------------- 고성능 데이터로더 빌드 함수 ---------------------- #
# 고성능 데이터로더 빌드 함수 정의
def build_highperf_loaders(cfg, trn_df, val_df, image_dir, logger, epoch=0):
    img_size = cfg["train"]["img_size"]             # 설정에서 이미지 크기 추출
    batch_size = cfg["train"]["batch_size"]         # 설정에서 배치 크기 추출
    total_epochs = cfg["train"]["epochs"]           # 설정에서 총 에폭 수 추출

    # 데이터로더 빌드 로그
    logger.write(f"[DATA] build highperf loaders | img_size={img_size} bs={batch_size}")
    logger.write(f"[DATA] augmentation: baseline-advanced (normal + hard augmentation)")
    
    # 고성능 데이터셋 생성
    train_ds = HighPerfDocClsDataset(
        trn_df,                                     # 학습 데이터프레임
        image_dir,                                  # 이미지 디렉터리 경로
        img_size=img_size,                          # 이미지 크기
        epoch=epoch,                                # 현재 에폭
        total_epochs=total_epochs,                  # 총 에폭 수
        is_train=True,                              # 학습 모드 플래그
        id_col=cfg["data"]["id_col"],               # ID 컬럼명
        target_col=cfg["data"]["target_col"],       # 타겟 컬럼명
        logger=logger                               # 로거 객체
    )
    
    # 검증용 고성능 데이터셋 생성
    valid_ds = HighPerfDocClsDataset(
        val_df,                                     # 검증 데이터프레임
        image_dir,                                  # 이미지 디렉터리 경로
        img_size=img_size,                          # 이미지 크기
        epoch=epoch,                                # 현재 에폭
        total_epochs=total_epochs,                  # 총 에폭 수
        is_train=False,                             # 평가 모드 플래그
        id_col=cfg["data"]["id_col"],               # ID 컬럼명
        target_col=cfg["data"]["target_col"],       # 타겟 컬럼명
        logger=logger                               # 로거 객체
    )
    
    # 데이터셋 크기 로그
    logger.write(f"[DATA] dataset sizes | train={len(train_ds)} valid={len(valid_ds)}")
    
    # 학습용 데이터로더 생성
    train_ld = DataLoader(
        train_ds,                                   # 학습 데이터셋
        batch_size=batch_size,                      # 배치 크기
        shuffle=True,                               # 데이터 셔플 활성화
        num_workers=cfg["project"]["num_workers"],  # 워커 프로세스 수
        pin_memory=True,                            # 메모리 고정 활성화
        drop_last=False                             # 마지막 배치 유지
    )
    
    # 검증용 데이터로더 생성
    valid_ld = DataLoader(
        valid_ds,                                   # 검증 데이터셋
        batch_size=batch_size,                      # 배치 크기
        shuffle=False,                              # 셔플 비활성화 (검증용)
        num_workers=cfg["project"]["num_workers"],  # 워커 프로세스 수
        pin_memory=True,                            # 메모리 고정 활성화
        drop_last=False                             # 마지막 배치 유지
    )
    
    # 학습/검증 데이터로더 반환
    return train_ld, valid_ld


# ---------------------- 고성능 학습 파이프라인 실행 함수 ---------------------- #
# 고성능 학습 파이프라인 실행 함수 정의
def run_highperf_training(cfg_path: str):
    #--------------------------- 설정 및 로거 초기화 ------------------------- #
    cfg = load_yaml(cfg_path)                               # YAML 설정 파일 로드
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))    # 설정 파일 디렉터리 경로
    
    #------------------------ 랜덤 시드 및 실행 ID 설정 ---------------------- #
    set_seed(cfg["project"]["seed"])                        # 랜덤 시드 고정
    run_id = f'{cfg["project"]["run_name"]}'  # 실행 ID 생성
    
    #------------------------- 실험 디렉터리 및 로거 설정 ---------------------- #
    day = time.strftime(cfg["project"]["date_format"])      # 현재 날짜 문자열
    time_str = time.strftime(cfg["project"]["time_format"]) # 현재 시간 문자열
    # 타임스탬프 포함된 폴더명 생성 (예: swin-highperf_20250907_1530)
    folder_name = f"{day}_{time_str}_{cfg['project']['run_name']}"
    exp_root = ensure_dir(os.path.join(cfg["output"]["exp_dir"], day, folder_name))  # 실험 루트 디렉터리
    ckpt_dir = ensure_dir(os.path.join(exp_root, "ckpt"))   # 체크포인트 디렉터리
    
    # lastest-train 폴더에도 동일한 구조로 생성
    lastest_exp_root = ensure_dir(os.path.join(cfg["output"]["exp_dir"], "lastest-train", folder_name))
    lastest_ckpt_dir = ensure_dir(os.path.join(lastest_exp_root, "ckpt"))
    
    #------------------------- 설정 파일 백업 ------------------------- #
    # 증강 타입에 따른 로그 파일명 생성
    aug_type = "advanced_augmentation" if cfg["train"].get("use_advanced_augmentation", False) else "basic_augmentation"
    log_filename = f'{cfg["project"]["log_prefix"]}_{day}-{time.strftime("%H%M")}_{run_id}_{aug_type}.log'  # 로그 파일명 생성
    log_path = create_log_path("train", log_filename)       # 날짜별 로그 파일 전체 경로
    
    # 로거 객체 생성
    logger = Logger(
        log_path=log_path,                                  # 로그 파일 경로
        print_also=cfg["project"]["verbose"]                # 콘솔 출력 여부
    )
    
    # 파이프라인 시작 로그
    logger.write("[BOOT] high-performance training pipeline started")
    
    try:
        #--------------------------- 디바이스 및 경로 설정 ------------------------- #
        # 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                   # GPU/CPU 디바이스 설정
        logger.write(f"[BOOT] device={device}")                                                 # 디바이스 정보 로그

        # 경로 확인
        train_csv = resolve_path(cfg_dir, cfg["data"]["train_csv"])                             # 학습 CSV 경로 해결
        image_dir = resolve_path(cfg_dir, cfg["data"].get("image_dir_train", "data/raw/train")) # 이미지 디렉터리 경로 해결
        require_file(train_csv, "train_csv 확인")                                                # 학습 CSV 파일 존재성 검증
        require_dir(image_dir, "image_dir 확인")                                                 # 이미지 디렉터리 존재성 검증

        # 데이터 로드
        df = pd.read_csv(train_csv)                                                             # 학습 데이터 CSV 로드
        logger.write(f"[DATA] loaded train data | shape={df.shape}")                            # 데이터 로드 로그

        #--------------------------- 폴드 분할 --------------------------- #
        folds = cfg["data"]["folds"]    # 폴드 수 설정
        
        # 폴드 컬럼이 없는 경우에만 생성
        if "fold" not in df.columns:
            # 계층적 K-폴드 객체 생성
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["project"]["seed"])
            
            # 폴드 컬럼 초기화
            df["fold"] = -1
            
            # 폴드별 분할
            for f, (_, v_idx) in enumerate(skf.split(df, df[cfg["data"]["target_col"]])):
                # 검증 인덱스에 폴드 번호 할당
                df.loc[df.index[v_idx], "fold"] = f
        
        #--------------------------- WandB 설정 --------------------------- #
        # 다중 모델 여부에 따른 모델명 결정
        if is_multi_model_config(cfg):
            # 다중 모델인 경우 첫 번째 폴드의 모델명 사용 (WandB 표시용)
            wandb_model_name, _ = get_model_for_fold(cfg, 0)
        else:
            # 단일 모델인 경우 기존 방식
            wandb_model_name = cfg["model"]["name"]
            
        wandb_config = create_wandb_config(                     # WandB 설정 생성
            model_name=wandb_model_name,                        # 모델명
            img_size=cfg["train"]["img_size"],                  # 이미지 크기
            batch_size=cfg["train"]["batch_size"],              # 배치 크기
            learning_rate=cfg["train"]["lr"],                   # 학습률
            epochs=cfg["train"]["epochs"],                      # 에폭 수
            mixup_alpha=cfg["train"].get("mixup_alpha", 1.0),  # Mixup 알파값
            hard_augmentation=True,                             # Hard Augmentation 플래그
            optimizer=cfg["train"]["optimizer"],                # 옵티마이저 타입
            scheduler=cfg["train"]["scheduler"]                 # 스케줄러 타입
        )
        
        #--------------------------- 폴드별 학습 --------------------------- #
        # 폴드 결과 저장 리스트
        fold_results = []
        
        # 각 폴드별 반복
        for fold in range(folds):
            #--------------------------- 폴드별 로거 설정 -------------------------- #
            logger.write(f"\n{'='*50}")                             # 폴드 구분선
            logger.write(f"FOLD {fold+1}/{folds} START")            # 폴드 시작 로그
            logger.write(f"{'='*50}")                               # 폴드 구분선
            
            # 학습/검증 데이터 분할
            trn_df = df[df["fold"] != fold].reset_index(drop=True)  # 학습 데이터프레임
            val_df = df[df["fold"] == fold].reset_index(drop=True)  # 검증 데이터프레임
            
            # 폴드 데이터 크기 로그
            logger.write(f"[FOLD {fold}] train={len(trn_df)} valid={len(val_df)}")
            
            #-------------------------- WandB 로거 초기화 -------------------------- #
            # 동적 실행 이름 생성 (submissions와 동일한 형식)
            current_date = pd.Timestamp.now().strftime('%Y%m%d')
            current_time = pd.Timestamp.now().strftime('%H%M')
            # 다중 모델인 경우 이미 정의된 wandb_model_name 사용
            if is_multi_model_config(cfg):
                model_name_for_exp = wandb_model_name  # 이미 WandB 설정 시 정의된 변수 재사용
            else:
                model_name_for_exp = cfg["model"]["name"]
            dynamic_experiment_name = f"{current_date}_{current_time}_{model_name_for_exp}_ensemble_tta"
            
            # WandB 초기화
            wandb_logger = WandbLogger(
                experiment_name=dynamic_experiment_name,        # 동적 실험명
                config=wandb_config,                            # 설정 딕셔너리
                tags=["high-performance", "mixup", "hard-aug"]  # 태그 리스트
            )
            
            # WandB 실행 초기화
            wandb_logger.init_run(fold=fold)
            
            #-------------------------- 모델 생성 -------------------------- #
            # 다중 모델 설정 확인 및 모델 생성
            if is_multi_model_config(cfg):
                # 다중 모델: 폴드별 다른 모델 사용
                model = build_model_for_fold(cfg, fold, cfg["data"]["num_classes"]).to(device)
                model_name, _ = get_model_for_fold(cfg, fold)
                logger.write(f"[MULTI-MODEL] Fold {fold} using model: {model_name}")
            else:
                # 단일 모델: 모든 폴드에 같은 모델 사용
                model_name = get_recommended_model(cfg["model"]["name"])
                model = build_model(
                    model_name,                         # 모델명
                    cfg["data"]["num_classes"],         # 클래스 수
                    cfg["model"]["pretrained"],         # 사전훈련 여부
                    cfg["model"]["drop_rate"],          # 드롭아웃 비율
                    cfg["model"]["drop_path_rate"],     # 드롭패스 비율
                    cfg["model"]["pooling"]             # 풀링 타입
                ).to(device)                            # GPU로 모델 이동
                logger.write(f"[SINGLE-MODEL] All folds using model: {model_name}")
            
            #-------------------------- 옵티마이저 및 스케줄러 설정 ------------------------- #
            # AdamW 옵티마이저 사용 시
            if cfg["train"]["optimizer"] == "adamw":
                # AdamW 옵티마이저 생성
                optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
            # 기본 Adam 옵티마이저 사용 시
            else:
                # Adam 옵티마이저 생성
                optimizer = Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
            
            # 코사인 감쇠 스케줄러 생성
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
            
            #-------------------------- 손실 함수 및 스케일러 설정 ------------------------- #
            # 교차 엔트로피 손실 함수
            criterion = nn.CrossEntropyLoss()
            # AMP 스케일러 (조건부)
            scaler = GradScaler() if cfg["train"].get("mixed_precision", True) else None
            
            #-------------------------- 학습 준비 완료 -------------------------- #
            # 최고 F1 점수 추적
            best_f1 = 0.0   # 최고 F1 점수 초기화
            
            # 최고 모델 저장 경로
            best_model_path = os.path.join(ckpt_dir, f"best_model_fold_{fold+1}.pth")
            lastest_best_model_path = os.path.join(lastest_ckpt_dir, f"best_model_fold_{fold+1}.pth")
            
            #-------------------------- 폴드별 학습 -------------------------- #
            # 에포크별 학습
            for epoch in range(1, cfg["train"]["epochs"] + 1):
                # 데이터로더 생성 (에포크별 Hard Aug 강도 조절)  # 데이터로더 생성 주석
                train_ld, valid_ld = build_highperf_loaders(cfg, trn_df, val_df, image_dir, logger, epoch-1)
                
                # 학습
                train_loss = train_one_epoch_mixup(                         # Mixup 학습 함수 호출
                    model, train_ld, criterion, optimizer, scaler, device,  # 모델/데이터/손실/옵티마이저/스케일러/디바이스
                    logger, wandb_logger, epoch,                            # 로거/WandB로거/에폭
                    max_grad_norm=cfg["train"].get("max_grad_norm"),        # 그래디언트 클리핑 노름
                    mixup_alpha=cfg["train"].get("mixup_alpha", 1.0),       # Mixup 알파값
                    use_mixup=cfg["train"].get("use_mixup", True)           # Mixup 사용 여부
                )

                # 검증
                val_loss, val_f1, _, _ = validate_highperf(                 # 고성능 검증 함수 호출
                    model, valid_ld, criterion, device, logger, wandb_logger, epoch  # 모델/데이터/손실/디바이스/로거/에폭
                ) 
                
                # 스케줄러가 있는 경우 스케줄러 업데이트
                if scheduler:
                    scheduler.step()    # 스케줄러 스텝 실행
                
                # 현재 F1이 최고 기록보다 높은 경우 최고 모델 저장
                if val_f1 > best_f1:
                    best_f1 = val_f1                                    # 최고 F1 점수 업데이트
                    model_state = {                                     # 모델 상태 딕셔너리
                        'epoch': epoch,                                 # 에폭 번호
                        'model_state_dict': model.state_dict(),         # 모델 가중치
                        'optimizer_state_dict': optimizer.state_dict(), # 옵티마이저 상태
                        'f1': val_f1,                                   # F1 점수
                        'loss': val_loss,                               # 손실값
                    }
                    torch.save(model_state, best_model_path)            # 날짜 폴더에 모델 저장
                    torch.save(model_state, lastest_best_model_path)     # lastest 폴더에도 모델 저장
                    
                    # 새로운 최고 기록 로그
                    logger.write(f"[FOLD {fold}] NEW BEST F1: {best_f1:.5f} (epoch {epoch})")
                
                # WandB 로깅
                wandb_logger.log_metrics({          # 메트릭 딕셔너리 로깅
                    "fold": fold,                   # 폴드 번호
                    "epoch": epoch,                 # 에폭 번호
                    "train/loss": train_loss,       # 학습 손실
                    "val/loss": val_loss,           # 검증 손실
                    "val/f1": val_f1,               # 검증 F1
                    "best_f1": best_f1              # 최고 F1
                })
            
            # 폴드 결과 저장
            fold_results.append({                   # 폴드 결과 리스트에 추가
                "fold": fold,                       # 폴드 번호
                "best_f1": best_f1,                 # 최고 F1 점수
                "model_path": best_model_path       # 모델 저장 경로
            })
            
            # 폴드 완료 로그
            logger.write(f"[FOLD {fold}] COMPLETED | Best F1: {best_f1:.5f}")
            # WandB 종료
            wandb_logger.finish()
        
        #------ --------------------- 전체 폴드 완료 후 결과 요약 ---------------------- #
        avg_f1 = float(np.mean([r["best_f1"] for r in fold_results]))   # 평균 F1 계산 (numpy float를 Python float로 변환)
        logger.write(f"\n{'='*50}")                                     # 결과 구분선
        logger.write(f"ALL FOLDS COMPLETED")                            # 전체 폴드 완료 로그
        logger.write(f"Average F1: {avg_f1:.5f}")                       # 평균 F1 로그
        for r in fold_results:                                          # 각 폴드 결과 출력
            logger.write(f"Fold {r['fold']}: {r['best_f1']:.5f}")       # 폴드별 F1 로그
        logger.write(f"{'='*50}")                                       # 결과 구분선
        
        # ---------------------- 결과 저장 ---------------------- #
        results_path = os.path.join(exp_root, "fold_results.yaml")      # 결과 파일 경로
        
        # 결과를 YAML로 저장
        dump_yaml({"fold_results": fold_results, "average_f1": avg_f1}, results_path)
        
        # lastest-train 폴더에도 결과 저장
        lastest_results_path = os.path.join(lastest_exp_root, "fold_results.yaml")
        dump_yaml({"fold_results": fold_results, "average_f1": avg_f1}, lastest_results_path)
        
        # ---------------------- 시각화 생성 ---------------------- #
        try:
            # 시각화를 위한 히스토리 데이터 준비
            # WandB 로그에서 기록된 데이터를 사용하거나 기본 구조 생성
            history_data = {
                'train_loss': [],
                'val_loss': [],
                'val_f1': [],
                'epochs': list(range(1, cfg["train"]["epochs"] + 1))
            }
            
            # 시각화 생성 - 다중 모델을 고려한 모델명 사용
            if is_multi_model_config(cfg):
                # 다중 모델인 경우 "multi-model-ensemble" 이름 사용
                model_name_viz = "multi-model-ensemble"
            else:
                model_name_viz = cfg["model"]["name"]
            
            # fold_results를 딕셔너리 형태로 변환
            fold_results_dict = {
                'fold_results': fold_results,
                'average_f1': avg_f1,
                'total_folds': cfg["train"]["n_folds"]
            }
            
            visualize_training_pipeline(
                fold_results=fold_results_dict,
                model_name=model_name_viz,
                output_dir=exp_root,
                history_data=history_data
            )
            logger.write(f"[VIZ] Training visualizations created in {exp_root}")
            
        except Exception as viz_error:
            logger.write(f"[WARNING] Visualization failed: {str(viz_error)}")
        
        # ---------------------- lastest-train 폴더에 복사 ---------------------- #
        # lastest-train 폴더 경로 설정
        lastest_train_dir = os.path.join("experiments", "train", "lastest-train")
        experiment_folder_name = cfg["project"]["run_name"]     # 실험 폴더명
        lastest_train_model_path = os.path.join(lastest_train_dir, experiment_folder_name)
        
        # lastest-train 디렉터리 생성
        os.makedirs(lastest_train_dir, exist_ok=True)
        
        # 기존 모델 폴더가 있으면 삭제 (덮어쓰기를 위해)
        if os.path.exists(lastest_train_model_path):
            shutil.rmtree(lastest_train_model_path)
            logger.write(f"[CLEANUP] Removed existing lastest-train/{experiment_folder_name}")
        
        # 현재 실험 결과를 lastest-train으로 복사
        shutil.copytree(exp_root, lastest_train_model_path)
        logger.write(f"[COPY] Results copied to lastest-train/{experiment_folder_name}")
        logger.write(f"📁 Latest results: {lastest_train_model_path}")
        
        # 학습 성공 로그
        logger.write(f"[SUCCESS] Training completed | avg_f1={avg_f1:.5f}")
    
    # 예외 발생 시 처리
    except Exception as e:
        logger.write(f"[ERROR] Training failed: {str(e)}")              # 에러 로그
        raise                                                           # 예외 재발생
    # 종료 처리
    finally:
        logger.write("[SHUTDOWN] Training pipeline ended")              # 파이프라인 종료 로그


# ---------------------- 메인 실행부 ---------------------- #
if __name__ == "__main__":
    import sys      # sys 모듈 import
    
    # 커맨드라인 인자 개수 확인
    if len(sys.argv) != 2:
        print("Usage: python train_highperf.py <config_path>")  # 사용법 출력
        sys.exit(1)                                             # 프로그램 종료
    
    run_highperf_training(sys.argv[1])                          # 고성능 학습 파이프라인 실행
