# src/training/train_highperf.py
"""
고성능 학습 파이프라인
- Mixup 지원
- Hard Augmentation
- WandB 로깅
- Swin Transformer & ConvNext 지원
"""

import os, time, numpy as np, torch, torch.nn as nn, pandas as pd, psutil
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 프로젝트 유틸
from src.utils.seed import set_seed
from src.utils.logger import Logger
from src.utils.wandb_logger import WandBLogger, create_wandb_config
from src.utils.common import (
    load_yaml, ensure_dir, dump_yaml, jsonl_append, short_uid,
    resolve_path, require_file, require_dir
)

# 데이터/모델 관련
from src.data.dataset import HighPerfDocClsDataset, mixup_data
from src.models.build import build_model, get_recommended_model
from src.metrics.f1 import macro_f1_from_logits


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup용 손실 함수"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch_mixup(model, loader, criterion, optimizer, scaler, device,
                         logger, wandb_logger, epoch, max_grad_norm=None, 
                         mixup_alpha=1.0, use_mixup=True):
    """Mixup을 지원하는 학습 함수"""
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    # 데이터셋 epoch 업데이트 (Hard Augmentation 강도 조절)
    if hasattr(loader.dataset, 'update_epoch'):
        loader.dataset.update_epoch(epoch)
    
    logger.write(f"[EPOCH {epoch}] >>> TRAIN start | steps={len(loader)} mixup={use_mixup}")
    
    for step, (imgs, labels) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}"), 1):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        # Mixup 적용
        if use_mixup and np.random.random() > 0.5:
            mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, mixup_alpha)
            
            with autocast(enabled=scaler is not None):
                logits = model(mixed_imgs)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            with autocast(enabled=scaler is not None):
                logits = model(imgs)
                loss = criterion(logits, labels)
        
        # Backward
        if scaler:
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # 메트릭 누적
        running_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)
        
        # WandB 로깅
        if wandb_logger and step % 10 == 0:
            wandb_logger.log_metrics({
                "train/batch_loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
                "train/step": step
            })
        
        # 로그 출력
        if step % 50 == 0 or step == 1 or step == len(loader):
            lr = optimizer.param_groups[0]["lr"]
            logger.write(
                f"[EPOCH {epoch}][TRAIN step {step}/{len(loader)}] "
                f"loss={loss.item():.5f} lr={lr:.6f} bs={imgs.size(0)}"
            )
    
    epoch_loss = running_loss / total_samples
    logger.write(f"[EPOCH {epoch}] <<< TRAIN end | loss={epoch_loss:.5f}")
    
    return epoch_loss


@torch.no_grad()
def validate_highperf(model, loader, criterion, device, logger, wandb_logger, epoch=None):
    """고성능 검증 함수"""
    phase = f"EPOCH {epoch}" if epoch is not None else "EVAL"
    logger.write(f"[{phase}] >>> VALID start | steps={len(loader)}")
    
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_logits = []
    all_targets = []
    
    for step, (imgs, labels) in enumerate(tqdm(loader, desc=f"Valid Epoch {epoch}"), 1):
        imgs, labels = imgs.to(device), labels.to(device)
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        running_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())
    
    # 메트릭 계산
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    f1 = macro_f1_from_logits(logits, targets)
    epoch_loss = running_loss / total_samples
    
    # WandB 로깅
    if wandb_logger:
        wandb_logger.log_metrics({
            "val/loss": epoch_loss,
            "val/f1": f1,
            "val/epoch": epoch
        })
    
    logger.write(f"[{phase}] <<< VALID end | loss={epoch_loss:.5f} macro_f1={f1:.5f}")
    
    return epoch_loss, f1, logits, targets


def build_highperf_loaders(cfg, trn_df, val_df, image_dir, logger, epoch=0):
    """고성능 데이터로더 생성"""
    img_size = cfg["train"]["img_size"]
    batch_size = cfg["train"]["batch_size"]
    total_epochs = cfg["train"]["epochs"]
    
    logger.write(f"[DATA] build highperf loaders | img_size={img_size} bs={batch_size}")
    
    # 고성능 데이터셋 생성
    train_ds = HighPerfDocClsDataset(
        trn_df,
        image_dir,
        img_size=img_size,
        epoch=epoch,
        total_epochs=total_epochs,
        is_train=True,
        id_col=cfg["data"]["id_col"],
        target_col=cfg["data"]["target_col"],
        logger=logger
    )
    
    valid_ds = HighPerfDocClsDataset(
        val_df,
        image_dir,
        img_size=img_size,
        epoch=epoch,
        total_epochs=total_epochs,
        is_train=False,
        id_col=cfg["data"]["id_col"],
        target_col=cfg["data"]["target_col"],
        logger=logger
    )
    
    logger.write(f"[DATA] dataset sizes | train={len(train_ds)} valid={len(valid_ds)}")
    
    # 데이터로더 생성
    train_ld = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["project"]["num_workers"],
        pin_memory=True,
        drop_last=False
    )
    
    valid_ld = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["project"]["num_workers"],
        pin_memory=True,
        drop_last=False
    )
    
    return train_ld, valid_ld


def run_highperf_training(cfg_path: str):
    """고성능 학습 파이프라인 실행"""
    # 설정 로드
    cfg = load_yaml(cfg_path)
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    
    # 시드 및 실행 ID
    set_seed(cfg["project"]["seed"])
    run_id = f'{cfg["project"]["run_name"]}-{short_uid()}'
    
    # 로거 설정
    day = time.strftime(cfg["project"]["date_format"])
    exp_root = ensure_dir(os.path.join(cfg["output"]["exp_dir"], day, cfg["project"]["run_name"]))
    ckpt_dir = ensure_dir(os.path.join(exp_root, "ckpt"))
    
    logger = Logger(
        log_dir=ensure_dir(cfg["output"]["log_dir"]),
        log_prefix=f'{cfg["project"]["log_prefix"]}_{day}-{time.strftime("%H%M")}_{run_id}',
        also_print=cfg["project"]["verbose"]
    )
    
    logger.write("[BOOT] high-performance training pipeline started")
    
    try:
        # 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.write(f"[BOOT] device={device}")
        
        # 경로 확인
        train_csv = resolve_path(cfg_dir, cfg["data"]["train_csv"])
        image_dir = resolve_path(cfg_dir, cfg["data"].get("image_dir_train", "data/raw/train"))
        require_file(train_csv, "train_csv 확인")
        require_dir(image_dir, "image_dir 확인")
        
        # 데이터 로드
        df = pd.read_csv(train_csv)
        logger.write(f"[DATA] loaded train data | shape={df.shape}")
        
        # 폴드 분할
        folds = cfg["data"]["folds"]
        if "fold" not in df.columns:
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["project"]["seed"])
            df["fold"] = -1
            for f, (_, v_idx) in enumerate(skf.split(df, df[cfg["data"]["target_col"]])):
                df.loc[df.index[v_idx], "fold"] = f
        
        # WandB 설정
        wandb_config = create_wandb_config(
            model_name=cfg["model"]["name"],
            img_size=cfg["train"]["img_size"],
            batch_size=cfg["train"]["batch_size"],
            learning_rate=cfg["train"]["lr"],
            epochs=cfg["train"]["epochs"],
            mixup_alpha=cfg["train"].get("mixup_alpha", 1.0),
            hard_augmentation=True,
            optimizer=cfg["train"]["optimizer"],
            scheduler=cfg["train"]["scheduler"]
        )
        
        # 폴드별 학습
        fold_results = []
        
        for fold in range(folds):
            logger.write(f"\n{'='*50}")
            logger.write(f"FOLD {fold+1}/{folds} START")
            logger.write(f"{'='*50}")
            
            # 폴드 데이터 분할
            trn_df = df[df["fold"] != fold].reset_index(drop=True)
            val_df = df[df["fold"] == fold].reset_index(drop=True)
            
            logger.write(f"[FOLD {fold}] train={len(trn_df)} valid={len(val_df)}")
            
            # WandB 초기화
            wandb_logger = WandBLogger(
                experiment_name=f'{cfg["project"]["run_name"]}-{cfg["model"]["name"]}',
                config=wandb_config,
                tags=["high-performance", "mixup", "hard-aug"]
            )
            wandb_logger.init_run(fold=fold)
            
            # 모델 생성
            model_name = get_recommended_model(cfg["model"]["name"])
            model = build_model(
                model_name,
                cfg["data"]["num_classes"],
                cfg["model"]["pretrained"],
                cfg["model"]["drop_rate"],
                cfg["model"]["drop_path_rate"],
                cfg["model"]["pooling"]
            ).to(device)
            
            # 옵티마이저 및 스케줄러
            if cfg["train"]["optimizer"] == "adamw":
                optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
            else:
                optimizer = Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
            
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
            
            # 손실 함수 및 스케일러
            criterion = nn.CrossEntropyLoss()
            scaler = GradScaler() if cfg["train"].get("mixed_precision", True) else None
            
            # 최고 F1 점수 추적
            best_f1 = 0.0
            best_model_path = os.path.join(ckpt_dir, f"best_model_fold_{fold+1}.pth")
            
            # 에포크별 학습
            for epoch in range(1, cfg["train"]["epochs"] + 1):
                # 데이터로더 생성 (에포크별 Hard Aug 강도 조절)
                train_ld, valid_ld = build_highperf_loaders(cfg, trn_df, val_df, image_dir, logger, epoch-1)
                
                # 학습
                train_loss = train_one_epoch_mixup(
                    model, train_ld, criterion, optimizer, scaler, device,
                    logger, wandb_logger, epoch,
                    max_grad_norm=cfg["train"].get("max_grad_norm"),
                    mixup_alpha=cfg["train"].get("mixup_alpha", 1.0),
                    use_mixup=cfg["train"].get("use_mixup", True)
                )
                
                # 검증
                val_loss, val_f1, _, _ = validate_highperf(
                    model, valid_ld, criterion, device, logger, wandb_logger, epoch
                )
                
                # 스케줄러 업데이트
                if scheduler:
                    scheduler.step()
                
                # 최고 모델 저장
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'f1': val_f1,
                        'loss': val_loss,
                    }, best_model_path)
                    logger.write(f"[FOLD {fold}] NEW BEST F1: {best_f1:.5f} (epoch {epoch})")
                
                # WandB 로깅
                wandb_logger.log_metrics({
                    "fold": fold,
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/f1": val_f1,
                    "best_f1": best_f1
                })
            
            # 폴드 결과 저장
            fold_results.append({
                "fold": fold,
                "best_f1": best_f1,
                "model_path": best_model_path
            })
            
            logger.write(f"[FOLD {fold}] COMPLETED | Best F1: {best_f1:.5f}")
            wandb_logger.finish()
        
        # 전체 결과 요약
        avg_f1 = np.mean([r["best_f1"] for r in fold_results])
        logger.write(f"\n{'='*50}")
        logger.write(f"ALL FOLDS COMPLETED")
        logger.write(f"Average F1: {avg_f1:.5f}")
        for r in fold_results:
            logger.write(f"Fold {r['fold']}: {r['best_f1']:.5f}")
        logger.write(f"{'='*50}")
        
        # 결과 저장
        results_path = os.path.join(exp_root, "fold_results.yaml")
        dump_yaml({"fold_results": fold_results, "average_f1": avg_f1}, results_path)
        
        logger.write(f"[SUCCESS] Training completed | avg_f1={avg_f1:.5f}")
        
    except Exception as e:
        logger.write(f"[ERROR] Training failed: {str(e)}")
        raise
    finally:
        logger.write("[SHUTDOWN] Training pipeline ended")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_highperf.py <config_path>")
        sys.exit(1)
    
    run_highperf_training(sys.argv[1])
