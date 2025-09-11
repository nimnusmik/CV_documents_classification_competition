#!/usr/bin/env python
# coding: utf-8

# # 2단계 캐스케이드 분류 시스템 구현
# 
# ## 목표
# - 분류기 A: 17개 클래스 전체 분류 (이미 완료)
# - 분류기 B: 취약 클래스 3,4,7,14만 분류 (새로 학습)
# - 캐스케이드: A가 취약 클래스로 예측하면 B로 재분류
# 
# ## 취약 클래스 분석
# - Class 3: 61.0% 정확도
# - Class 4: (분석 필요)
# - Class 7: 60.0% 정확도  
# - Class 14: 50.0% 정확도
# 

# In[11]:


# 필요한 라이브러리 import
import os
import time
import random
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# In[12]:


import torch, gc

def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# In[13]:
# 1) 변수로 경로 저장
base_dir = "/root/computervisioncompetition-cv-1/mywork/experiments"

# 2) 작업 디렉토리 변경 (여기서부터 모든 상대 경로 기준)
os.chdir(base_dir)
# In[14]:


# 하이퍼파라미터 설정
model_name = 'convnext_base_384_in22ft1k'  # 기존과 동일한 모델
img_size = 512
LR = 2e-4
EPOCHS = 100
BATCH_SIZE = 10
num_workers = 8

# 취약 클래스 설정
vulnerable_classes = [3, 4, 7, 14]
print(f"Target vulnerable classes: {vulnerable_classes}")


# In[15]:


# 데이터셋 클래스 정의 (기존과 동일, __init__만 수정)
class ImageDataset(Dataset):
    def __init__(self, data, path, epoch=0, total_epochs=10, is_train=True):
        if isinstance(data, str):
            df_temp = pd.read_csv(data)
        else:
            df_temp = data
        
        # 수정: 항상 ['ID', 'target'] 컬럼만 선택하여 self.df 초기화
        self.df = df_temp[['ID', 'target']].values
        self.path = path
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.is_train = is_train
        
        # Hard augmentation 확률 계산
        self.p_hard = 0.2 + 0.3 * (epoch / total_epochs) if is_train else 0
        
        # Normal augmentation
        self.normal_aug = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
            A.OneOf([
                A.Rotate(limit=[90, 90], p=1.0),
                A.Rotate(limit=[180, 180], p=1.0),
                A.Rotate(limit=[270, 270], p=1.0),
            ], p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.GaussNoise(var_limit=(30.0, 100.0), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Hard augmentation
        self.hard_aug = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
            A.OneOf([
                A.Rotate(limit=[90, 90], p=1.0),
                A.Rotate(limit=[180, 180], p=1.0),
                A.Rotate(limit=[270, 270], p=1.0),
                A.Rotate(limit=[-15, 15], p=1.0),
            ], p=0.8),
            A.OneOf([
                A.MotionBlur(blur_limit=15, p=1.0),
                A.GaussianBlur(blur_limit=15, p=1.0),
            ], p=0.95),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.9),
            A.GaussNoise(var_limit=(50.0, 150.0), p=0.8),
            A.JpegCompression(quality_lower=70, quality_upper=100, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)).convert('RGB'))
        
        # 배치별 증강 선택
        if self.is_train and random.random() < self.p_hard:
            img = self.hard_aug(image=img)['image']
        else:
            img = self.normal_aug(image=img)['image']
        
        return img, target


# In[16]:


# Mixup 함수 정의
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 학습 함수
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    scaler = GradScaler()
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)
        
        # Cutmix/Mixup 적용 (30% 확률)
        if random.random() < 0.3:
            mixed_x, y_a, y_b, lam = mixup_data(image, targets, alpha=1.0)
            with autocast(): 
                preds = model(mixed_x)
            loss = lam * loss_fn(preds, y_a) + (1 - lam) * loss_fn(preds, y_b)
        else:
            with autocast(): 
                preds = model(image)
            loss = loss_fn(preds, targets)

        model.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

# 검증 함수
def validate_one_epoch(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for image, targets in pbar:
            image = image.to(device)
            targets = targets.to(device)
            
            preds = model(image)
            loss = loss_fn(preds, targets)
            
            val_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(targets.detach().cpu().numpy())
            
            pbar.set_description(f"Val Loss: {loss.item():.4f}")
    
    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')
    
    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
    }


# In[17]:


# ========================================
# 1. 취약 클래스 데이터 준비
# ========================================

# 원본 데이터 로드
train_df = pd.read_csv("../data/train.csv")
print(f"Original dataset size: {len(train_df)}")

# 취약 클래스만 필터링
filtered_df = train_df[train_df['target'].isin(vulnerable_classes)].copy()
print(f"Filtered dataset size: {len(filtered_df)}")

# 클래스별 샘플 수 확인
print("\nClass distribution:")
for cls in vulnerable_classes:
    count = len(filtered_df[filtered_df['target'] == cls])
    print(f"Class {cls}: {count} samples")

# 라벨 재매핑 (3->0, 4->1, 7->2, 14->3)
label_mapping = {3: 0, 4: 1, 7: 2, 14: 3}
filtered_df['original_target'] = filtered_df['target']  # 원본 라벨 보존
filtered_df['target'] = filtered_df['target'].map(label_mapping)

print("\nLabel mapping:")
for orig, new in label_mapping.items():
    print(f"Original class {orig} -> New class {new}")

# 클래스 불균형 확인
print("\nNew class distribution:")
for new_cls in range(4):
    count = len(filtered_df[filtered_df['target'] == new_cls])
    print(f"New class {new_cls}: {count} samples")


# In[ ]:


# ========================================
# 2. 3-Fold Cross Validation으로 서브셋 모델 B 모델 학습
# ========================================

# 3-Fold 설정
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# 결과 저장용 리스트
fold_results = []
fold_models = []

print(f"Starting {N_FOLDS}-Fold Cross Validation for Subset Model...")

for fold, (train_idx, val_idx) in enumerate(skf.split(filtered_df, filtered_df['target'])):
    print(f"\n{'='*50}")
    print(f"SUBSET FOLD {fold + 1}/{N_FOLDS}")
    print(f"{'='*50}")
    
    # 현재 fold의 train/validation 데이터 분할
    train_fold_df = filtered_df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = filtered_df.iloc[val_idx].reset_index(drop=True)
    
    # 현재 fold의 Dataset 생성
    trn_dataset = ImageDataset(
        train_fold_df,
        "../data/train/",
        epoch=0,
        total_epochs=EPOCHS,
        is_train=True
    )
    
    val_dataset = ImageDataset(
        val_fold_df,
        "../data/train/",
        epoch=0,
        total_epochs=EPOCHS,
        is_train=False
    )
    
    # 현재 fold의 DataLoader 생성
    trn_loader = DataLoader(
        trn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(trn_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 모델 초기화 (4개 클래스)
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=4  # 취약 클래스 4개
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 현재 fold의 최고 성능 추적
    best_val_f1 = 0.0
    best_model = None
    
    # 현재 fold 학습
    for epoch in range(EPOCHS):
        # Training
        train_ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device)
        
        # Validation
        val_ret = validate_one_epoch(val_loader, model, loss_fn, device)
        
        # Scheduler step
        scheduler.step()
        
        print(f"Epoch {epoch+1:2d} | "
              f"Train Loss: {train_ret['train_loss']:.4f} | "
              f"Train F1: {train_ret['train_f1']:.4f} | "
              f"Val Loss: {val_ret['val_loss']:.4f} | "
              f"Val F1: {val_ret['val_f1']:.4f}")
        
        # 최고 성능 모델 저장
        if val_ret['val_f1'] > best_val_f1:
            best_val_f1 = val_ret['val_f1']
            best_model = copy.deepcopy(model.state_dict())
    
    # 현재 fold 결과 저장
    fold_results.append({
        'fold': fold + 1,
        'best_val_f1': best_val_f1,
        'train_samples': len(trn_dataset),
        'val_samples': len(val_dataset)
    })
    
    fold_models.append(best_model)
    
    print(f"Subset Fold {fold + 1} Best Validation F1: {best_val_f1:.4f}")

# 결과 요약
print(f"\n{'='*60}")
print("SUBSET MODEL CROSS VALIDATION RESULTS")
print(f"{'='*60}")

val_f1_scores = [result['best_val_f1'] for result in fold_results]
mean_f1 = np.mean(val_f1_scores)
std_f1 = np.std(val_f1_scores)

for result in fold_results:
    print(f"Fold {result['fold']}: {result['best_val_f1']:.4f}")

print(f"\nMean CV F1: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Best single fold: {max(val_f1_scores):.4f}")


# In[8]:


# ========================================
# 3. 서브셋 모델 저장
# ========================================

# 서브셋 모델들 저장
save_dir = "subset_models"
os.makedirs(save_dir, exist_ok=True)

print(f"\nSaving subset models to {save_dir}/")
for fold, state_dict in enumerate(fold_models):
    model_path = f"{save_dir}/subset_fold_{fold}_model.pth"
    torch.save({
        'model_state_dict': state_dict,
        'fold': fold,
        'classes': vulnerable_classes,
        'label_mapping': label_mapping,
        'model_name': model_name,
        'img_size': img_size,
        'num_classes': 4,
        'best_f1': fold_results[fold]['best_val_f1']
    }, model_path)
    print(f"✅ Fold {fold} model saved: {model_path}")

print("\n🎉 4-Class subset training completed!")
print(f"📊 Final Results Summary:")
print(f"   - Target classes: {vulnerable_classes}")
print(f"   - Training samples: {len(filtered_df)}")
print(f"   - Mean CV F1: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"   - Models saved in: {save_dir}/")


# In[ ]:


# ========================================
# 4. 캐스케이드 분류 시스템 구현
# ========================================

class CascadeClassifier:
    """
    2단계 캐스케이드 분류 시스템
    
    1단계: 분류기 A (17개 클래스 전체 분류)
    2단계: 분류기 B (취약 클래스 3,4,7,14만 분류)
    """
    
    def __init__(self, main_models, subset_models, vulnerable_classes=[3,4,7,14], 
                 confidence_threshold=0.4):
        """
        Args:
            main_models: 분류기 A의 앙상블 모델들 (17개 클래스)
            subset_models: 분류기 B의 앙상블 모델들 (4개 클래스)
            vulnerable_classes: 취약 클래스 리스트
            confidence_threshold: 2단계 분류기로 넘어갈 신뢰도 임계값
        """
        self.main_models = main_models
        self.subset_models = subset_models
        self.vulnerable_classes = vulnerable_classes
        self.confidence_threshold = confidence_threshold
        
        # 취약 클래스 매핑 (원본 클래스 -> 서브셋 클래스) / 결과: {3: 0, 4: 1, 7: 2, 14: 3}
        self.class_mapping = {cls: idx for idx, cls in enumerate(vulnerable_classes)}
        
        print(f"캐스케이드 분류기 초기화 완료")
        print(f"- 취약 클래스: {vulnerable_classes}")
        print(f"- 신뢰도 임계값: {confidence_threshold}")
        print(f"- 메인 모델 수: {len(main_models)}")
        print(f"- 서브셋 모델 수: {len(subset_models)}")
    
    def predict_single(self, image, device):
        """
        단일 이미지에 대한 캐스케이드 예측
        
        Args:
            image: 전처리된 이미지 텐서 [C, H, W]
            device: GPU/CPU 디바이스
            
        Returns:
            final_prediction: 최종 예측 클래스
            confidence: 예측 신뢰도
            used_cascade: 사용된 분류기 ('main' 또는 'cascade')
        """
        image = image.unsqueeze(0).to(device)  # 배치 차원 추가
        
        # 1단계: 메인 분류기로 예측
        main_probs = self._predict_main_ensemble(image)
        main_pred = torch.argmax(main_probs, dim=1).item()
        main_confidence = torch.max(main_probs).item()
        
        # 1단계 예측이 취약 클래스이고 신뢰도가 낮으면 2단계로
        if (main_pred in self.vulnerable_classes and 
            main_confidence < self.confidence_threshold):
            
            # 2단계: 서브셋 분류기로 재예측
            subset_probs = self._predict_subset_ensemble(image)
            subset_pred_idx = torch.argmax(subset_probs, dim=1).item()
            subset_confidence = torch.max(subset_probs).item()
            
            # 서브셋 예측을 원본 클래스로 변환
            final_prediction = self.vulnerable_classes[subset_pred_idx]
            final_confidence = subset_confidence
            used_cascade = 'cascade'
            
            print(f"캐스케이드 사용: {main_pred}({main_confidence:.3f}) -> {final_prediction}({subset_confidence:.3f})")
            
        else:
            # 1단계 예측 그대로 사용
            final_prediction = main_pred
            final_confidence = main_confidence
            used_cascade = 'main'
        
        return final_prediction, final_confidence, used_cascade
    
    def _predict_main_ensemble(self, image):
        """메인 분류기 앙상블 예측"""
        ensemble_probs = torch.zeros(1, 17).to(image.device)
        
        with torch.no_grad():
            for model in self.main_models:
                model.eval()
                preds = model(image)
                probs = torch.softmax(preds, dim=1)
                ensemble_probs += probs / len(self.main_models)
        
        return ensemble_probs
    
    def _predict_subset_ensemble(self, image):
        """서브셋 분류기 앙상블 예측"""
        ensemble_probs = torch.zeros(1, 4).to(image.device)
        
        with torch.no_grad():
            for model in self.subset_models:
                model.eval()
                preds = model(image)
                probs = torch.softmax(preds, dim=1)
                ensemble_probs += probs / len(self.subset_models)
        
        return ensemble_probs
    
    def predict_batch(self, dataloader, device):
        """
        배치 데이터에 대한 캐스케이드 예측
        
        Args:
            dataloader: 테스트 데이터로더
            device: GPU/CPU 디바이스
            
        Returns:
            predictions: 최종 예측 리스트
            confidences: 예측 신뢰도 리스트
            cascade_usage: 캐스케이드 사용 통계
        """
        all_predictions = []
        all_confidences = []
        cascade_usage = {'main': 0, 'cascade': 0}
        
        for images, _ in tqdm(dataloader, desc="Cascade Prediction"):
            batch_predictions = []
            batch_confidences = []
            
            for i in range(images.size(0)):
                single_image = images[i]
                pred, conf, used = self.predict_single(single_image, device)
                
                batch_predictions.append(pred)
                batch_confidences.append(conf)
                cascade_usage[used] += 1
            
            all_predictions.extend(batch_predictions)
            all_confidences.extend(batch_confidences)
        
        return all_predictions, all_confidences, cascade_usage


# In[19]:


# ========================================
# 5. 메인 모델과 서브셋 모델 로드
# ========================================

# 메인 모델들 로드 (17개 클래스)
print("메인 모델들 로드 중...")
main_models = []
for fold in range(5):

    #model_path = f"best_model_fold_{fold+1}.pth"
    #model_path = f"fold_{fold+1}_best.pth"
    model_path = f"BH_512_base_best_model_fold_{fold+1}.pth"
    
    if os.path.exists(model_path):
        # 메인 모델 생성 (17개 클래스)
        main_model = timm.create_model(model_name, pretrained=True, num_classes=17).to(device)
        main_model.load_state_dict(torch.load(model_path, map_location=device))
        main_model.eval()
        
        main_models.append(main_model)
        print(f"✅ 메인 모델 {fold+1} 로드 완료")
    else:
        print(f"❌ 메인 모델 {fold+1} 파일을 찾을 수 없습니다: {model_path}")

print(f"총 {len(main_models)}개의 메인 모델 로드 완료")

# 서브셋 모델들 로드 (4개 클래스)
print("\n서브셋 모델들 로드 중...")
subset_models = []
for fold in range(5):
    model_path = f"{save_dir}/subset_fold_{fold}_model.pth"
    
    if os.path.exists(model_path):
        # 서브셋 모델 생성 (4개 클래스)
        subset_model = timm.create_model(model_name, pretrained=True, num_classes=4).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        subset_model.load_state_dict(checkpoint['model_state_dict'])
        subset_model.eval()
        
        subset_models.append(subset_model)
        print(f"✅ 서브셋 모델 {fold} 로드 완료 (F1: {checkpoint['best_f1']:.4f})")
    else:
        print(f"❌ 서브셋 모델 {fold} 파일을 찾을 수 없습니다: {model_path}")

print(f"총 {len(subset_models)}개의 서브셋 모델 로드 완료")


# In[15]:


# ========================================
# 6. 캐스케이드 분류기 초기화 및 테스트
# ========================================

# 캐스케이드 분류기 초기화
cascade_classifier = CascadeClassifier(
    main_models=main_models,      # 분류기 A (17개 클래스)
    subset_models=subset_models,  # 분류기 B (4개 클래스)
    vulnerable_classes=vulnerable_classes, # 취약 클래스
    confidence_threshold=0.7      # 신뢰도 임계값 (조정 가능)
)

# 간단한 테스트 (취약 클래스 샘플로)
print("\n" + "="*50)
print("캐스케이드 분류기 테스트")
print("="*50)

# 취약 클래스 샘플 하나 가져오기
test_sample = filtered_df.iloc[0]
test_image_path = f"../data/train/{test_sample['ID']}"
test_image = Image.open(test_image_path).convert('RGB')

# 이미지 전처리
transform = A.Compose([
    A.LongestMaxSize(max_size=img_size),
    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

test_tensor = transform(image=np.array(test_image))['image']
true_class = test_sample['original_target']  # 원본 클래스

print(f"테스트 이미지: {test_sample['ID']}")
print(f"실제 클래스: {true_class}")

# 캐스케이드 예측
pred, conf, used = cascade_classifier.predict_single(test_tensor, device)

print(f"예측 결과: {pred}")
print(f"예측 신뢰도: {conf:.4f}")
print(f"사용된 분류기: {used}")
print(f"정답 여부: {'✅' if pred == true_class else '❌'}")


# In[16]:


# ========================================
# 7. 캐스케이드 시스템으로 테스트 데이터 예측
# ========================================


# 테스트 데이터 로드
test_df = pd.read_csv("../data/sample_submission.csv")
print(f"테스트 데이터 크기: {len(test_df)}")

# 테스트 데이터셋 생성
test_dataset = ImageDataset(
    test_df,
    "../data/test/",
    epoch=0,
    total_epochs=EPOCHS,
    is_train=False  # 테스트이므로 증강 없음
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,  # 배치 크기 줄임
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

print("캐스케이드 시스템으로 테스트 데이터 예측 시작...")

# 캐스케이드 예측 실행
test_predictions, test_confidences, cascade_usage = cascade_classifier.predict_batch(
    test_loader, device
)

print(f"\n캐스케이드 사용 통계:")
print(f"- 메인 분류기만 사용: {cascade_usage['main']}개 ({cascade_usage['main']/len(test_predictions)*100:.1f}%)")
print(f"- 캐스케이드 사용: {cascade_usage['cascade']}개 ({cascade_usage['cascade']/len(test_predictions)*100:.1f}%)")

# 결과 저장
result_df = test_df.copy()
result_df['target'] = test_predictions
result_df['confidence'] = test_confidences




print(f"\n✅ 캐스케이드 예측 완료!")

print(f"📊 예측 통계:")
print(f"   - 평균 신뢰도: {np.mean(test_confidences):.4f}")
print(f"   - 최소 신뢰도: {np.min(test_confidences):.4f}")
print(f"   - 최대 신뢰도: {np.max(test_confidences):.4f}")

# 클래스별 예측 분포
print(f"\n📈 클래스별 예측 분포:")
for cls in range(17):
    count = sum(1 for p in test_predictions if p == cls)
    print(f"   Class {cls}: {count}개 ({count/len(test_predictions)*100:.1f}%)")


# In[ ]:


# 결과 저장
result_df = test_df.copy()
result_df['target'] = test_predictions
result_df['confidence'] = test_confidences

# submission 파일 저장
output_path = "../data/output/cascade_submission3.csv"
print(f"📁 결과 저장: {output_path}")
result_df[['ID', 'target']].to_csv(output_path, index=False)


# In[ ]:




