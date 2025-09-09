# src/inference/infer_highperf.py
"""
고성능 추론 파이프라인
- Swin Transformer & ConvNext 지원: 최신 모델 아키텍처 지원
- Test Time Augmentation (TTA): 다양한 증강으로 예측 정확도 향상
- 앙상블 예측: 여러 모델 결과 통합
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import os                                            # 파일/디렉터리 경로 처리
import torch                                         # PyTorch 메인 모듈
import pandas as pd                                  # 데이터프레임 처리
import numpy as np                                   # 수치 계산 라이브러리
from torch.utils.data import DataLoader, Dataset    # 데이터 로더 클래스, 데이터셋 클래스
import torch.nn.functional as F                      # PyTorch 함수형 인터페이스
from typing import Optional                          # 타입 힌트 (옵셔널)
from tqdm import tqdm                                # 진행률 표시바

# ------------------------- 프로젝트 유틸 Import ------------------------- #
from src.utils import (
    load_yaml, resolve_path, require_file, require_dir, create_log_path
)  # 핵심 유틸리티
from src.logging.logger import Logger                # 로그 기록 클래스
from src.data.dataset import HighPerfDocClsDataset   # 고성능 문서 분류 데이터셋
from src.data.transforms import get_essential_tta_transforms, get_tta_transforms_by_type  # TTA 변환 함수들
from src.models.build import build_model, get_recommended_model  # 모델 빌드/추천 함수

# ------------------------- 시각화 및 출력 관리 ------------------------- #
from src.utils.visualizations import visualize_inference_pipeline, create_organized_output_structure



# ---------------------- Essential TTA 데이터셋 ---------------------- #
class ConfigurableTTADataset(Dataset):
    """설정 가능한 TTA를 위한 데이터셋"""
    
    def __init__(self, csv_path, image_dir, img_size=384, tta_type="essential", test_df=None, id_col="ID"):
        """
        초기화
        
        Args:
            csv_path: 추론할 데이터 CSV 경로 (None이면 test_df 사용)
            image_dir: 이미지 디렉토리 경로  
            img_size: 이미지 크기
            tta_type: "essential" (5가지) 또는 "comprehensive" (15가지)
            test_df: CSV 대신 사용할 DataFrame (캘리브레이션 모드용)
            id_col: ID 컬럼명
        """
        if test_df is not None:
            self.df = test_df
        else:
            self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.img_size = img_size
        self.tta_type = tta_type
        self.id_col = id_col
        self.transforms = get_tta_transforms_by_type(tta_type, img_size)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 샘플의 모든 TTA 변형 반환
        
        Returns:
            (augmented_images, image_id): TTA 변형 리스트, 이미지 ID
        """
        from PIL import Image
        
        row = self.df.iloc[idx]
        image_id = str(row[self.id_col])
        
        # 이미지 로드
        img_path = os.path.join(self.image_dir, image_id)
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # 모든 TTA 변형 적용
        augmented_images = []
        for transform in self.transforms:
            aug_img = transform(image=img)['image']
            augmented_images.append(aug_img)
            
        return augmented_images, image_id


# ---------------------- TTA 예측 함수 ---------------------- #
@torch.no_grad()    # gradient 계산 비활성화
# TTA 예측 함수 정의 (기존 방식 - 호환성 유지)
def predict_with_tta(model, loader, device, num_tta=5):
    """Test Time Augmentation을 사용한 예측 (기존 방식)"""
    model.eval()                                     # 모델을 평가 모드로 설정
    all_preds = []                                   # 모든 TTA 예측 결과 저장 리스트
    
    # TTA 횟수만큼 반복
    for _ in range(num_tta):
        batch_preds = []                             # 배치별 예측 결과 저장 리스트
        
        # 배치별 추론 시작
        for imgs, _ in tqdm(loader, desc="TTA Inference"):
            imgs = imgs.to(device)                   # 이미지를 GPU로 이동
            logits = model(imgs)                     # 모델 순전파
            probs = F.softmax(logits, dim=1)         # 로짓을 확률로 변환
            batch_preds.append(probs.cpu())          # 예측 결과를 CPU로 이동하여 저장
        
        # 배치 결합
        tta_preds = torch.cat(batch_preds, dim=0)    # 모든 배치 예측 결과 연결
        all_preds.append(tta_preds)                  # TTA 예측 결과 추가
    
    # TTA 평균
    final_preds = torch.stack(all_preds).mean(dim=0) # 모든 TTA 결과의 평균 계산
    return final_preds                               # 최종 예측 결과 반환


# ---------------------- Essential TTA 예측 함수 ---------------------- #
@torch.no_grad()    # gradient 계산 비활성화
def predict_with_essential_tta(model, tta_loader, device):
    """팀원의 Essential TTA를 사용한 예측"""
    model.eval()                                     # 모델을 평가 모드로 설정
    all_predictions = []                             # 모든 예측 결과 저장 리스트
    
    for batch_idx, (images_list, _) in enumerate(tqdm(tta_loader, desc="Essential TTA")):
        batch_size = images_list[0].size(0)          # 배치 크기 추출
        batch_probs = torch.zeros(batch_size, 17).to(device)  # 17개 클래스에 대한 확률 초기화
        
        # 각 TTA 변형별 예측
        for images in images_list:                   # 5가지 TTA 변형 순회
            images = images.to(device)               # 이미지를 GPU로 이동
            logits = model(images)                   # 모델 순전파  
            probs = F.softmax(logits, dim=1)         # 로짓을 확률로 변환
            batch_probs += probs / len(images_list)  # 평균을 위해 누적
            
        all_predictions.append(batch_probs.cpu())    # CPU로 이동하여 저장
    
    # 모든 배치 결합
    final_predictions = torch.cat(all_predictions, dim=0)
    return final_predictions                         # 최종 예측 확률 반환


# ---------------------- 폴드 모델 로드 함수 ---------------------- #
# 폴드별 학습된 모델들 로드 함수 정의
def load_fold_models(fold_results_path, device):
    fold_results = load_yaml(fold_results_path)      # 폴드 결과 YAML 파일 로드
    models = []                                      # 모델 리스트 초기화
    
    # 각 폴드 정보 반복
    for fold_info in fold_results["fold_results"]:
        model_path = fold_info["model_path"]         # 모델 체크포인트 경로 추출
        
        # 모델 파일 존재 확인
        if os.path.exists(model_path):
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=device)
            # 체크포인트를 모델 리스트에 추가
            models.append(checkpoint)
        # 모델 파일이 없는 경우
        else:
            # 경고 메시지 출력
            print(f"Warning: Model not found: {model_path}")
    
    # 로드된 모델들 반환
    return models


# ---------------------- 앙상블 예측 함수 ---------------------- #
# 앙상블 예측 함수 정의 (기존 방식)
def ensemble_predict(models, test_loader, cfg, device, use_tta=True):
    all_ensemble_preds = [] # 모든 앙상블 예측 결과 저장 리스트
    
    # 각 모델 체크포인트 반복
    for i, checkpoint in enumerate(models):
        print(f"Processing model {i+1}/{len(models)}...")           # 현재 처리 중인 모델 번호 출력
        
        # 모델 생성 및 가중치 로드
        model_name = get_recommended_model(cfg["model"]["name"])    # 권장 모델명 추출
        
        # 모델 빌드
        model = build_model(
            model_name,                                             # 모델명
            cfg["data"]["num_classes"],                             # 클래스 수
            pretrained=False,                                       # 가중치는 체크포인트에서 로드
            drop_rate=cfg["model"]["drop_rate"],                    # 드롭아웃 비율
            drop_path_rate=cfg["model"]["drop_path_rate"],          # 드롭패스 비율
            pooling=cfg["model"]["pooling"]                         # 풀링 타입
        ).to(device)                                                # GPU로 모델 이동
        
        # 체크포인트에서 가중치 로드
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # TTA 사용 시
        if use_tta:                                  
            # TTA 예측 수행
            preds = predict_with_tta(model, test_loader, device, num_tta=3)
        # TTA 미사용 시
        else:
            model.eval()                                            # 모델을 평가 모드로 설정
            batch_preds = []                                        # 배치별 예측 결과 저장 리스트
            for imgs, _ in tqdm(test_loader, desc=f"Model {i+1} Inference"):  # 배치별 추론 시작
                imgs = imgs.to(device)                              # 이미지를 GPU로 이동
                logits = model(imgs)                                # 모델 순전파
                probs = F.softmax(logits, dim=1)                    # 로짓을 확률로 변환
                batch_preds.append(probs.cpu())                     # 예측 결과를 CPU로 이동하여 저장
                
            # 모든 배치 예측 결과 결합
            preds = torch.cat(batch_preds, dim=0)
        
        # 현재 모델 예측 결과를 앙상블 리스트에 추가
        all_ensemble_preds.append(preds)
        
        # 메모리 정리
        del model                   # 모델 객체 삭제
        torch.cuda.empty_cache()    # GPU 메모리 캐시 정리
    
    # 앙상블 평균
    ensemble_preds = torch.stack(all_ensemble_preds).mean(dim=0)    # 모든 모델 예측 결과의 평균 계산
    return ensemble_preds                                           # 앙상블 예측 결과 반환


# ---------------------- Essential TTA 앙상블 예측 함수 ---------------------- #
def ensemble_predict_with_essential_tta(models, tta_loader, cfg, device):
    """팀원의 Essential TTA를 사용한 앙상블 예측"""
    print(f"🚀 Essential TTA 앙상블 예측 시작 (모델 수: {len(models)})")
    
    all_ensemble_preds = []  # 모든 앙상블 예측 결과 저장 리스트
    
    # 각 모델 체크포인트 반복
    for i, checkpoint in enumerate(models):
        print(f"📊 모델 {i+1}/{len(models)} 처리 중...")
        
        # 모델 생성 및 가중치 로드
        model_name = get_recommended_model(cfg["model"]["name"])
        
        # 모델 빌드
        model = build_model(
            model_name,
            cfg["data"]["num_classes"],
            pretrained=False,
            drop_rate=cfg["model"]["drop_rate"],
            drop_path_rate=cfg["model"]["drop_path_rate"],
            pooling=cfg["model"]["pooling"]
        ).to(device)
        
        # 체크포인트에서 가중치 로드
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Essential TTA 예측 수행
        model_preds = predict_with_essential_tta(model, tta_loader, device)
        all_ensemble_preds.append(model_preds)
        
        print(f"✅ 모델 {i+1} 완료 (예측 형태: {model_preds.shape})")
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
    
    # 앙상블 평균 계산
    print("🔄 앙상블 평균 계산 중...")
    ensemble_preds = torch.stack(all_ensemble_preds).mean(dim=0)
    
    print(f"🎉 Essential TTA 앙상블 예측 완료! 최종 예측 형태: {ensemble_preds.shape}")
    return ensemble_preds


# ---------------------- 설정 가능한 TTA 헬퍼 함수 ---------------------- #
def create_configurable_tta_dataloader(sample_csv, test_dir, img_size=384, tta_type="essential", batch_size=32, num_workers=8):
    """설정 가능한 TTA 데이터로더 생성 헬퍼 함수"""
    
    # TTA 데이터셋 생성
    tta_dataset = ConfigurableTTADataset(sample_csv, test_dir, img_size, tta_type)
    
    # TTA 변형 수 계산
    num_tta_transforms = len(tta_dataset.transforms)
    
    # 데이터로더 생성
    tta_loader = DataLoader(
        tta_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: (
            [torch.stack([item[0][i] for item in x]) for i in range(num_tta_transforms)],  # TTA 변형들
            [item[1] for item in x]  # 이미지 ID들
        )
    )
    
    tta_type_name = "Essential (5가지)" if tta_type == "essential" else "Comprehensive (15가지)"
    print(f"🔧 {tta_type_name} TTA 데이터로더 생성 완료")
    print(f"   - 데이터셋 크기: {len(tta_dataset)}")
    print(f"   - 배치 크기: {batch_size}")  
    print(f"   - TTA 변형 수: {num_tta_transforms}가지")
    
    return tta_loader


# ---------------------- 하위 호환성을 위한 래퍼 함수 ---------------------- #
def create_essential_tta_dataloader(sample_csv, test_dir, img_size=384, batch_size=32, num_workers=8):
    """Essential TTA 데이터로더 생성 (하위 호환성)"""
    return create_configurable_tta_dataloader(sample_csv, test_dir, img_size, "essential", batch_size, num_workers)


# ---------------------- 사용 예제 (주석) ---------------------- #
"""
팀원의 Essential TTA를 사용한 추론 예제:

```python
from src.inference.infer_highperf import (
    load_fold_models, 
    create_essential_tta_dataloader,
    ensemble_predict_with_essential_tta
)
from src.data.transforms import get_essential_tta_transforms

# 1. 폴드 모델들 로드
models = load_fold_models("./experiments/train/lastest-train/fold_results.yaml", device)

# 2. Essential TTA 데이터로더 생성
tta_loader = create_essential_tta_dataloader(
    sample_csv="../data/raw/sample_submission.csv",
    test_dir="../data/raw/test",
    img_size=384,
    batch_size=32
)

# 3. Essential TTA 앙상블 예측
ensemble_probs = ensemble_predict_with_essential_tta(models, tta_loader, cfg, device)

# 4. 최종 예측 및 저장
predictions = torch.argmax(ensemble_probs, dim=1).numpy()
```

주요 개선사항:
- ✅ 팀원의 5가지 Essential TTA 구현 (원본, 90°, 180°, 270°, 밝기개선)
- ✅ 기존 단순 반복 TTA 대신 다양한 변형 적용
- ✅ 앙상블 + Essential TTA 조합으로 성능 향상 기대
- ✅ 기존 코드 호환성 유지 (predict_with_tta 함수 보존)
"""


# ---------------------- 고성능 추론 파이프라인 실행 함수 ---------------------- #
# 고성능 추론 파이프라인 실행 함수 정의
def run_highperf_inference(cfg_path: str, fold_results_path: str, output_path: Optional[str] = None):
    # 설정 로드
    cfg = load_yaml(cfg_path)                                       # YAML 설정 파일 로드
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))            # 설정 파일 디렉터리 경로
    
    # 로거 설정
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    logger = Logger(
        log_path=create_log_path("infer", f"infer_highperf_{timestamp}.log")  # 날짜별 로그 파일 경로
    )
    
    # 파이프라인 시작 로그
    logger.write("[BOOT] high-performance inference pipeline started")
    
    try:
        # GPU/CPU 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.write(f"[BOOT] device={device}")                         # 디바이스 정보 로그
        
        # 경로 확인
        sample_csv = resolve_path(cfg_dir, cfg["data"]["sample_csv"])   # 샘플 CSV 경로 해결
        test_dir = resolve_path(cfg_dir, cfg["data"]["image_dir_test"]) # 테스트 이미지 디렉터리 경로 해결
        require_file(sample_csv, "sample_csv 확인")                     # 샘플 CSV 파일 존재성 검증
        require_dir(test_dir, "test_dir 확인")                          # 테스트 디렉터리 존재성 검증
        
        # 테스트 데이터 로드
        test_df = pd.read_csv(sample_csv)                               # 테스트 데이터 CSV 로드
        logger.write(f"[DATA] loaded test data | shape={test_df.shape}")# 테스트 데이터 로드 로그
        
        # 테스트 데이터셋 생성
        test_ds = HighPerfDocClsDataset(
            test_df,                                 # 테스트 데이터프레임
            test_dir,                                # 테스트 이미지 디렉터리
            img_size=cfg["train"]["img_size"],       # 이미지 크기
            is_train=False,                          # 평가 모드 플래그
            id_col=cfg["data"]["id_col"],            # ID 컬럼명
            target_col=None,                         # 추론 모드 (타깃 없음)
            logger=logger                            # 로거 객체
        )                                            # 테스트 데이터셋 생성 완료
        
        # 테스트 데이터로더
        test_loader = DataLoader(                    # 테스트용 데이터로더 생성
            test_ds,                                 # 테스트 데이터셋
            batch_size=cfg["train"]["batch_size"],   # 배치 크기
            shuffle=False,                           # 셔플 비활성화 (추론용)
            num_workers=cfg["project"]["num_workers"],  # 워커 프로세스 수
            pin_memory=True                          # 메모리 고정 활성화
        )
        
        logger.write(f"[DATA] test dataset size: {len(test_ds)}")  # 테스트 데이터셋 크기 로그
        
        # TTA 설정 확인
        tta_enabled = cfg.get("inference", {}).get("tta", True)
        tta_type = cfg.get("inference", {}).get("tta_type", "essential")
        logger.write(f"[TTA] TTA enabled: {tta_enabled}, type: {tta_type}")
        
        # 폴드별 모델 로드
        models = load_fold_models(fold_results_path, device)
        logger.write(f"[MODELS] loaded {len(models)} fold models")
        
        if tta_enabled:
            # 설정 가능한 TTA 데이터로더 생성
            tta_loader = create_configurable_tta_dataloader(
                sample_csv=sample_csv,
                test_dir=test_dir,
                img_size=cfg["train"]["img_size"],
                tta_type=tta_type,
                batch_size=cfg["train"]["batch_size"],
                num_workers=cfg["project"]["num_workers"]
            )
            
            # TTA 앙상블 예측 수행
            logger.write(f"[INFERENCE] starting {tta_type} TTA ensemble prediction...")
            ensemble_preds = ensemble_predict_with_essential_tta(models, tta_loader, cfg, device)
        else:
            # 기본 앙상블 예측 (TTA 없음)
            logger.write(f"[INFERENCE] starting basic ensemble prediction (no TTA)...")
            ensemble_preds = ensemble_predict(models, test_loader, cfg, device, use_tta=False)
        
        # 최종 예측 클래스
        final_predictions = ensemble_preds.argmax(dim=1).numpy()    # 가장 높은 확률의 클래스 선택
        
        # 신뢰도 점수 계산
        confidence_scores = ensemble_preds.max(dim=1)[0].numpy()    # 최대 확률값을 신뢰도로 사용
        
        #-------------- 결과 저장 및 로그 ---------------------- #
        # 출력 경로가 지정되지 않은 경우 동적 파일명 생성
        if output_path is None:
            current_date = pd.Timestamp.now().strftime('%Y%m%d')
            current_time = pd.Timestamp.now().strftime('%H%M')
            model_name = cfg["model"]["name"]
            
            # TTA 타입 포함한 파일명 생성
            tta_suffix = f"_{tta_type}_tta" if tta_enabled else "_no_tta"
            
            filename = f"{current_date}_{current_time}_{model_name}_ensemble{tta_suffix}.csv"
            output_path = f"submissions/{current_date}/{filename}"
        
        # 출력 디렉터리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 제출 파일 생성
        submission = test_df.copy()                                             # 테스트 데이터프레임 복사
        submission[cfg["data"]["target_col"]] = final_predictions               # 예측 결과 추가
        submission.to_csv(output_path, index=False)                             # CSV 파일로 저장
        
        logger.write(f"[SUCCESS] Inference completed | output: {output_path}")  # 추론 완료 로그
        logger.write(f"[RESULT] Prediction distribution:")                      # 예측 분포 로그 시작
        
        # 각 클래스별 예측 수 계산
        for i, count in enumerate(np.bincount(final_predictions)):
            # 클래스별 분포 로그
            logger.write(f"  Class {i}: {count} samples ({count/len(final_predictions)*100:.1f}%)")
        
        #-------------- 추론 결과 시각화 ---------------------- #
        try:
            # 시각화를 위한 출력 디렉터리 설정
            viz_output_dir = os.path.dirname(output_path)
            model_name = cfg["model"]["name"]
            
            # 시각화 생성
            visualize_inference_pipeline(
                predictions=ensemble_preds.numpy(),
                model_name=model_name,
                output_dir=viz_output_dir,
                confidence_scores=confidence_scores
            )
            logger.write(f"[VIZ] Inference visualizations created in {viz_output_dir}")
            
        except Exception as viz_error:
            logger.write(f"[WARNING] Visualization failed: {str(viz_error)}")
        
        #-------------- lastest-infer 폴더에 결과 저장 ---------------------- #
        try:
            import shutil
            import time
            
            # experiments/infer/날짜/실험명/ 구조 생성
            date_str = time.strftime('%Y%m%d')
            timestamp = time.strftime('%Y%m%d_%H%M')
            run_name = cfg.get("project", {}).get("run_name", "inference")
            
            # 날짜별 infer 결과 디렉터리
            infer_output_dir = f"experiments/infer/{date_str}/{timestamp}_{run_name}"
            os.makedirs(infer_output_dir, exist_ok=True)
            
            # lastest-infer 폴더에 직접 저장 (기존 내용 삭제 후)
            lastest_infer_dir = "experiments/infer/lastest-infer"
            
            # 기존 lastest-infer 폴더 삭제 (완전 교체)
            if os.path.exists(lastest_infer_dir):
                shutil.rmtree(lastest_infer_dir)
                logger.write(f"[CLEANUP] Removed existing lastest-infer folder")
            
            os.makedirs(lastest_infer_dir, exist_ok=True)
            
            # 추론 결과 CSV를 lastest-infer에 복사
            import copy
            lastest_output_path = os.path.join(lastest_infer_dir, f"submission_{timestamp}.csv")
            shutil.copy2(output_path, lastest_output_path)
            
            # 설정 파일도 복사
            import yaml
            config_copy_path = os.path.join(lastest_infer_dir, "config.yaml")
            with open(config_copy_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            
            # 시각화 결과도 복사 (있다면)
            viz_source_dir = os.path.dirname(output_path)
            if os.path.exists(os.path.join(viz_source_dir, "images")):
                shutil.copytree(os.path.join(viz_source_dir, "images"), 
                               os.path.join(lastest_infer_dir, "images"))
            
            logger.write(f"[COPY] Results copied directly to lastest-infer")
            logger.write(f"📁 Latest inference results: {lastest_infer_dir}")
            
        except Exception as copy_error:
            logger.write(f"[WARNING] Failed to copy to lastest-infer: {str(copy_error)}")
        
        # 출력 파일 경로 반환
        return output_path
        
    # 예외 발생 시
    except Exception as e:
        logger.write(f"[ERROR] Inference failed: {str(e)}") # 에러 로그
        raise                                               # 예외 재발생
    # 최종적으로 실행
    finally:
        logger.write("[SHUTDOWN] Inference pipeline ended") # 파이프라인 종료 로그


# ---------------------- 메인 실행 블록 ---------------------- #
if __name__ == "__main__":
    import sys  # sys 모듈 import
    
    # 인자 개수 확인
    if len(sys.argv) < 3:
        print("Usage: python infer_highperf.py <config_path> <fold_results_path> [output_path]")    # 사용법 출력
        sys.exit(1)                                                                                 # 프로그램 종료
    
    cfg_path = sys.argv[1]                                                          # 설정 파일 경로
    fold_results_path = sys.argv[2]                                                 # 폴드 결과 파일 경로
    output_path = sys.argv[3] if len(sys.argv) > 3 else None                        # 출력 경로 (선택사항)
    
    result_path = run_highperf_inference(cfg_path, fold_results_path, output_path)  # 고성능 추론 실행
    print(f"Inference completed! Results saved to: {result_path}")                  # 추론 완료 메시지 출력
    
    run_highperf_inference(cfg_path, fold_results_path, output_path)                # 고성능 추론 실행
