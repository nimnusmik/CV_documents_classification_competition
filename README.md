# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
**서버 제공사**
- **Upstage 제공 서버** 사용
- SSH 접속을 통한 원격 개발 환경

**GPU 사양**
- NVIDIA GeForce RTX 3090
- VRAM: 24GB
- CUDA Version: 12.2
- Driver Version: 535.86.10

**시스템 환경**
- Linux 기반 서버
- CUDA 지원 환경


### Requirements
**기본 제공 패키지 (requirements.txt)**

- albumentations==1.3.1
- ipykernel==6.27.1
- ipython==8.15.0
- ipywidgets==8.1.1
- jupyter==1.0.0
- matplotlib-inline==0.1.6
- numpy==1.26.0
- pandas==2.1.4
- Pillow==9.4.0
- timm==0.9.12


**추가 설치 패키지**
- matplotlib
- seaborn
- libgl1-mesa-glx

**딥러닝 & 데이터 처리**
- **PyTorch**: 딥러닝 프레임워크
- **timm**: 사전훈련 모델 라이브러리
- **albumentations**: 데이터 증강
- **pandas, numpy**: 데이터 처리

**실험 관리 & 최적화**
- **wandb**: 실험 추적 및 시각화
- **Mixed Precision (AMP)**: 메모리 효율화

## 1. Competiton Info

### Overview

- 문서 이미지 분류 대회로, 17개 클래스의 다양한 문서 유형(의료 문서, 신분증, 자동차 관련 문서 등)을 분류하는 과제. 주어진 학습 데이터 1,570장과 평가 데이터 3,140장을 활용하여 정확한 문서 분류 모델을 개발하는게 목표.

### Timeline

- 2025.09.01 : 대회 시작
- ~ 2025.09.03 : EDA/Preprocessing
- ~ 2025.09.05 : 모델링
- ~ 2025.09.17 : 모델 고도화 및 최적화 

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview

**학습 데이터셋**
- **train 폴더**: 1,570장의 이미지
- **train.csv**: 1,570개 행, 학습 이미지의 정답 클래스 정보
  - ID: 학습 샘플의 파일명
  - target: 학습 샘플의 정답 클래스 번호
- **meta.csv**: 17개 행, 클래스 정보
  - target: 17개의 클래스 번호
  - class_name: 클래스 번호에 대응하는 클래스 이름

**평가 데이터셋**
- **test 폴더**: 3,140장의 이미지
- **sample_submission.csv**: 3,140개 행, 제출용 파일
  - ID: 평가 샘플의 파일명
  - target: 예측 결과 입력 컬럼 (초기값 0)

**데이터 특징**
- 평가 데이터는 학습 데이터와 달리 랜덤하게 Rotation 및 Flip 등이 적용됨
- 훼손된 이미지들이 존재함

### EDA

**1. 클래스 분포 분석**
- **총 17개 클래스**, 전체 **1,570개** 이미지
- **균등 분포**: 13개 클래스가 정확히 **100개씩**
- **불균형 클래스**:
  - 클래스 1: **46개** (임신의료비지급신청서)
  - 클래스 13: **74개** (이력서)
  - 클래스 14: **50개** (의견서)
- **불균형 비율**: 최대 2.17배 차이

**2. 이미지 해상도 분석**
- **Train 해상도 특성**:
  - Width: 450px 근처 강한 집중
  - Height: 600px 근처 강한 집중
  - 특징: 단일 해상도 환경 (450×600 중심)

- **Test 해상도 특성**:
  - 두 개의 명확한 클러스터:
    - **400px 그룹**: Width/Height 모두 400px 근처
    - **600px 그룹**: Width/Height 모두 600px 근처
  - 특징: 다중 해상도 환경으로 분산

- **정리**:
  - Train: 단일 해상도 환경 (일관된 스캔/촬영 조건)
  - Test: 다중 해상도 환경 (다양한 장비/조건)
  - 다양한 종횡비를 가지는 이미지
  - Train vs. Test 종횡비 분포 차이

**3. 이미지 품질 분석**
- **대비 분포**: 대부분 적당한 대비, 일부 고대비 이미지 존재
- **선명도 분포**: 대부분 적절한 선명도, 일부 블러된 이미지
- **파일 크기 분포**: 대부분 비슷한 압축률, 일부 고해상도/저압축
- **밝기 분포**: 다양한 조명 조건

**4. 도메인 분류**
- **의료 도메인**: 진단서, 처방전, 외래증명서, 약국영수증, 진료비영수증
- **신원/신분 도메인**: 여권, 주민등록증, 운전면허증
- **자동차 도메인**: 대시보드, 등록증, 번호판
- **기타 문서**: 이력서, 의견서, 계좌번호, 각종 신청서

**5. 도메인 특성**
- 다양한 시각적 특징: 각 도메인별로 완전히 다른 외형
- 하나의 CNN으로 모든 도메인 분류의 한계 예상

### Data Processing

**EDA 기반 증강 전략**
1. **Mixup 데이터 증강**
2. **종횡비 보존 + 패딩**
3. **Augmentation**: 회전/밝기/대비/노이즈/블러 등
   - **Normal Augmentation**
   - **점증적 Hard Augmentation (30% ~ 50%)**
   - **Progressive Hard Augmentation**
4. **ImageNet 표준화**

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
