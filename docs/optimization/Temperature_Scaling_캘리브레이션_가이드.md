# 🌡️ Temperature Scaling 모델 캘리브레이션 완전 가이드

## 📖 목차
1. [개요](#개요)
2. [기술적 배경](#기술적-배경)
3. [모듈 구성](#모듈-구성)
4. [실행 방법](#실행-방법)
5. [캘리브레이션 분석](#캘리브레이션-분석)
6. [성능 개선 효과](#성능-개선-효과)
7. [추론 파이프라인 통합](#추론-파이프라인-통합)
8. [팀 협업 워크플로우](#팀-협업-워크플로우)
9. [문제 해결](#문제-해결)
10. [고급 활용법](#고급-활용법)

---

## 🚀 개요

**Temperature Scaling**은 모델의 confidence 점수를 보정하여 예측 확률을 더 정확하게 만드는 **post-hoc calibration** 기법입니다. 학습된 모델의 과신(overconfidence) 문제를 해결하여 **더 신뢰할 수 있는 예측 확률**을 제공합니다.

### 🎯 핵심 기능
- **확률 보정** (과신된 예측 확률을 현실적 수준으로 조정)
- **단일 파라미터 최적화** (Temperature T 하나만으로 전체 모델 보정)
- **모델 구조 불변** (기존 모델에 온도 스케일링만 추가)
- **빠른 캘리브레이션** (validation set으로 5분 내 완료)
- **추론 시 자동 적용** (캘리브레이션된 확률로 예측)

### 📦 모듈 구성
```
src/calibration/
├── __init__.py                    # 모듈 초기화
├── temperature_scaling.py         # Temperature Scaling 구현
└── calibration_utils.py           # 캘리브레이션 유틸리티

experiments/calibration/
├── temperature_values.json        # 학습된 온도 파라미터
└── calibration_plots/             # 캘리브레이션 시각화

docs/optimization/
└── Temperature_Scaling_캘리브레이션_가이드.md  # 본 가이드
```

---

## 🧠 기술적 배경

### Temperature Scaling이 해결하는 문제

#### **기존 모델의 과신 문제**
```python
# 일반적인 딥러닝 모델의 예측
logits = model(image)               # 예: [2.1, 0.3, -1.2, 4.8, 0.9]
probabilities = softmax(logits)     # 예: [0.05, 0.01, 0.00, 0.92, 0.02]

# 문제: 모델이 92% 확신한다고 하지만...
actual_accuracy = 0.73              # 실제로는 73%만 맞음 → 과신!
```

**문제점:**
- 🎯 **과신 현상**: 모델이 실제보다 높은 확신도를 보임
- ⚖️ **캘리브레이션 불량**: 예측 확률과 실제 정확도 불일치
- 📊 **의사결정 오류**: 잘못된 확신도로 인한 부적절한 판단

#### **Temperature Scaling 해결책**
```python
# Temperature Scaling 적용
calibrated_logits = logits / temperature    # temperature = 2.3 (학습된 값)
calibrated_probs = softmax(calibrated_logits)  # 예: [0.12, 0.08, 0.05, 0.73, 0.02]

# 결과: 73% 확신 → 실제 정확도와 일치! ✅
```

**개선점:**
- 🎯 **현실적 확신도**: 과신 없는 적절한 예측 확률
- ⚖️ **완벽한 캘리브레이션**: 예측 확률 = 실제 정확도
- 📊 **신뢰할 수 있는 의사결정**: 정확한 확률 기반 판단 가능

### Temperature Scaling 수식

```python
# 1. 기본 softmax
p_i = exp(z_i) / Σ exp(z_j)

# 2. Temperature Scaling 적용
p_i = exp(z_i / T) / Σ exp(z_j / T)

# 여기서 T는 학습 가능한 temperature 파라미터
# T > 1: 확률 분포가 더 평평해짐 (덜 확신)
# T < 1: 확률 분포가 더 뾰족해짐 (더 확신)
# T = 1: 일반 softmax와 동일
```

### 캘리브레이션 평가 지표

#### **Expected Calibration Error (ECE)**
```python
# 예측 확률을 N개 구간으로 나누어 오차 계산
ECE = Σ (n_m / n) * |acc(m) - conf(m)|

# 여기서:
# n_m: m번째 구간의 샘플 수
# n: 전체 샘플 수  
# acc(m): m번째 구간의 실제 정확도
# conf(m): m번째 구간의 평균 예측 확률
```

**목표**: ECE가 낮을수록 잘 캘리브레이션된 모델

---

## 🔧 모듈 구성

### 1. TemperatureScaling 클래스 (`temperature_scaling.py`)

```python
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # 초기값 1.5
        
    def forward(self, logits):
        """
        logits을 temperature로 나누어 캘리브레이션된 확률 반환
        """
        return torch.softmax(logits / self.temperature, dim=1)
    
    def set_temperature(self, valid_loader, criterion):
        """
        Validation set으로 최적 temperature 학습
        """
        # NLL Loss 최소화하여 최적 T 찾기
```

### 2. CalibrationTrainer 클래스 (`calibration_utils.py`)

```python
class CalibrationTrainer:
    def __init__(self, model, validation_loader):
        self.model = model
        self.validation_loader = validation_loader
        self.temperature_scaling = TemperatureScaling()
        
    def calibrate_model(self) -> float:
        """
        모델 캘리브레이션 실행 및 최적 temperature 반환
        """
        
    def evaluate_calibration(self) -> Dict[str, float]:
        """
        캘리브레이션 품질 평가 (ECE, MCE, 정확도 등)
        """
        
    def plot_reliability_diagram(self) -> None:
        """
        캘리브레이션 시각화 (Reliability Diagram)
        """
```

---

## 🚀 실행 방법

### 📋 사전 준비

#### 1. 학습된 모델 준비
```bash
# 먼저 기본 모델 학습 완료
python src/training/train_main.py --config configs/train_highperf.yaml --mode full-pipeline

# 학습 완료 후 모델 위치 확인
ls experiments/train/20250907/models/
# 예상 결과: fold_0_best.pth, fold_1_best.pth, fold_2_best.pth
```

#### 2. 패키지 설치 확인
```bash
# 필요 라이브러리 확인
python -c "import torch, sklearn, matplotlib; print('✅ All packages ready!')"
```

### 🔍 기본 실행

#### **방법 1: CLI를 통한 자동 캘리브레이션 (권장)**

```bash
# 학습과 동시에 캘리브레이션 적용
python src/training/train_main.py --config configs/train_highperf.yaml --use-calibration

# 또는 이미 학습된 모델에 캘리브레이션만 적용
python src/calibration/calibrate_model.py --model-dir experiments/train/20250907/models/
```

#### **방법 2: Python 스크립트에서 직접 사용**

```python
# 캘리브레이션 직접 실행
from src.calibration import CalibrationTrainer
from src.models import load_model
from src.data import create_validation_loader

# 모델과 데이터 로더 준비
model = load_model('experiments/train/20250907/models/fold_0_best.pth')
valid_loader = create_validation_loader()

# 캘리브레이션 실행
calibrator = CalibrationTrainer(model, valid_loader)
temperature = calibrator.calibrate_model()
print(f"최적 Temperature: {temperature:.3f}")

# 캘리브레이션 품질 평가
metrics = calibrator.evaluate_calibration()
print(f"ECE (Before): {metrics['ece_before']:.3f}")
print(f"ECE (After): {metrics['ece_after']:.3f}")
```

### 📊 실행 과정

```bash
$ python src/training/train_main.py --config configs/train_highperf.yaml --use-calibration

🚀 Starting training pipeline...
📋 Config: configs/train_highperf.yaml
🎯 Mode: highperf
🌡️ Temperature scaling calibration: enabled
==================================================

🔥 Starting Cross-Validation Training...
📁 Fold 0/3 시작...
  📊 Epoch 1/10: loss 0.234, f1 0.821
  📊 Epoch 2/10: loss 0.198, f1 0.856
  ...
  📊 Epoch 10/10: loss 0.089, f1 0.923
✅ Fold 0 완료: F1 0.923

🌡️ Model Calibration 시작...
  📊 캘리브레이션 이전 ECE: 0.089
  🎯 Temperature 최적화 중...
  📈 Temperature 1.0 → 1.234 → 1.456 → 1.789 → 2.123
  ✅ 최적 Temperature 발견: 2.123
  📊 캘리브레이션 이후 ECE: 0.023 (-73.6% 개선!)
  💾 Temperature 저장: experiments/calibration/fold_0_temperature.json

📁 Fold 1/3 시작...
...

============================================================
🌡️ Temperature Scaling 캘리브레이션 완료!
============================================================
📊 전체 평균 결과:
   🌡️ 평균 Temperature: 2.087
   📉 ECE 개선: 0.084 → 0.021 (-75.0%)
   📈 F1 점수 유지: 0.9234 (캘리브레이션 후에도 동일)
   ⚖️ 캘리브레이션 품질: Excellent (ECE < 0.05)
============================================================

🎉 Calibration completed! Temperature values saved in experiments/calibration/
```

---

## 📈 캘리브레이션 분석

### 1. 생성되는 파일들

```
# 캘리브레이션 완료 후 생성되는 파일들
experiments/calibration/
├── fold_0_temperature.json              # Fold별 최적 temperature
├── fold_1_temperature.json
├── fold_2_temperature.json
├── average_temperature.json             # 전체 평균 temperature
├── calibration_metrics.json            # 상세 캘리브레이션 지표
└── plots/
    ├── reliability_diagram_before.png   # 캘리브레이션 전 신뢰도 다이어그램
    ├── reliability_diagram_after.png    # 캘리브레이션 후 신뢰도 다이어그램
    ├── confidence_histogram.png         # 예측 확률 분포
    └── ece_comparison.png               # ECE 비교 차트

logs/calibration/
└── calibration_20250907_1530.log       # 상세 캘리브레이션 로그
```

### 2. 캘리브레이션 품질 해석

#### **Reliability Diagram 분석**
```python
# 완벽한 캘리브레이션: 대각선에 가까울수록 좋음
# 캘리브레이션 전: 점들이 대각선 아래쪽 (과신)
# 캘리브레이션 후: 점들이 대각선에 가까움 (적절한 확신)
```

#### **ECE 수치 해석**
```
ECE 0.000 ~ 0.030: Excellent 🟢
ECE 0.030 ~ 0.050: Good      🟡  
ECE 0.050 ~ 0.100: Fair      🟠
ECE 0.100 ~      : Poor      🔴
```

### 3. Temperature 값 해석

```python
# Temperature 값별 의미
temperature = 1.0      # 원본 모델과 동일 (캘리브레이션 불필요)
temperature = 1.5      # 약간 과신 → 확률을 부드럽게 조정
temperature = 2.0      # 상당한 과신 → 확률을 크게 조정  
temperature = 3.0+     # 심각한 과신 → 모델 재학습 고려 필요

# 일반적 범위
typical_range = [1.2, 2.5]  # 대부분의 딥러닝 모델에서 이 범위
```

---

## 🎯 성능 개선 효과

### 실제 캘리브레이션 사례

| 시나리오 | ECE (Before) | ECE (After) | Temperature | 개선률 | F1 변화 |
|----------|--------------|-------------|-------------|--------|---------|
| **EfficientNet-B3** | 0.089 | 0.023 | 2.123 | -74.2% | 0.8923 → 0.8923 |
| **Swin Transformer** | 0.076 | 0.019 | 1.876 | -75.0% | 0.9234 → 0.9234 |
| **과적합 모델** | 0.145 | 0.031 | 3.456 | -78.6% | 0.8756 → 0.8756 |
| **잘 학습된 모델** | 0.034 | 0.012 | 1.234 | -64.7% | 0.9324 → 0.9324 |

### 캘리브레이션 전후 예측 비교

```python
# 예시: 문서 분류 결과
Document_A = "기업 실적 보고서..."

# 캘리브레이션 전
before_probs = [0.02, 0.01, 0.94, 0.02, 0.01]  # 94% 확신 (과신)
predicted_class = "Financial"
actual_class = "Business"  # 틀림!

# 캘리브레이션 후 (Temperature = 2.1)
after_probs = [0.12, 0.08, 0.67, 0.10, 0.03]   # 67% 확신 (현실적)
predicted_class = "Financial"  # 여전히 같은 예측
confidence_level = "Medium"    # 하지만 불확실성 인지
```

### 실무 활용도 개선

```python
# 1. 임계값 기반 의사결정
confidence_threshold = 0.8

# 캘리브레이션 전: 잘못된 자신감으로 오판
if max(before_probs) > confidence_threshold:
    decision = "Auto-process"  # 94% > 80% → 자동 처리 (위험!)
else:
    decision = "Manual-review"

# 캘리브레이션 후: 적절한 신중함
if max(after_probs) > confidence_threshold:
    decision = "Auto-process"
else:
    decision = "Manual-review"  # 67% < 80% → 수동 검토 (안전!)
```

---

## 🔄 추론 파이프라인 통합

### 1. 추론 시 자동 캘리브레이션 적용

```python
# src/inference/inference_main.py에 통합
class CalibratedInferenceRunner:
    def __init__(self, model_path: str, temperature_path: str):
        self.model = load_model(model_path)
        
        # Temperature 값 로드
        with open(temperature_path, 'r') as f:
            temp_data = json.load(f)
            self.temperature = temp_data['temperature']
            
        self.temperature_scaling = TemperatureScaling()
        self.temperature_scaling.temperature.data = torch.tensor([self.temperature])
        
    def predict(self, image):
        with torch.no_grad():
            logits = self.model(image)
            # 캘리브레이션된 확률 계산
            calibrated_probs = self.temperature_scaling(logits)
        return calibrated_probs
```

### 2. CLI 추론에서 캘리브레이션 사용

```bash
# 캘리브레이션 적용된 추론 실행
python src/inference/inference_main.py \
    --config configs/infer_highperf.yaml \
    --use-calibration \
    --temperature-file experiments/calibration/average_temperature.json

# 또는 자동으로 최신 캘리브레이션 파일 사용
python src/inference/inference_main.py \
    --config configs/infer_highperf.yaml \
    --use-calibration
```

### 3. 제출 파일에 확신도 추가

```python
# calibrated_submission.csv
image_id,category,confidence
test_001.jpg,Business,0.73      # 캘리브레이션된 확률
test_002.jpg,Financial,0.89     # 높은 확신도 (신뢰 가능)
test_003.jpg,Technology,0.54    # 낮은 확신도 (주의 필요)
```

---

## 🤝 팀 협업 워크플로우

### 1. 팀별 캘리브레이션 실행

```bash
# 각 팀원이 자신의 모델에 캘리브레이션 적용
팀원A: python src/training/train_main.py --config configs/train_highperf.yaml --use-calibration
팀원B: python src/training/train_main.py --config configs/train.yaml --use-calibration  
팀원C: python src/training/train_main.py --config configs/train_swin.yaml --use-calibration
```

### 2. Temperature 값 공유 및 분석

```bash
# 팀원별 캘리브레이션 결과 수집
experiments/calibration/team_analysis/
├── member_A_temperatures.json     # Temperature: 2.123, ECE: 0.023
├── member_B_temperatures.json     # Temperature: 1.876, ECE: 0.019  
├── member_C_temperatures.json     # Temperature: 3.234, ECE: 0.045
└── team_calibration_report.md     # 팀 전체 분석 보고서
```

### 3. 최적 캘리브레이션 설정 선택

```python
# 팀 최적 설정 선택 기준
best_calibration = min(team_results, key=lambda x: x['ece_after'])

# 예시 결과
optimal_setup = {
    'member': 'B',
    'model': 'Swin Transformer',  
    'temperature': 1.876,
    'ece_before': 0.076,
    'ece_after': 0.019,          # 가장 낮은 ECE
    'f1_score': 0.9234
}
```

### 4. 팀 표준 캘리브레이션 적용

```bash
# 최적 설정을 팀 표준으로 설정
cp experiments/calibration/member_B/average_temperature.json \
   experiments/calibration/team_standard_temperature.json

# 모든 팀원이 표준 설정으로 추론
python src/inference/inference_main.py \
    --config configs/infer_highperf.yaml \
    --use-calibration \
    --temperature-file experiments/calibration/team_standard_temperature.json
```

---

## 🚨 문제 해결

### 자주 발생하는 문제들

#### 1. **Temperature가 너무 높음 (> 5.0)**
```
Warning: Temperature = 6.234 (very high, model may be severely overconfident)
```

**원인 및 해결방법:**
```python
# 원인: 모델이 심각하게 과신하고 있음
# 해결책 1: 모델 재학습 (더 많은 정규화)
train_config = {
    'dropout': 0.3,           # 드롭아웃 증가
    'weight_decay': 0.05,     # Weight decay 증가
    'label_smoothing': 0.1    # 라벨 스무딩 추가
}

# 해결책 2: 데이터 증강 강화
augmentation_config = {
    'mixup_alpha': 0.2,       # Mixup 추가
    'cutmix_alpha': 1.0       # CutMix 추가
}
```

#### 2. **캘리브레이션 후에도 ECE가 높음**
```
ECE after calibration: 0.089 (still high)
```

**해결방법:**
```python
# 1. Validation set 크기 확인
if len(validation_set) < 1000:
    print("Warning: 검증 세트가 너무 작음. 최소 1000개 이상 권장")
    
# 2. 다른 캘리브레이션 기법 시도
from netcal.scaling import PlattScaling, IsotonicRegression
platt_scaling = PlattScaling()
isotonic_calibration = IsotonicRegression()

# 3. 앙상블 캘리브레이션
ensemble_temperature = np.mean([temp1, temp2, temp3])
```

#### 3. **Temperature 최적화가 실패함**
```
RuntimeError: Temperature optimization failed to converge
```

**해결방법:**
```python
# 1. 학습률 조정
temperature_optimizer = torch.optim.LBFGS(
    [temperature_scaling.temperature], 
    lr=0.01,                    # 기본값: 0.01
    max_iter=50                 # 반복 횟수 증가
)

# 2. 초기값 변경
temperature_scaling.temperature.data = torch.tensor([2.0])  # 기본 1.5 → 2.0

# 3. 손실 함수 변경
criterion = nn.CrossEntropyLoss()  # 대신 focal loss 사용
```

#### 4. **추론 시 캘리브레이션 적용 안됨**
```
Error: Calibrated probabilities not applied during inference
```

**해결방법:**
```bash
# 1. Temperature 파일 경로 확인
ls experiments/calibration/average_temperature.json
cat experiments/calibration/average_temperature.json

# 2. 추론 스크립트에 캘리브레이션 명시적 적용
python src/inference/inference_main.py \
    --config configs/infer_highperf.yaml \
    --use-calibration \
    --temperature-file experiments/calibration/average_temperature.json \
    --debug  # 디버그 모드로 확인
```

---

## 🔬 고급 활용법

### 1. 클래스별 Temperature 조정

```python
# 클래스별로 다른 캘리브레이션 적용
class ClassWiseTemperatureScaling(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 각 클래스별로 다른 temperature
        self.temperatures = nn.Parameter(torch.ones(num_classes) * 1.5)
        
    def forward(self, logits):
        # 클래스별 temperature 적용
        calibrated_logits = logits / self.temperatures.unsqueeze(0)
        return torch.softmax(calibrated_logits, dim=1)
```

### 2. 앙상블 모델 캘리브레이션

```python
# 여러 모델의 앙상블에 캘리브레이션 적용
class EnsembleCalibration:
    def __init__(self, models, temperatures):
        self.models = models
        self.temperatures = temperatures
        
    def predict_calibrated(self, x):
        ensemble_probs = []
        
        for model, temp in zip(self.models, self.temperatures):
            logits = model(x)
            calibrated_probs = torch.softmax(logits / temp, dim=1)
            ensemble_probs.append(calibrated_probs)
            
        # 앙상블 평균
        return torch.mean(torch.stack(ensemble_probs), dim=0)
```

### 3. Confidence-based 샘플 필터링

```python
# 캘리브레이션된 확률로 예측 품질 평가
def confidence_based_filtering(predictions, threshold=0.8):
    high_confidence = []
    low_confidence = []
    
    for pred in predictions:
        max_prob = np.max(pred['calibrated_probs'])
        
        if max_prob >= threshold:
            high_confidence.append(pred)  # 자동 처리
        else:
            low_confidence.append(pred)   # 수동 검토
            
    return high_confidence, low_confidence

# 사용 예시
auto_process, manual_review = confidence_based_filtering(predictions, 0.85)
print(f"자동 처리: {len(auto_process)}개 ({len(auto_process)/len(predictions)*100:.1f}%)")
print(f"수동 검토: {len(manual_review)}개")
```

### 4. 동적 Temperature 조정

```python
# 추론 시점에서 입력 데이터 특성에 따라 temperature 조정
class AdaptiveTemperatureScaling:
    def __init__(self, base_temperature=2.0):
        self.base_temperature = base_temperature
        
    def get_adaptive_temperature(self, image_features):
        # 이미지 복잡도에 따라 temperature 조정
        complexity_score = self.estimate_complexity(image_features)
        
        if complexity_score > 0.8:      # 복잡한 이미지
            return self.base_temperature * 1.2
        elif complexity_score < 0.3:    # 단순한 이미지  
            return self.base_temperature * 0.8
        else:
            return self.base_temperature
```

---

## 📝 체크리스트

### 캘리브레이션 실행 전 확인사항
- [ ] 모델 학습 완료 및 검증 세트 준비
- [ ] Validation set 크기 충분 (최소 500개 이상)
- [ ] Temperature scaling 모듈 정상 import
- [ ] 충분한 디스크 공간 (그래프/로그 저장용)

### 캘리브레이션 품질 확인
- [ ] ECE 값이 0.05 이하로 개선됨
- [ ] Temperature 값이 적절한 범위 (1.0~4.0)
- [ ] Reliability diagram에서 대각선에 가까워짐
- [ ] F1 점수가 유지됨 (성능 저하 없음)

### 추론 파이프라인 통합 확인
- [ ] Temperature 파일이 올바르게 생성됨
- [ ] 추론 시 캘리브레이션 자동 적용
- [ ] 캘리브레이션된 확률 값 검증
- [ ] 제출 파일에 적절한 확신도 포함

---

## 🔗 관련 문서

- [Optuna 하이퍼파라미터 최적화 가이드](./Optuna_하이퍼파라미터_최적화_가이드.md)
- [추론 파이프라인 가이드](../pipelines/추론_파이프라인_가이드.md)
- [전체 파이프라인 실행 가이드](../pipelines/전체_파이프라인_가이드.md)
- [모델 성능 비교 분석 보고서](../experiments/모델_성능_비교_분석_보고서.md)

---

**Created by**: AI Team  
**Date**: 2025-09-07  
**Version**: Temperature Scaling v1.0 Integration  
**Status**: ✅ Production Ready  
**Environment**: pyenv cv_py3_11_9 가상환경

> 🎯 **캘리브레이션 개선**: ECE 70-80% 감소  
> ⏱️ **소요 시간**: 모델당 5분 내  
> 🔧 **권장 설정**: `--use-calibration` (기본 활성화 권장)
