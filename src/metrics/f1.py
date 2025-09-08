# ------------------------- F1 점수 계산 모듈 ------------------------- #
# PyTorch 및 scikit-learn 메트릭 유틸 임포트
import torch                                                                    # PyTorch 텐서 연산
from sklearn.metrics import f1_score, classification_report, confusion_matrix   # scikit-learn 메트릭 함수들


# ---------------------- Macro F1 계산 ---------------------- #
def macro_f1_from_logits(logits, targets):        # 매크로 F1 점수 계산 함수
    """
    로짓(logits) 출력으로부터 macro-F1 점수를 계산
    - logits: 모델 출력 (배치 x 클래스 수)
    - targets: 정답 라벨 텐서
    """

    # argmax로 예측 클래스 산출 (Tensor → numpy)
    preds = torch.argmax(logits, dim=1).cpu().numpy()  # 가장 높은 확률의 클래스 선택

    # 정답 라벨을 numpy 배열로 변환
    t = targets.cpu().numpy()                     # GPU 텐서를 CPU numpy로 변환

    # macro-F1 점수 계산 (클래스별 평균 F1)
    return f1_score(t, preds, average="macro")    # 클래스별 F1 점수의 평균 반환


# ---------------------- 상세 리포트 생성 ---------------------- #
def report_from_logits(logits, targets):          # 분류 리포트 생성 함수
    """
    로짓(logits) 출력으로부터 분류 리포트와 혼동행렬을 반환
    - logits: 모델 출력 (배치 x 클래스 수)
    - targets: 정답 라벨 텐서
    """

    # argmax로 예측 클래스 산출 (Tensor → numpy)
    preds = torch.argmax(logits, dim=1).cpu().numpy()  # 가장 높은 확률의 클래스 선택

    # 정답 라벨을 numpy 배열로 변환
    t = targets.cpu().numpy()                     # GPU 텐서를 CPU numpy로 변환

    # classification_report와 confusion_matrix 생성 후 dict 반환
    return {                                      # 딕셔너리 형태로 결과 반환
        "report": classification_report(          # 클래스별 정밀도/재현율/F1 리포트
            t,                      # 정답 라벨 배열
            preds,                  # 예측 결과 배열
            output_dict=True,       # JSON/dict 형태 출력 지정
            zero_division=0         # 0으로 나눔 발생 시 0으로 처리
        ),
        "confusion_matrix": confusion_matrix(t, preds).tolist()  # 혼동행렬 (numpy → list 변환)
    }