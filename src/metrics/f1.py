# PyTorch 및 scikit-learn 메트릭 유틸 임포트
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ---------------------- Macro F1 계산 ---------------------- #
def macro_f1_from_logits(logits, targets):
    """
    로짓(logits) 출력으로부터 macro-F1 점수를 계산
    - logits: 모델 출력 (배치 x 클래스 수)
    - targets: 정답 라벨 텐서
    """

    # argmax로 예측 클래스 산출 (Tensor → numpy)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

    # 정답 라벨을 numpy 배열로 변환
    t = targets.cpu().numpy()

    # macro-F1 점수 계산 (클래스별 평균 F1)
    return f1_score(t, preds, average="macro")


# ---------------------- 상세 리포트 생성 ---------------------- #
def report_from_logits(logits, targets):
    """
    로짓(logits) 출력으로부터 분류 리포트와 혼동행렬을 반환
    - logits: 모델 출력 (배치 x 클래스 수)
    - targets: 정답 라벨 텐서
    """

    # argmax로 예측 클래스 산출 (Tensor → numpy)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

    # 정답 라벨을 numpy 배열로 변환
    t = targets.cpu().numpy()

    # classification_report와 confusion_matrix 생성 후 dict 반환
    return {
        "report": classification_report(
            t,                      # 정답 라벨
            preds,                  # 예측 결과
            output_dict=True,       # JSON/dict 형태 출력
            zero_division=0         # 0으로 나눔 발생 시 0으로 처리
        ),
        "confusion_matrix": confusion_matrix(t, preds).tolist()  # numpy → list 변환
    }