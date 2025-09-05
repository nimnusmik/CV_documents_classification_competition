
 팀원들을 위한 WandB 사용 가이드:

1. WandB 계정 생성 및 로그인:
   - https://wandb.ai 에서 계정 생성
   - 터미널에서: wandb login

2. 같은 프로젝트에 실험 추가:
   - PROJECT_NAME = "document-classification-team"
   - 각자 다른 실험명 사용 (예: "member1-experiment")
   
3. 실험 태그 규칙:
   - [멤버명, 실험타입, 모델명, 기타] 형식
   - 예: ["john", "hyperparameter-tuning", "efficientnet-b4"]

4. 이 베이스라인 활용:
   - model_name, LR, BATCH_SIZE 등 하이퍼파라미터 변경
   - config 딕셔너리에 새로운 설정 추가
   - wandb.init()의 name과 tags 수정

5. 팀 대시보드 확인:
   - https://wandb.ai/kimsunmin0227-hufs/document-classification-team/runs/jewojmhb
   
현재 베이스라인 성능:
   - CV F1 Score: 0.9015 ± 0.0027
   - 이 성능을 기준으로 개선 실험 진행!
