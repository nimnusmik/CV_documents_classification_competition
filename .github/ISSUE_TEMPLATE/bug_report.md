---
name: 🐛 버그 리포트
about: 버그나 오류를 신고할 때 사용
title: '[BUG] 간단한 버그 설명'
labels: ['type: bug', 'priority: high']
assignees: ''
---

## 🚨 버그 요약
**버그 한줄 설명:** GPU 메모리 부족 오류로 학습 중단

**발생 위치:** 
- [ ] 데이터 로딩
- [ ] 모델 학습  
- [ ] 추론/예측
- [ ] 제출 파일 생성
- [ ] 기타: ___________

**심각도:**
- [ ] 🔴 Critical (작업 완전 중단)
- [ ] 🟡 High (주요 기능 영향)
- [ ] 🟢 Medium (일부 기능 영향)
- [ ] ⚪ Low (사소한 문제)

## 🔍 버그 상세 설명
**무엇이 잘못되었나요?**
GPU 메모리가 부족해서 배치 크기 16으로 학습할 때 OOM 에러가 발생합니다.

**언제 발생하나요?**
- EfficientNet-B4 모델 학습 시작 후 2-3 에포크 진행 중

## 📝 재현 방법
**재현 단계:**
1. `python train.py --model efficientnet_b4 --batch_size 16` 실행
2. 학습 2-3 에포크 진행 대기
3. GPU 메모리 모니터링
4. OOM 에러 발생 확인

**사용한 명령어:**
```bash
python train.py --model efficientnet_b4 --batch_size 16 --epochs 30
