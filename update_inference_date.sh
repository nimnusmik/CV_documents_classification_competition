#!/bin/bash
# update_inference_date.sh
# 추론 설정 파일의 날짜를 빠르게 업데이트하는 간단한 스크립트

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 추론 설정 날짜 업데이트 유틸리티${NC}"
echo "========================================"

# 인자 파싱
TARGET_DATE=""
if [ "$1" == "--latest" ]; then
    # 가장 최신 실험 날짜 찾기
    if [ -d "experiments/train" ]; then
        TARGET_DATE=$(ls experiments/train/ | grep -E "^[0-9]{8}$" | sort | tail -1)
        if [ -z "$TARGET_DATE" ]; then
            echo -e "${RED}❌ experiments/train에서 날짜 디렉터리를 찾을 수 없습니다.${NC}"
            exit 1
        fi
        echo -e "${GREEN}📅 가장 최신 날짜: $TARGET_DATE${NC}"
    else
        echo -e "${RED}❌ experiments/train 디렉터리가 존재하지 않습니다.${NC}"
        exit 1
    fi
elif [ -n "$1" ]; then
    # 사용자가 지정한 날짜 사용
    if [[ $1 =~ ^[0-9]{8}$ ]]; then
        TARGET_DATE="$1"
        echo -e "${GREEN}📅 지정된 날짜: $TARGET_DATE${NC}"
    else
        echo -e "${RED}❌ 날짜는 YYYYMMDD 형식이어야 합니다 (예: 20250908)${NC}"
        exit 1
    fi
else
    # 오늘 날짜 사용
    TARGET_DATE=$(date +%Y%m%d)
    echo -e "${GREEN}📅 오늘 날짜 사용: $TARGET_DATE${NC}"
fi

# 해당 날짜의 실험 디렉터리 확인
EXPERIMENT_DIR="experiments/train/$TARGET_DATE"
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo -e "${RED}❌ 실험 디렉터리가 존재하지 않습니다: $EXPERIMENT_DIR${NC}"
    echo -e "${YELLOW}💡 사용 가능한 날짜들:${NC}"
    ls experiments/train/ | grep -E "^[0-9]{8}$" | sort
    exit 1
fi

# 모델 폴더 찾기
EFFICIENTNET_DIR=""
SWIN_DIR=""

for dir in "$EXPERIMENT_DIR"/*/; do
    dirname=$(basename "$dir")
    if [[ $dirname == *"efficientnet"* ]]; then
        EFFICIENTNET_DIR="$dirname"
    elif [[ $dirname == *"swin"* ]]; then
        SWIN_DIR="$dirname"
    fi
done

echo -e "${BLUE}📂 발견된 모델 실험들:${NC}"
[ -n "$EFFICIENTNET_DIR" ] && echo "   - EfficientNet: $EFFICIENTNET_DIR"
[ -n "$SWIN_DIR" ] && echo "   - Swin: $SWIN_DIR"

# 백업 생성 함수
backup_file() {
    local file="$1"
    if [ -f "$file" ]; then
        cp "$file" "$file.backup.$(date +%Y%m%d_%H%M%S)"
        echo -e "${YELLOW}💾 백업 생성: $file.backup.$(date +%Y%m%d_%H%M%S)${NC}"
    fi
}

# infer.yaml 업데이트
echo -e "\n${BLUE}🔧 configs/infer.yaml 업데이트 중...${NC}"
if [ -f "configs/infer.yaml" ] && [ -n "$EFFICIENTNET_DIR" ]; then
    backup_file "configs/infer.yaml"
    
    # 날짜와 폴더명 업데이트
    sed -i.tmp "s|experiments/train/[0-9]\{8\}/[^/]*/ckpt|experiments/train/$TARGET_DATE/$EFFICIENTNET_DIR/ckpt|g" configs/infer.yaml
    rm configs/infer.yaml.tmp
    
    echo -e "${GREEN}✅ infer.yaml 업데이트 완료${NC}"
else
    echo -e "${YELLOW}⚠️  infer.yaml 또는 EfficientNet 실험을 찾을 수 없습니다${NC}"
fi

# infer_highperf.yaml 업데이트
echo -e "\n${BLUE}🔧 configs/infer_highperf.yaml 업데이트 중...${NC}"
if [ -f "configs/infer_highperf.yaml" ] && [ -n "$SWIN_DIR" ]; then
    backup_file "configs/infer_highperf.yaml"
    
    # 날짜와 폴더명 업데이트
    sed -i.tmp "s|experiments/train/[0-9]\{8\}/[^/]*/fold_results.yaml|experiments/train/$TARGET_DATE/$SWIN_DIR/fold_results.yaml|g" configs/infer_highperf.yaml
    rm configs/infer_highperf.yaml.tmp
    
    echo -e "${GREEN}✅ infer_highperf.yaml 업데이트 완료${NC}"
else
    echo -e "${YELLOW}⚠️  infer_highperf.yaml 또는 Swin 실험을 찾을 수 없습니다${NC}"
fi

echo -e "\n${GREEN}✅ 업데이트 완료!${NC}"
echo -e "\n${BLUE}🚀 이제 다음 명령어로 추론을 실행할 수 있습니다:${NC}"
echo "   # EfficientNet 추론"
echo "   python src/inference/infer_main.py --config configs/infer.yaml --mode basic"
echo ""
echo "   # Swin 추론"
echo "   python src/inference/infer_main.py --config configs/infer_highperf.yaml --mode highperf"
