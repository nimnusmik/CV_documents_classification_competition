#!/bin/bash
"""
실험 폴더 정리 및 시각화 생성 스크립트

사용법:
  ./scripts/setup_visualization.sh [옵션]

옵션:
  --dry-run    : 실제 변경 없이 계획만 출력
  --viz-only   : 폴더 재구성 없이 시각화만 생성
  --help       : 도움말 출력
"""

set -e

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  실험 결과 시각화 시스템 설정${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "실험 폴더 정리 및 시각화 생성 스크립트"
    echo ""
    echo "사용법:"
    echo "  ./scripts/setup_visualization.sh [옵션]"
    echo ""
    echo "옵션:"
    echo "  --dry-run    실제 변경 없이 계획만 출력"
    echo "  --viz-only   폴더 재구성 없이 시각화만 생성"
    echo "  --help       이 도움말 출력"
    echo ""
    echo "예시:"
    echo "  ./scripts/setup_visualization.sh --dry-run"
    echo "  ./scripts/setup_visualization.sh --viz-only"
    echo "  ./scripts/setup_visualization.sh"
}

# 인자 파싱
DRY_RUN=false
VIZ_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --viz-only)
            VIZ_ONLY=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 메인 실행
main() {
    print_header
    
    # Python 환경 확인
    print_step "Python 환경 확인..."
    if ! command -v python &> /dev/null; then
        print_error "Python이 설치되지 않았습니다."
        exit 1
    fi
    
    # 필요한 패키지 확인
    print_step "필요한 패키지 확인..."
    python -c "import matplotlib, seaborn, pandas, numpy" 2>/dev/null || {
        print_warning "시각화 패키지가 설치되지 않았을 수 있습니다."
        echo "다음 명령어로 설치하세요:"
        echo "pip install matplotlib seaborn pandas numpy"
    }
    
    # experiments 폴더 확인
    if [ ! -d "experiments" ]; then
        print_warning "experiments 폴더가 존재하지 않습니다. 생성합니다..."
        mkdir -p experiments/{train,infer,optimization}
    fi
    
    # 폴더 재구성 실행
    if [ "$VIZ_ONLY" = false ]; then
        print_step "실험 폴더 재구성..."
        
        if [ "$DRY_RUN" = true ]; then
            python scripts/reorganize_experiments.py --dry-run
        else
            python scripts/reorganize_experiments.py
        fi
    fi
    
    # 시각화 생성
    if [ "$DRY_RUN" = false ]; then
        print_step "기존 결과 시각화 생성..."
        python scripts/reorganize_experiments.py --create-viz
    fi
    
    print_step "완료!"
    echo ""
    echo -e "${GREEN}✅ 시각화 시스템이 설정되었습니다!${NC}"
    echo ""
    echo "새로운 폴더 구조:"
    echo "experiments/"
    echo "├── train/YYYYMMDD/model_name/images/"
    echo "├── infer/YYYYMMDD/model_name/images/"
    echo "└── optimization/YYYYMMDD/model_name/images/"
    echo ""
    echo "앞으로 모든 학습/추론/최적화 결과에 자동으로 시각화가 생성됩니다."
}

main "$@"
