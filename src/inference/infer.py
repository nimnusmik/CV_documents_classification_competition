# 라이브러리 import
import os, time, numpy as np, torch, pandas as pd         # 경로/시간/수치계산/딥러닝/데이터 처리
from torch.utils.data import DataLoader                  # PyTorch DataLoader
from tqdm import tqdm                                    # 진행률 표시
import torchvision.transforms.functional as TF           # torchvision 이미지 변환 함수
import PIL.Image as Image                                # 이미지 처리

# 프로젝트 내부 유틸 import
from src.logging.logger import Logger                      # 로그 기록 유틸
from src.utils import load_yaml, ensure_dir, resolve_path, require_file, require_dir  # 핵심 유틸
from src.data.dataset import DocClsDataset               # 데이터셋 클래스
from src.data.transforms import build_valid_tfms         # 검증용 변환 파이프라인
from src.models.build import build_model                 # 모델 빌드 함수

# ---------------------- 로거 생성 함수 ---------------------- #
def _make_logger(cfg):
    logs_dir = ensure_dir(cfg["output"]["logs_dir"])     # 로그 디렉토리 생성 보장
    log_name = f"infer_{time.strftime('%Y%m%d-%H%M')}_{cfg['project']['run_name']}.log"
    log_path = os.path.join(logs_dir, log_name)  # 로그 파일 경로
    logger = Logger(log_path)                            # Logger 객체 생성
    logger.start_redirect()                              # 표준 출력 리다이렉트
    if hasattr(logger, "tqdm_redirect"):                 # tqdm 리다이렉트 지원 여부 확인
        logger.tqdm_redirect()
    logger.write(f">> Inference logger: {log_path}")     # 로거 시작 로그 출력
    return logger                                        # 로거 반환

# ---------------------- 텐서 회전(TTA용) ---------------------- #
def _rotate_tensor(x, deg):
    imgs = []                                            # 회전된 이미지 담을 리스트
    for i in range(x.size(0)):                           # 배치 내 각 이미지 반복
        pil = TF.to_pil_image(x[i].cpu())                # 텐서를 PIL 이미지로 변환
        pil = pil.rotate(deg, resample=Image.BILINEAR)   # 지정 각도로 회전
        imgs.append(TF.to_tensor(pil).to(x.device))      # 다시 텐서로 변환 후 디바이스에 올림
    return torch.stack(imgs, 0)                          # 배치 텐서로 결합하여 반환

# ---------------------- 추론 실행 함수 ---------------------- #
@torch.no_grad()                                         # 추론 중 그래디언트 계산 비활성화
def run_inference(cfg_path: str, out: str|None=None, ckpt: str|None=None):
    cfg = load_yaml(cfg_path)                            # YAML 설정 로드
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path)) # 설정 파일 디렉토리 절대경로
    logger = _make_logger(cfg)                           # 로거 생성

    logger.write("[BOOT] inference pipeline started")    # 파이프라인 시작 로그
    exit_status = "SUCCESS"                              # 종료 상태 기본값
    exit_code = 0                                        # 종료 코드 기본값
    
    try:
        # ---------------------- 경로 해석 ---------------------- #
        # 샘플 CSV 절대경로
        sample_csv = resolve_path(cfg_dir, cfg["data"]["sample_csv"])
        # 테스트 이미지 경로
        image_dir  = resolve_path(cfg_dir, cfg["data"].get("image_dir_test", cfg["data"].get("image_dir", "data/raw/test")))
        require_file(sample_csv, "data.sample_csv 확인")    # CSV 파일 존재 확인
        require_dir(image_dir,  "data.image_dir_test 확인") # 디렉토리 존재 확인
        # 경로 로그
        logger.write(f"[PATH] OK | sample_csv={sample_csv} | image_dir_test={image_dir}")

        # ---------------------- 설정 출력 ---------------------- #
        logger.write(f"[CFG] data={cfg['data']}")           # 데이터 설정 로그
        logger.write(f"[CFG] model={cfg['model']}")         # 모델 설정 로그
        logger.write(f"[CFG] inference={cfg['inference']}") # 추론 설정 로그

        # ---------------------- 데이터 준비 ---------------------- #
        # 제출용 샘플 CSV 로드
        df_sub  = pd.read_csv(sample_csv)                

        # 테스트 ID 컬럼만 추출하여 별도 DataFrame 생성
        test_df = df_sub[[cfg["data"]["id_col"]]].copy() 

        # 데이터 크기와 상위 3개 샘플 로그 기록
        logger.write(
            f"[DATA] test size={len(test_df)} | head={test_df.head(3).to_dict(orient='records')}"
        )  

        # 테스트 데이터셋(DocClsDataset) 생성
        ds = DocClsDataset(                              
            test_df,                                     # 추출된 테스트 ID DataFrame
            image_dir,                                   # 테스트 이미지 디렉터리
            cfg["data"]["image_ext"],                    # 이미지 확장자 설정
            cfg["data"]["id_col"],                       # ID 컬럼명
            target_col=None,                             # 테스트 데이터이므로 라벨 없음
            transform=build_valid_tfms(cfg["train"]["img_size"])  # 검증용 변환 파이프라인
        )

        # DataLoader 생성 (배치 단위 로딩)
        ld = DataLoader(
            ds,                                          # 위에서 정의한 Dataset
            batch_size=cfg["train"]["batch_size"],       # 배치 크기
            shuffle=False,                               # 순서 유지 (shuffle 비활성)
            num_workers=cfg["project"]["num_workers"],   # 멀티프로세스 로딩
            pin_memory=True                              # CUDA 전송 최적화
        )

        # DataLoader 상태 로그 기록 (스텝 수, 배치 크기)
        logger.write(
            f"[DATA] dataloader built | steps={len(ld)} bs={cfg['train']['batch_size']}"
        )

        # ---------------------- 모델 준비 ---------------------- #
        # 디바이스 선택
        device = "cuda" if (cfg["project"]["device"]=="cuda" and torch.cuda.is_available()) else "cpu"
        # 모델 빌드 후 eval 모드
        model = build_model(cfg["model"]["name"], cfg["data"]["num_classes"], cfg["model"]["pretrained"]).to(device).eval()

        # ---------------------- 체크포인트 로드 ---------------------- #
        # ckpt 인자가 직접 지정된 경우
        if ckpt:
            # 사용자 입력 ckpt 경로를 config 기준 절대경로로 변환
            ckpt_path = resolve_path(cfg_dir, ckpt)

        # ckpt 인자가 없는 경우 - config 파일의 ckpt.path 확인
        elif "ckpt" in cfg and "path" in cfg["ckpt"]:
            # config 파일의 ckpt.path 설정 사용
            ckpt_path = resolve_path(cfg_dir, cfg["ckpt"]["path"])
            logger.write(f"[CKPT] Using config ckpt.path: {cfg['ckpt']['path']}")

        # config에도 ckpt.path가 없는 경우 (기본 best_fold0.pth 사용)
        else:
            # 학습 결과 디렉터리 패턴 검색
            import glob
            day = time.strftime(cfg["project"]["date_format"])      # 날짜 문자열
            run_name = cfg['project']['run_name']                   # 실행 이름
            
            # 패턴: exp_dir/날짜/run_name_날짜_시간/ckpt/best_fold0.pth
            pattern = resolve_path(cfg_dir, os.path.join(
                cfg["output"]["exp_dir"],                           # 실험 결과 루트 디렉터리
                day,                                                # 날짜 폴더
                f"{run_name}_{day}_*",                             # run_name_날짜_시간 패턴
                "ckpt",                                             # 체크포인트 저장 디렉터리
                "best_fold0.pth"                                    # 기본 체크포인트 파일명
            ))
            
            # 패턴에 맞는 파일 검색
            matching_files = glob.glob(pattern)
            if matching_files:
                # 가장 최근 파일 선택 (시간순 정렬)
                ckpt_path = sorted(matching_files)[-1]
            else:
                # 패턴 매칭 실패 시 기존 방식으로 fallback
                ckpt_path = resolve_path(cfg_dir, os.path.join(
                    cfg["output"]["exp_dir"],                       # 실험 결과 루트 디렉터리
                    day,                                            # 날짜 형식에 맞춘 하위 폴더
                    f"{run_name}",                                  # 실행 이름(run_name) 폴더 (기존 방식)
                    "ckpt",                                         # 체크포인트 저장 디렉터리
                    "best_fold0.pth"                                # 기본 체크포인트 파일명
                ))
            
        # ckpt 존재 확인
        require_file(ckpt_path, "--ckpt로 직접 지정하거나 학습 결과 경로 확인")
        state = torch.load(ckpt_path, map_location=device)          # 체크포인트 로드
        
        # 체크포인트 구조에 따라 모델 가중치 로드
        if "model" in state:
            model.load_state_dict(state["model"], strict=True)      # 구형 체크포인트 형식
        elif "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=True)  # 신형 체크포인트 형식
        else:
            model.load_state_dict(state, strict=True)               # state_dict 직접 저장된 경우
        logger.write(f"[CKPT] loaded: {ckpt_path}")                 # 로드 로그

        # ---------------------- TTA 설정 ---------------------- #
        # TTA 회전 각도 목록
        degs = cfg["inference"]["tta_rot_degrees"] if cfg["inference"]["tta"] else [0]
        # TTA 설정 로그
        logger.write(f"[TTA] enabled={cfg['inference']['tta']} degs={degs}")

        # ---------------------- 추론 루프 ---------------------- #
        logits_all = []                     # 전체 결과 저장 리스트
        logger.write("[INFER] >>> start")   # 추론 시작 로그
        
        # DataLoader 반복
        for step, (imgs, ids) in enumerate(tqdm(ld, desc="infer"), 1):
            imgs = imgs.to(device)  # 이미지를 디바이스로 이동
            probs_accum = None      # 누적 확률 초기화
            
            # 각 TTA 각도 반복
            for d in degs:
                x = imgs if d==0 else _rotate_tensor(imgs, d)   # 회전 적용
                logits = model(x)                               # 모델 추론
                probs  = torch.softmax(logits, dim=1)           # 확률 변환
                # 확률 누적
                probs_accum = probs if probs_accum is None else probs_accum + probs
                
            # probs_accum이 None이 아님을 보장
            if probs_accum is not None:
                probs = (probs_accum / len(degs)).cpu().numpy()     # 평균 확률 계산
            else:
                # 백업: probs_accum이 None인 경우 (에러 방지)
                probs = torch.zeros((imgs.size(0), 17)).cpu().numpy()
            logits_all.append(probs)                            # 결과 저장
            
            # 주기적으로 로그
            if step == 1 or (step % 20) == 0 or step == len(ld):
                # 현재 진행 상황 로그
                logger.write(f"[INFER] step {step}/{len(ld)} processed")

        probs = np.concatenate(logits_all, axis=0)       # 결과 결합
        preds = probs.argmax(axis=1)                     # 예측 클래스 산출

        # ---------------------- 결과 저장 ---------------------- #
        # 동적 파일명 생성 (날짜_모델명 형식)
        if out is None:
            current_date = pd.Timestamp.now().strftime('%Y%m%d')
            current_time = pd.Timestamp.now().strftime('%H%M')
            model_name = cfg["model"]["name"]
            tta_suffix = "_tta" if cfg.get("inference", {}).get("tta", False) else ""
            
            # 증강 타입 결정 (학습 설정과 동일한 로직 사용)
            aug_type = "advanced_augmentation" if cfg["train"].get("use_advanced_augmentation", False) else "basic_augmentation"
            
            filename = f"{current_date}_{current_time}_{model_name}{tta_suffix}_{aug_type}.csv"
            out_path = f"submissions/{current_date}/{filename}"
        else:
            out_path = resolve_path(cfg_dir, out)
            
        # 디렉토리 보장
        ensure_dir(os.path.dirname(out_path))
        
        # 제출 DataFrame 생성
        sub = pd.DataFrame({
            cfg["data"]["id_col"]: test_df[cfg["data"]["id_col"]].values,   # ID 열
            cfg["data"]["target_col"]: preds                                # 예측 열
        })
        
        sub.to_csv(out_path, index=False)   # CSV 저장
        logger.write(f"[OUT] submission saved: {out_path} | shape={sub.shape}") # CSV 저장 로그

        # 추론 완료 로그
        logger.write("[INFER] <<< finished successfully")

    except Exception as e:                                                  # 예외 처리
        exit_status = "ERROR"                                               # 상태 ERROR
        exit_code = 1                                                       # 종료 코드 1
        logger.write(f"[ERROR] {type(e).__name__}: {e}", print_error=True)  # 에러 로그 기록
        raise                                                               # 예외 재발생
    finally:
        logger.write(f"[EXIT] INFERENCE {exit_status} code={exit_code}")    # 종료 상태 로그
        logger.write(">> Stopping logger and restoring stdio")              # 로거 종료 로그
        logger.stop_redirect()                                              # 출력 리다이렉트 해제
        logger.close()                                                      # 로거 닫기