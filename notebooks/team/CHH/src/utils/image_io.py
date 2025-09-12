from __future__ import annotations                  # 최신 타입 힌트(전방 참조) 사용
import cv2, hashlib, os, time                       # OpenCV, 해시, 경로/파일, 지연 처리
import numpy as np                                  # 수치 배열 처리

# ---------------------- MD5 해시 계산 ---------------------- #
# 파일 내용을 청크로 읽어 MD5 해시 계산
def compute_md5(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()   # MD5 해시 객체 생성
    
    # 바이너리 모드 파일 오픈
    with open(path, "rb") as f:
        # EOF까지 반복 읽기
        while True:
            b = f.read(chunk_size)  # 지정 청크 크기만큼 읽기
            
            # 더 이상 읽을 바이트가 없으면 종료
            if not b: break
            
            # 읽은 바이트로 해시 상태 갱신
            h.update(b)
            
    # 32자리 MD5 문자열 반환
    return h.hexdigest()

# ---------------------- 빈 이미지 생성 ---------------------- #
# 지정 값으로 채운 RGB 빈 이미지 생성
def make_blank_image(fill_value: int = 127, size=(224, 224)) -> np.ndarray:
    """회색 대체 이미지 (탐색 모드에서만 사용 권장)."""
    return np.full((size[1], size[0], 3), fill_value, dtype=np.uint8)

# ---------------------- 이미지 읽기 ------------------------- #
# 이미지를 RGB로 읽어옴. 실패 시 strict=True면 예외 발생, False면 회색 대체 반환.
def imread_rgb(path: str, strict: bool = True) -> np.ndarray:
    # 파일 존재 여부 확인
    if not os.path.exists(path):
        # strict 모드에서 예외 발생
        if strict:
            raise FileNotFoundError(f"[imread_rgb] 파일이 존재하지 않습니다: {path}")

        # strict 모드가 아닐 경우 빈 이미지 반환
        return make_blank_image()

    # 이미지 로드
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # 이미지 로드 실패 시 처리
    if img is None:
        # strict 모드에서 예외 발생
        if strict:
            raise ValueError(f"[imread_rgb] 이미지 로드 실패: {path}")

        # strict 모드가 아닐 경우 빈 이미지 반환
        return make_blank_image()

    # BGR→RGB 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 최종 이미지 반환
    return img

# 안전 로딩: 실패 시 재시도 후 RGB로 반환, 최종 실패 시 None
def imread_rgb(path: str, retries: int = 2, sleep: float = 0.05) -> np.ndarray | None:
    # 허용 재시도 횟수만큼 시도
    for t in range(retries + 1):
        # 디코드 및 색상 변환 시도
        try:
            # 파일에서 바이트 배열 로드 후 디코딩
            img = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8),  # 파일에서 바이트 배열 로드
                cv2.IMREAD_COLOR                    # 컬러 이미지로 디코딩
            )
            
            # 디코드 결과 None일 경우
            if img is None:
                # 예외 유도로 하위 로직 통일
                raise ValueError("cv2.imdecode returned None")
            
            # BGR→RGB 변환 후 반환
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 로딩/디코드 실패 처리
        except Exception:
            # 남은 재시도가 있을 경우 대기 후 반복
            if t < retries:
                # 설정된 대기 후 재시도
                time.sleep(sleep)
            # 마지막 시도까지 실패한 경우
            else:
                # None 반환하여 상위에서 판단
                return None

# ---------------------- 이미지 손상 여부 ---------------------- #
# 이미지 파일 존재/크기/디코드 가능 여부로 손상 판정
def is_broken_image(path: str) -> bool:
    # 경로가 존재하지 않으면 손상으로 간주
    if not os.path.exists(path): return True
    
    # 파일 크기가 0이면 손상으로 간주
    if os.path.getsize(path) == 0: return True
    
    # 최종적으로 읽기 실패 시 손상 처리
    return imread_rgb(path) is None