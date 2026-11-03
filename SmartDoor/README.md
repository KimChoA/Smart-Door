# Smart Guard Home

Raspberry Pi 5, Python, Flask, OpenCV 기반 스마트 출입 보안 시스템입니다.
YOLOv8 얼굴 검출과 YuNet·SFace를 활용한 사용자 인증을 도어락 및 관리자 웹과 연동합니다.

## 주요 기능

- 등록 사용자 얼굴 인식 및 서보모터 도어락 자동 개방
- 비밀번호 인증, 실내 스위치 및 관리자 원격 개방
- 초인종 알림과 관리자 웹 실시간 영상 팝업
- 초음파 센서의 20cm 이내 접근 감지 및 미인증 접근 10초 경과 시 경고·사진 저장
- 비밀번호 3회 연속 실패 시 보안 이벤트 및 사진 기록
- RGB LED·부저 상태 표시, SQLite 출입·보안 기록 조회
- 별도 웹 서버를 통한 얼굴 등록 및 기존 등록자 데이터 보강

## 기술 구성

| 구분 | 사용 기술 |
|---|---|
| 장치 | Raspberry Pi 5, USB 웹캠, 서보모터, 초음파 센서, 스위치, 초인종, RGB LED, 부저 |
| 서버·웹 | Python, Flask, Flask-SocketIO, HTML/CSS/JavaScript |
| 영상·얼굴 인식 | OpenCV, YOLOv8, YuNet, SFace, NumPy |
| 데이터베이스 | SQLite |
| GPIO | gpiozero, lgpio |

## 파일 구성

| 파일 | 역할 |
|---|---|
| `app.py` | 통합 서버, 카메라 스트리밍, 인증 및 이벤트 연동 |
| `database.py` | DB 초기화, 사용자·출입·보안 기록 관리 |
| `door_hardware.py` | 서보, 실내 스위치, 초인종, 부저 제어 |
| `security_hardware.py` | 초음파 접근 감지와 RGB LED 제어 |
| `face_engine.py` | 얼굴 검출·특징 비교 및 인증 |
| `face_register.py` | 얼굴 등록·데이터 보강 서버 |
| `templates/` | 도어락 및 관리자 웹 화면 |

## GPIO 배선

아래 값은 이 저장소 코드의 BCM 번호입니다.

| 장치 | GPIO |
|---|---|
| 서보모터 | 13 |
| 실내 스위치 | 21 |
| 초인종 | 20 |
| 부저 | 14 |
| 초음파 TRIG / ECHO | 23 / 24 |
| RGB LED R / G / B | 17 / 27 / 22 |

잠금 위치는 -0.5, 개방 위치는 0.5이며 개방 유지 시간은 2초입니다. 실제 기구에 맞게 조정해야 합니다.

## 설치 및 실행

Raspberry Pi OS 환경에서 실행합니다. 의존성 버전은 원본에 기록되어 있지 않아 고정하지 않았습니다. 하드웨어와 OS에 따라 호환 버전 조정이 필요합니다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-gpiozero python3-lgpio
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. 모델 준비

원본 ZIP의 다음 모델을 로컬 `models/` 폴더에 복사합니다. 공개용 압축파일에는 가중치를 포함하지 않았습니다. 원본의 모델 출처·버전·배포 조건은 확인이 필요합니다.

- `models/yolov8n-face.pt`
- `models/yunet.onnx`
- `models/sface.onnx`

### 2. DB 초기화 및 비밀번호 설정

기존 사용자·사진·기록 DB는 포함하지 않습니다. 새 DB를 만들고 비밀번호를 직접 입력합니다.

```bash
python database.py
python - <<'PY'
from getpass import getpass
from database import set_doorlock_password
password = getpass('새 도어락 비밀번호 (최소 4자리): ')
if password != getpass('비밀번호 확인: '):
    raise SystemExit('비밀번호가 일치하지 않습니다.')
set_doorlock_password(password)
print('설정 완료')
PY
```

### 3. 얼굴 등록

통합 서버를 종료한 상태에서 등록 서버를 실행합니다. 같은 웹캠을 두 서버가 동시에 사용하지 않도록 합니다. 등록 서버의 기본 카메라 인덱스는 2이므로 환경변수로 장치를 지정합니다.

```bash
SMARTGUARD_CAMERA=0 python face_register.py
```

`http://<라즈베리파이-IP>:5004`에서 새 사용자를 등록합니다. 등록을 마친 후 서버를 종료합니다.

### 4. 통합 서버 실행

```bash
SMARTGUARD_CAMERA=0 python app.py
```

- 도어락: `http://<라즈베리파이-IP>:5000/doorlock`
- 관리자: `http://<라즈베리파이-IP>:5000/admin`

`SMARTGUARD_PORT`로 통합 서버 포트를 변경할 수 있습니다. `SMARTGUARD_SECRET_KEY`를 설정하지 않으면 실행할 때마다 세션 키가 생성됩니다.

## 공개용 파일 구성 및 제한

- 얼굴 사진, 얼굴 특징 파일, 촬영 사진, 운영 DB와 로그는 제외했습니다.
- `requirements.txt`, `.gitignore`, 이 README를 추가했으며 원본 Python·HTML은 유지했습니다.
- 관리자 화면과 얼굴 등록 서버에 별도 로그인 인증이 없는 실습용 구현입니다. 신뢰할 수 있는 로컬 네트워크에서 사용하며 실제 운영 전 접근 인증이 필요합니다.
- 잠금 상태 표시는 서보 명령 기준이며 실제 잠금 센서 측정값이 아닙니다.
- `app.py`의 장기 접근 이벤트 문구는 5초로 남아 있으나 실제 감지 기준은 `security_hardware.py`의 10초입니다.
- Python 문법 검사를 완료했습니다. Raspberry Pi GPIO·카메라·모델을 사용하는 실제 동작은 별도 확인이 필요합니다.
