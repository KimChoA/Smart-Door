import atexit

import os

import secrets

import threading

import time

from datetime import datetime

from pathlib import Path

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file, session

from flask_socketio import SocketIO, join_room, emit

from werkzeug.security import check_password_hash

from database import (

    init_database, get_connection, find_user_by_face,

    add_access_log, add_security_event,

)

from door_hardware import DoorHardware, LOCK_POSITION, UNLOCK_POSITION, OPEN_SECONDS

from security_hardware import SecurityMonitor

from face_engine import FaceEngine

BASE = Path(__file__).resolve().parent

CAPTURES = BASE / 'data' / 'captures'

# 요청에 따라 실제 GPIO와 USB 카메라는 항상 활성화.

# 테스트 모드가 필요할 때에만 별도 사본에서 변경하세요.

GPIO_MODE = True

CAMERA_ENABLED = True

CAMERA_INDEX = int(os.getenv('SMARTGUARD_CAMERA', '0'))

PORT = int(os.getenv('SMARTGUARD_PORT', '5000'))

SERVER_SECRET = os.getenv('SMARTGUARD_SECRET_KEY', '')

# 관리자 페이지는 로그인 없이 열리지만 원격 문 열기는 도어락 비밀번호 확인 필수.

app = Flask(__name__, template_folder=str(BASE / 'templates'))

app.config.update(

    SECRET_KEY=SERVER_SECRET or secrets.token_hex(32),

    SESSION_COOKIE_HTTPONLY=True,

    SESSION_COOKIE_SAMESITE='Lax',

    SESSION_COOKIE_SECURE=os.getenv('SMARTGUARD_HTTPS', '0') == '1',

)

# 개발용 임시 키는 재시작할 때 변경되어 기존 로그인 세션이 해제됨

socketio = SocketIO(app, async_mode='threading')

state_lock = threading.RLock()

camera_lock = threading.Lock()

camera_frame_ready = threading.Event()

door_lock = threading.Lock()

approach_lock = threading.Lock()

approach_active = False

approach_handled = False

approach_authorized = False

state = {

    'door_locked': True,                # 명령한 위치 기준, 센서 측정값 아님

    'door_active': False,              # 문 열기~잠금 복귀 제어 작업 진행 중

    'door_motion_error': None,        # 최근 도어락 제어 오류

    'security_enabled': True,

    'camera_connected': False,

    'face_status': '대기',

    'face_recognition_enabled': True,

    'failed_attempts': 0,

    'detection_seconds': 10,

    'hardware_mode': GPIO_MODE,

}

latest_jpeg = None

latest_frame = None

face_results = []

face_results_at = 0.0

face_engine = None

face_lock = threading.Lock()

camera_thread_started = False

detection_running = False

hardware_started = False

door_device = DoorHardware(enabled=GPIO_MODE)

security_device = SecurityMonitor(

    enabled=GPIO_MODE,

    on_warning=lambda cm: handle_proximity_warning(cm),

    is_security_enabled=lambda: is_security_enabled(),

    is_known_face=lambda: is_approach_authorized(),

    on_approach=lambda active: update_approach(active),

)

def stamp():

    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def snapshot_state():

    with state_lock:

        return dict(state)

def publish_state():

    socketio.emit('status_updated', snapshot_state(), to='admins')

def is_security_enabled():

    with state_lock:

        return state['security_enabled']

def is_approach_authorized():

    with approach_lock:

        return approach_active and approach_authorized

def update_approach(active):

    global approach_active, approach_handled, approach_authorized, face_results, face_results_at

    with approach_lock:

        approach_active = bool(active)

        approach_handled = False

        approach_authorized = False

    with face_lock:

        face_results = []

        face_results_at = 0.0

    with state_lock:

        state['face_status'] = '인식 중' if active else '대기'

    try:

        publish_state()

    except Exception as exc:

        print('[접근 상태 알림 오류]', exc)

    if active and face_engine is not None:

        face_engine.request_reset()

    if not active:
        door_device.stop_warning_buzzer()

    print('[접근 감지]', '얼굴 인증 시작' if active else '접근 종료, 다음 인증 대기')

def is_known_face_present():

    return is_approach_authorized()

def handle_proximity_warning(distance_cm):

    with approach_lock:
        still_unauthorized = approach_active and not approach_authorized

    if not still_unauthorized or not is_security_enabled():
        return

    door_device.start_warning_buzzer()

    if is_known_face_present():

        print('[근접 감지] 등록된 얼굴 확인: 보안 사진 및 경고 기록 생략')

        return

    security_event(

        'prolonged_presence',

        f'현관 앞 물체가 5초 이상 감지되었습니다. 거리: {distance_cm:.1f}cm',

        take_photo=True,

    )

def access_log(method, success, source, detail='', user_id=None):

    log_id = add_access_log(

        method, success, source, user_id=user_id, detail=detail

    )

    socketio.emit('access_updated', {'id': log_id, 'timestamp': stamp()}, to='admins')

    return log_id

def capture_photo():

    # 비밀번호 실패/접근 보안 이벤트 등 사진 기록이 필요한 경우 프레임을 기다림

    if CAMERA_ENABLED and not camera_frame_ready.is_set():

        camera_frame_ready.wait(timeout=2.0)

    with camera_lock:

        photo = latest_jpeg

    if photo is None:

        print('[카메라] 촬영 가능한 프레임이 없어 이벤트 사진을 저장하지 못했습니다.')

        return None

    CAPTURES.mkdir(parents=True, exist_ok=True)

    name = 'event_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.jpg'

    (CAPTURES / name).write_bytes(photo)

    return name

def security_event(event_type, message, take_photo=False, access_id=None):

    filename = capture_photo() if take_photo else None

    event_id = add_security_event(

        event_type, message, media_path=filename, access_log_id=access_id

    )

    event = {

        'id': event_id,

        'timestamp': stamp(),

        'event_type': event_type,

        'message': message,

        'media_path': filename,

    }

    socketio.emit('security_alert', event, to='admins')

    print('[보안 이벤트]', message, '사진:', filename)

    return event

# 통합 도어락 제어: 한 프로세스만 GPIO12 소유


def request_open_door(source='unknown', user_id=None):

    if not door_lock.acquire(blocking=False):

        print('[도어락] 이미 작동 중')

        return False

    door_device.stop_warning_buzzer()

    with state_lock:

        state['door_active'] = True

        state['door_motion_error'] = None

    try:

        publish_state()

    except Exception as exc:

        print('[도어락 상태 알림 오류]', exc)

    def safe_side_effect(label, action):

        try:

            action()

        except Exception as exc:

            print(f'[도어락 부가 기능 오류] {label}: {exc}')

    def worker():

        opened = False

        locked = False

        try:

            print('[서보] 열림 위치로 이동')

            door_device.move(UNLOCK_POSITION)

            opened = True

            with state_lock:

                state['door_locked'] = False

            safe_side_effect('LED 열림', lambda: security_device.set_door_open(True))

            safe_side_effect('상태 송신', publish_state)

            safe_side_effect('열림 로그', lambda: access_log(

                'door_open', True, source,

                detail=f'{source} 문 열림 위치 이동 명령 완료', user_id=user_id

            ))

            time.sleep(OPEN_SECONDS)

        except Exception as exc:

            print('[도어락 열림 오류]', repr(exc))

            with state_lock:

                state['door_motion_error'] = f'서보 개방 중 오류: {exc}'

        finally:

            # LED, DB, 네트워크 오류와 관계없이 잠금 복귀를 우선 시도합니다.

            try:

                print('[서보] 잠금 위치로 복귀 시도')

                door_device.move(LOCK_POSITION)

                locked = True

                with state_lock:

                    state['door_locked'] = True

                print('[서보] 잠금 위치 이동 명령 완료 (물리적 잠금 확인 아님)')

            except Exception as exc:

                print('[도어락 잠금 복귀 오류]', repr(exc))

                with state_lock:

                    state['door_motion_error'] = f'서보 잠금 복귀 오류: {exc}'

            finally:

                safe_side_effect('LED 복귀', lambda: security_device.set_door_open(False))

                if locked:

                    safe_side_effect('잠금 로그', lambda: access_log(

                        'door_lock', True, source,

                        detail='문 잠금 위치 복귀 명령 완료', user_id=user_id

                    ))

                else:

                    safe_side_effect('잠금 실패 이벤트', lambda: security_event(

                        'door_error', '서보 잠금 복귀 실패. 실제 도어 상태를 확인하세요.'

                    ))

                with state_lock:

                    state['door_active'] = False

                safe_side_effect('상태 송신', publish_state)

                door_lock.release()

    try:

        threading.Thread(target=worker, daemon=True, name='door-worker').start()

    except Exception:

        with state_lock:

            state['door_active'] = False

        door_lock.release()

        raise

    return True

def on_indoor_open():

    print('[실내 스위치] 버튼 입력 감지!', flush=True)
    accepted = request_open_door('house_switch')
    print(f'[실내 스위치] 문 열기 요청: {accepted}', flush=True)

def handle_doorbell():

    try:

        # 초인종은 사진을 촬영/저장하지 않고 실시간 영상만 팝업에 표시한다.

        security_event(

            'doorbell',

            '초인종이 울렸습니다. 실시간 현관 영상을 확인하세요.',

            take_photo=False,

        )

        door_device.play_doorbell()

    except Exception as exc:

        print('[초인종 오류]', exc)

def on_doorbell():

    threading.Thread(target=handle_doorbell, daemon=True, name='doorbell').start()

def record_password_attempt(success, source='doorlock'):

    with state_lock:

        if success:

            state['failed_attempts'] = 0

        else:

            state['failed_attempts'] += 1

        count = state['failed_attempts']

        alert = not success and count >= 3

        if alert:

            state['failed_attempts'] = 0

    log_id = access_log(

        'password', success, source,

        '비밀번호 인증 성공' if success else '비밀번호 인증 실패',

    )

    if alert:

        security_event(

            'password_alert', '비밀번호 3회 연속 실패',

            take_photo=True, access_id=log_id,

        )

    publish_state()

    return count, alert

# 카메라 (원본 USB/V4L2 OpenCV 방식 유지)


def camera_worker():

    global latest_jpeg, latest_frame

    try:

        import cv2

    except ImportError as exc:

        print('[카메라] opencv-python/python3-opencv 미설치:', exc)

        return

    while True:

        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

        if not cap.isOpened():

            cap.release()

            cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():

            cap.release()

            with camera_lock:

                latest_jpeg = None

                latest_frame = None

                camera_frame_ready.clear()

            with state_lock:

                state['camera_connected'] = False

            publish_state()

            print(f'[카메라] 장치 {CAMERA_INDEX} 연결 실패. 2초 후 재시도')

            time.sleep(2)

            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cap.set(cv2.CAP_PROP_FPS, 15)

        print(f'[카메라] 장치 {CAMERA_INDEX} 열림. 첫 프레임 대기 중')

        first_frame = True

        errors = 0

        while True:

            ok, frame = cap.read()

            if not ok:

                errors += 1

                if errors > 15:

                    break

                time.sleep(.1)

                continue

            errors = 0

            with camera_lock:

                latest_frame = frame.copy()

            display = frame.copy()

            with face_lock:

                face_items = list(face_results) if time.monotonic() - face_results_at < 2.0 else []

            for item in face_items:

                x1, y1, x2, y2 = item['box']

                color = (70, 200, 95) if item['user_id'] is not None else (70, 70, 225)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                label = f"Known #{item['user_id']}" if item['user_id'] is not None else 'Unknown'

                cv2.putText(display, label, (x1, max(17, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)

            ok, encoded = cv2.imencode(

                '.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 85]

            )

            if ok:

                with camera_lock:

                    latest_jpeg = encoded.tobytes()

                    camera_frame_ready.set()

                if first_frame:

                    first_frame = False

                    with state_lock:

                        state['camera_connected'] = True

                    publish_state()

                    print('[카메라] 첫 영상 프레임 수신 완료')

            time.sleep(.06)

        cap.release()

        with camera_lock:

            latest_jpeg = None

            latest_frame = None

            camera_frame_ready.clear()

        with state_lock:

            state['camera_connected'] = False

        publish_state()

        time.sleep(1)

def get_camera_frame():

    with approach_lock:

        active = approach_active

    if not active:

        return None

    with camera_lock:

        return latest_frame.copy() if latest_frame is not None else None

def on_face_results(items):

    global face_results, face_results_at

    with approach_lock:

        if not approach_active:

            return

        authorized = approach_authorized

    with face_lock:

        face_results = list(items)

        face_results_at = time.monotonic()

    with state_lock:

        names = [item['name'] for item in items if item['user_id'] is not None]

        new_status = ', '.join(dict.fromkeys(names)) if names else ('Unknown' if items else '인식 중')

        changed = state['face_status'] != new_status

        state['face_status'] = new_status

    if changed:

        publish_state()

def on_face_open(user):

    global approach_handled, approach_authorized

    with state_lock:

        enabled = state['face_recognition_enabled']

    if not enabled:

        return False

    with face_lock:

        recent = time.monotonic() - face_results_at < 2.0

        registered_present = (
            recent and any(
                item.get('user_id') == user['id']
                for item in face_results
            )
        )

    if not registered_present:

        print('[얼굴 인증] 현재 화면에 해당 등록 사용자가 없어 개방 보류')

        return False

    with approach_lock:

        if not approach_active or approach_handled:

            return False

        approach_handled = True

    accepted = request_open_door('face', user_id=user['id'])

    if accepted:

        door_device.stop_warning_buzzer()

        with approach_lock:

            approach_authorized = True

        print('[얼굴 자동 개방] 접근 1회 처리:', user['name'])

    else:

        print('[얼굴 자동 개방] 다른 문 작동 중 - 이번 접근에서 재요청 안 함')

    return accepted

def start_face_engine():

    global face_engine

    face_engine = FaceEngine(

        base=BASE, frame_source=get_camera_frame,

        on_results=on_face_results, on_recognized=on_face_open,

        on_access=lambda item: access_log(

            'face_recognition', item['user_id'] is not None,

            'usb_camera',

            f"등록 사용자 {item['name']} 얼굴 인식" if item['user_id'] is not None else '미등록 또는 인식 불확실 얼굴',

            user_id=item['user_id'],

        ),

    )

    face_engine.start()

def start_camera():

    global camera_thread_started

    if camera_thread_started or not CAMERA_ENABLED:

        return

    camera_thread_started = True

    threading.Thread(target=camera_worker, daemon=True, name='camera').start()

def mjpeg_frames():

    while True:

        with camera_lock:

            frame = latest_jpeg

        if frame is None:

            time.sleep(.2)

            continue

        yield (

            b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

        )

        time.sleep(.09)

# 웹 페이지 / API


@app.get('/')

def index():

    return redirect('/doorlock')

@app.get('/doorlock')

def doorlock_page():

    return render_template('doorlock.html')

@app.get('/admin')

def admin_page():

    # 별도의 로그인은 없지만 POST 요청의 CSRF 검증은 계속 사용

    if 'csrf_token' not in session:

        session['csrf_token'] = secrets.token_urlsafe(32)

    return render_template(

        'admin.html',

        csrf_token=session['csrf_token'],

        hardware_mode=GPIO_MODE,

    )

@app.get('/admin/login')

def old_admin_login_redirect():

    # 이전에 저장한 관리자 로그인 주소는 대시보드로 이동

    return redirect('/admin')

def valid_csrf():

    expected = session.get('csrf_token', '')

    actual = request.headers.get('X-CSRF-Token', '') or request.form.get('csrf_token', '')

    return bool(expected) and secrets.compare_digest(expected, actual)

# 로그인 없이 조회 가능. 상태 변경 API는 CSRF 토큰을 요구

@app.before_request

def protect_admin_endpoints():

    path = request.path

    protected = (

        path.startswith('/api/admin/')

        or path.startswith('/api/access/')

        or path.startswith('/api/security/')

        or path.startswith('/api/camera/')

        or path.startswith('/api/test/')

        or path in ('/api/door/open', '/api/door/reject')

    )

    if not protected:

        return None

    if request.method in ('POST', 'PUT', 'PATCH', 'DELETE') and not valid_csrf():

        return jsonify(success=False, message='CSRF 인증 실패. 관리자 페이지를 새로고침하세요.'), 403

    return None

@app.post('/api/door/unlock')

def receive_password():

    data = request.get_json(silent=True) or {}

    password = data.get('password')

    if not isinstance(password, str) or not password:

        return jsonify(success=False, message='비밀번호를 입력하세요.'), 400

    with get_connection() as conn:

        row = conn.execute(

            'SELECT password_hash FROM doorlock_settings WHERE id = 1'

        ).fetchone()

    if row is None:

        return jsonify(success=False, message='등록된 비밀번호가 없습니다.'), 503

    success = check_password_hash(row['password_hash'], password)

    count, alert = record_password_attempt(success)

    if success:

        if not request_open_door('password'):

            return jsonify(

                success=False,

                message='인증 성공. 현재 문이 작동 중입니다.'

            ), 409

        return jsonify(success=True, message='인증 성공! 문이 열립니다.')

    if alert:

        return jsonify(success=False, message='비밀번호 3회 오류! 보안 경고 발생')

    return jsonify(success=False, message=f'비밀번호 오류 ({count}/3)')

@app.get('/api/door/status')

def door_status():

    # 키패드에서 잠금 복귀 완료를 확인하기 위한 최소한의 상태만 반환.

    current = snapshot_state()

    return jsonify({

        'door_locked': current['door_locked'],

        'door_active': current['door_active'],

        'door_motion_error': current['door_motion_error'],

    })

@app.get('/api/admin/status')

def admin_status():

    return jsonify(snapshot_state())

@app.get('/api/access/logs')

def list_access():

    with get_connection() as conn:

        rows = conn.execute('''

            SELECT a.id, a.timestamp, a.method, a.success, a.source,

                   a.detail, u.name AS user_name

            FROM access_logs a

            LEFT JOIN registered_users u ON u.id = a.user_id

            ORDER BY a.id DESC LIMIT 100

        ''').fetchall()

    return jsonify([dict(row) for row in rows])

@app.get('/api/security/events')

def list_security():

    with get_connection() as conn:

        rows = conn.execute('''

            SELECT * FROM security_events ORDER BY id DESC LIMIT 100

        ''').fetchall()

    return jsonify([dict(row) for row in rows])

@app.get('/api/security/media/<int:event_id>')

def security_media(event_id):

    with get_connection() as conn:

        row = conn.execute(

            'SELECT media_path FROM security_events WHERE id = ?', (event_id,)

        ).fetchone()

    if not row or not row['media_path']:

        abort(404)

    filename = row['media_path']

    if Path(filename).name != filename:

        abort(403)

    path = CAPTURES / filename

    if not path.is_file() or path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):

        abort(404)

    return send_file(path)

@app.get('/api/face/status')

def face_status():

    with face_lock:

        items = list(face_results)

    return jsonify({'detections': items, 'registered_users': face_engine.user_count if face_engine else 0})

@app.get('/api/camera/snapshot')

def camera_snapshot():

    """USB 카메라가 실제 JPEG 프레임을 공급하는지 확인하는 진단 주소."""

    with camera_lock:

        frame = latest_jpeg

    if frame is None:

        return jsonify(success=False, message='카메라 프레임이 아직 없습니다.'), 503

    return Response(frame, mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})

@app.get('/api/camera/stream')

def camera_stream():

    return Response(

        mjpeg_frames(),

        mimetype='multipart/x-mixed-replace; boundary=frame',

        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'},

    )

@app.post('/api/door/open')

def remote_open():

    # 관리자 페이지 로그인은 생략하지만 실제 문 열기는 PIN 인증 필요

    data = request.get_json(silent=True) or {}

    password = data.get('password')

    if not isinstance(password, str) or not password:

        return jsonify(success=False, message='도어락 비밀번호를 입력하세요.'), 400

    with get_connection() as conn:

        row = conn.execute(

            'SELECT password_hash FROM doorlock_settings WHERE id = 1'

        ).fetchone()

    if row is None:

        return jsonify(success=False, message='등록된 도어락 비밀번호가 없습니다.'), 503

    success = check_password_hash(row['password_hash'], password)

    count, alert = record_password_attempt(success, source='admin')

    if not success:

        return jsonify(

            success=False,

            message=('비밀번호 3회 오류! 보안 경고 발생' if alert

                     else f'도어락 비밀번호 오류 ({count}/3)'),

        ), 401

    if not request_open_door('admin'):

        return jsonify(success=False, message='현재 문이 작동 중입니다.'), 409

    access_log('remote', True, 'admin', '관리자 원격 문 열기 요청')

    return jsonify(success=True, message='문 열기 명령을 전달했습니다.')

@app.post('/api/door/reject')

def remote_reject():

    security_event('visitor_rejected', '관리자가 방문자 출입을 거절했습니다.')

    return jsonify(success=True)

# 관리자 페이지의 기존 모의 테스트용 API

# 실제 GPIO 모드에서 가짜 이벤트를 만들지 않도록 비활성화

def simulate_unknown_dwell():

    global detection_running

    with state_lock:

        if detection_running:

            return False

        detection_running = True

        state['face_status'] = '미등록 사용자'

    publish_state()

    def worker():

        global detection_running

        try:

            time.sleep(5)

            if is_security_enabled():

                security_event(

                    'intrusion', '미등록 사용자가 5초 이상 체류했습니다.',

                    take_photo=True,

                )

        finally:

            with state_lock:

                detection_running = False

                state['face_status'] = '대기'

            publish_state()

    threading.Thread(target=worker, daemon=True).start()

    return True

@app.post('/api/test/event')

def test_event():

    if GPIO_MODE:

        return jsonify(

            success=False, message='GPIO 모드에서 테스트 API 비활성화'

        ), 403

    data = request.get_json(silent=True) or {}

    kind = data.get('type')

    if kind == 'doorbell':

        on_doorbell()

    elif kind == 'password_ok':

        record_password_attempt(True)

    elif kind == 'password_fail':

        record_password_attempt(False)

    elif kind == 'unknown_person':

        if not simulate_unknown_dwell():

            return jsonify(success=False, message='이미 감지 중입니다.'), 409

    elif kind == 'face_registered':

        label = str(data.get('face_label', '')).strip()

        user = find_user_by_face(label) if label else None

        if not user:

            return jsonify(

                success=True, recognized=False, message='미등록 사용자'

            )

        if not request_open_door('camera', user_id=user['id']):

            return jsonify(success=False, message='문이 작동 중입니다.'), 409

        with state_lock:

            state['face_status'] = user['name']

        access_log('face', True, 'camera', '등록 얼굴 인식 테스트', user['id'])

        publish_state()

        return jsonify(success=True, recognized=True, message='등록 얼굴 확인')

    else:

        return jsonify(success=False, message='지원하지 않는 테스트'), 400

    return jsonify(success=True, message='테스트 이벤트 실행 완료')

@socketio.on('connect')

def connected(auth=None):

    # 관리자 로그인 없이 웹소켓 실시간 알림 구독 허용

    join_room('admins')

    emit('status_updated', snapshot_state())

def initialize():

    global hardware_started

    if not SERVER_SECRET:

        print('[안내] SMARTGUARD_SECRET_KEY가 없어 실행 시 임시 CSRF 세션 키를 사용합니다.')

    init_database()

    if GPIO_MODE:

        # 두 GPIO 모듈을 한 프로세스가 소유. 기존 서버/테스트 동시 실행 금지.

        try:

            door_device.initialize(on_indoor_open, on_doorbell)

            security_device.initialize()

            security_device.start()

        except Exception:

            security_device.close()

            door_device.close()

            raise

    start_camera()

    start_face_engine()

    hardware_started = True

def cleanup():

    if face_engine is not None:

        face_engine.stop()

    security_device.close()

    door_device.close()

atexit.register(cleanup)

if __name__ == '__main__':

    initialize()

    print('Smart Guard Home 통합 서버')

    print('GPIO:', GPIO_MODE, '| CAMERA:', CAMERA_ENABLED, '| PORT:', PORT)

    print('도어락: /doorlock | 관리자: /admin')

    try:

        socketio.run(

            app, host='0.0.0.0', port=PORT,

            debug=False, use_reloader=False,

            allow_unsafe_werkzeug=True,  # 학습용 LAN 개발 서버로만 사용

        )

    finally:

        cleanup()