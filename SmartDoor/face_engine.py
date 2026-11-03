
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from database import get_connection

# 실시간 CCTV는 계속 전송하지만 얼굴 인식 연산 주기는 제한한다.
FACE_PROCESS_INTERVAL = 0.5  # 약 2회/초 (모델 처리 시간이 더 길면 그 속도를 따름)


class FaceEngine:
    def __init__(self, base, frame_source, on_results, on_recognized, on_access,
                 should_process=None):
        self.base = Path(base)
        self.frame_source = frame_source
        self.on_results = on_results
        self.on_recognized = on_recognized
        self.on_access = on_access
        self.should_process = should_process or (lambda: True)

        self.stop_event = threading.Event()
        self.active_event = threading.Event()
        self.active_event.set()
        self.state_lock = threading.Lock()
        self.thread = None
        self.user_count = 0
        self.threshold = 0.45
        self.margin = 0.06
        self.required_frames = 3
        self.absent_seconds = 2.0
        self.last_seen = {}
        self.counts = {}
        self.opened_until_absent = set()
        self.last_logged = {}
        self.users = []

    def set_active(self, active):
        with self.state_lock:
            was_active = self.active_event.is_set()
            if active == was_active:
                return
            if active:
                self.active_event.set()
            else:
                self.active_event.clear()
                # 재접근하면 다시 인증할 수 있게 방문 상태 초기화
                self.last_seen.clear()
                self.counts.clear()
                self.opened_until_absent.clear()

        if not active:
            # CCTV에 이전 얼굴 박스와 등록자 이름이 남지 않도록 초기화
            try:
                self.on_results([])
            except Exception as exc:
                print('[얼굴 상태 초기화 오류]', exc)
        print('[얼굴 인식] 초음파 활성화:', 'ON' if active else 'OFF')

    def load_users(self):
        result = []
        with get_connection() as conn:
            rows = conn.execute('''
                SELECT id, name, face_data_path
                FROM registered_users
                WHERE is_active=1 ORDER BY id
            ''').fetchall()
        for row in rows:
            path = Path(row['face_data_path'] or '')
            if not path.is_absolute():
                path = self.base / path
            if not path.is_file() or path.suffix != '.npz':
                print('[얼굴 데이터 없음]', row['name'], path)
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    features = np.asarray(data['features'], dtype=np.float32)
                if features.ndim != 2 or not len(features):
                    continue
                features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
                representative = features.mean(axis=0)
                representative /= max(float(np.linalg.norm(representative)), 1e-8)
                result.append({
                    'id': row['id'], 'name': row['name'], 'feature': representative
                })
            except Exception as exc:
                print('[얼굴 특징 로딩 오류]', exc)
        self.user_count = len(result)
        print('[얼굴 인식] 등록 사용자:', self.user_count)
        return result

    def identify(self, feature):
        if not self.users:
            return None, 0.0
        feature = np.asarray(feature, dtype=np.float32).flatten()
        feature /= max(float(np.linalg.norm(feature)), 1e-8)
        scored = sorted(
            ((float(np.dot(feature, u['feature'])), u) for u in self.users),
            key=lambda pair: pair[0], reverse=True
        )
        best, user = scored[0]
        if best < self.threshold or (
            len(scored) > 1 and best - scored[1][0] < self.margin
        ):
            return None, best
        return user, best

    def start(self):
        paths = [self.base / 'models' / name for name in (
            'yolov8n-face.pt', 'yunet.onnx', 'sface.onnx'
        )]
        if not all(path.is_file() for path in paths):
            print('[얼굴 인식] 모델 파일이 없어 기능을 시작하지 않습니다:', paths)
            return
        self.thread = threading.Thread(
            target=self.run, name='face-engine', daemon=True
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.active_event.set()  # 대기 중이라면 깨움
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def run(self):
        try:
            cv2.setNumThreads(2)
            # 얼굴 인식이 켜져 있는 동안 Pi CPU 독점을 줄이기 위한 제한
            try:
                import torch
                torch.set_num_threads(2)
            except (ImportError, RuntimeError) as exc:
                print('[얼굴 인식] PyTorch 스레드 제한 생략:', exc)
            yolo = YOLO(str(self.base / 'models/yolov8n-face.pt'))
            yunet = cv2.FaceDetectorYN.create(
                str(self.base / 'models/yunet.onnx'), '', (320, 240), .75, .3, 5000
            )
            sface = cv2.FaceRecognizerSF.create(
                str(self.base / 'models/sface.onnx'), ''
            )
            self.users = self.load_users()
            print('[얼굴 인식] 모델 로딩 완료, 상시 인식 시작')
        except Exception as exc:
            print('[얼굴 모델 오류]', exc)
            return

        while not self.stop_event.is_set():
            # 근접 감지 전에는 YOLO 인식을 수행하지 않는다.
            if not self.active_event.is_set():
                self.stop_event.wait(0.1)
                continue
            # 문 열림/잠금 이동이 진행 중일 때 얼굴 인식 연산을 잠시 쉬어준다.
            if not self.should_process():
                self.stop_event.wait(0.1)
                continue

            start = time.monotonic()
            frame = self.frame_source()
            if frame is None:
                self.stop_event.wait(.15)
                continue
            try:
                height, width = frame.shape[:2]
                small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                results = yolo.predict(
                    small, imgsz=320, conf=.45, device='cpu', verbose=False
                )
                yunet.setInputSize((320, 240))
                _, faces = yunet.detect(small)
                items = []
                used = set()
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = [
                            float(x) for x in box.xyxy[0].tolist()
                        ]
                        selected, best_iou = None, 0.0
                        if faces is not None:
                            for i, face in enumerate(faces):
                                if i in used:
                                    continue
                                fx, fy, fw, fh = face[:4]
                                w = max(0, min(x2, fx + fw) - max(x1, fx))
                                h = max(0, min(y2, fy + fh) - max(y1, fy))
                                inter = w * h
                                union = (x2-x1)*(y2-y1) + fw*fh - inter
                                iou = inter / max(union, 1)
                                if iou > best_iou:
                                    best_iou, selected = iou, i
                        user, score = None, 0.0
                        if selected is not None and best_iou >= .15:
                            used.add(selected)
                            aligned = sface.alignCrop(small, faces[selected])
                            user, score = self.identify(sface.feature(aligned))
                        items.append({
                            'box': [
                                int(x1 * width / 320), int(y1 * height / 240),
                                int(x2 * width / 320), int(y2 * height / 240)
                            ],
                            'user_id': user['id'] if user else None,
                            'name': user['name'] if user else 'Unknown',
                            'score': round(score, 3),
                        })
                # 인식 도중 50cm 밖으로 나갔다면 이전 결과는 버린다.
                if self.active_event.is_set() and self.should_process():
                    self.on_results(items)
                    self.events(items)
            except Exception as exc:
                print('[얼굴 식별 오류]', exc)
                self.stop_event.wait(.5)
            # 얼굴 인식 처리 완료 후 0.5초 쉬기
            self.stop_event.wait(0.5)

    def events(self, items):
        now = time.monotonic()
        visible = {
            'unknown' if it['user_id'] is None else str(it['user_id'])
            for it in items
        }
        for key in list(self.last_seen):
            if key not in visible and now - self.last_seen[key] >= self.absent_seconds:
                self.counts.pop(key, None)
                self.opened_until_absent.discard(key)
                del self.last_seen[key]
        for item in items:
            key = 'unknown' if item['user_id'] is None else str(item['user_id'])
            self.last_seen[key] = now
            if key not in self.counts:
                self.counts[key] = 0
            self.counts[key] += 1
            if self.counts[key] < self.required_frames:
                continue
            if now - self.last_logged.get(key, -1e9) >= 30:
                try:
                    self.on_access(item)
                    self.last_logged[key] = now
                except Exception as exc:
                    print('[얼굴 DB 기록 오류]', exc)
            if item['user_id'] is not None and key not in self.opened_until_absent:
                self.opened_until_absent.add(key)
                try:
                    self.on_recognized({
                        'id': item['user_id'], 'name': item['name']
                    })
                except Exception as exc:
                    print('[얼굴 자동 개방 오류]', exc)
