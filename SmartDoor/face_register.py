
import time
import uuid
import threading
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template_string

from database import init_database, get_connection


# 기본 설정

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FACE_DIR = BASE_DIR / "faces"

YUNET_PATH = MODEL_DIR / "yunet.onnx"
SFACE_PATH = MODEL_DIR / "sface.onnx"

CAMERA_INDEX = int(os.getenv("SMARTGUARD_CAMERA", "2"))
TARGET_COUNT = 20
MIN_COUNT = 15
MIN_EXTRA_COUNT = 5

FACE_DIR.mkdir(exist_ok=True)

frame_lock = threading.Lock()
registration_lock = threading.Lock()
latest_frame = None

registration = None
running = True


# 얼굴 모델 로딩

if not YUNET_PATH.exists() or not SFACE_PATH.exists():
    raise FileNotFoundError(
        "얼굴 모델이 없습니다. yunet.onnx와 sface.onnx를 확인하세요."
    )

detector = cv2.FaceDetectorYN.create(
    str(YUNET_PATH),
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    str(SFACE_PATH),
    ""
)

print("[얼굴 등록] 모델 로딩 완료")


# USB 웹캠 수집

def camera_worker():
    global latest_frame

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2
    )

    if not camera.isOpened():
        print("[카메라] USB 웹캠 연결 실패")
        return

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[카메라] USB 웹캠 정상 연결")

    try:
        while running:
            success, frame = camera.read()

            if not success:
                time.sleep(0.1)
                continue

            with frame_lock:
                latest_frame = frame

    finally:
        camera.release()
        print("[카메라] 종료")


# 얼굴 검출 및 특징 추출

def extract_face(frame):
    # YuNet 고정 입력 크기 사용
    small = cv2.resize(frame, (320, 320))

    detector.setInputSize((320, 320))

    _, faces = detector.detect(small)

    if faces is None or len(faces) == 0:
        raise ValueError("얼굴을 찾지 못했습니다.")

    if len(faces) > 1:
        raise ValueError("한 사람만 카메라 앞에 서주세요.")

    face = faces[0]

    # 얼굴 위치와 랜드마크 정렬
    aligned = recognizer.alignCrop(
        small,
        face
    )

    # 얼굴 특징 벡터 추출
    feature = recognizer.feature(
        aligned
    ).flatten().astype(np.float32)

    # 정규화
    norm = np.linalg.norm(feature)

    if norm <= 0:
        raise ValueError("얼굴 특징 추출에 실패했습니다.")

    feature = feature / norm

    return aligned, feature


# 실시간 영상 스트리밍

def generate_frames():
    while running:
        with frame_lock:
            frame = (
                latest_frame.copy()
                if latest_frame is not None
                else None
            )

        if frame is None:
            time.sleep(0.1)
            continue

        height, width = frame.shape[:2]

        # 촬영 가이드
        x1 = int(width * 0.22)
        x2 = int(width * 0.78)
        y1 = int(height * 0.12)
        y2 = int(height * 0.88)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (80, 210, 140),
            2
        )

        cv2.putText(
            frame,
            "Face Registration",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if success:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )

        time.sleep(0.07)


# 기존 사용자 확인 및 사진 보강

def existing_features(row):
    path = Path(row["face_data_path"] or "")
    if not path.is_absolute():
        path = BASE_DIR / path
    path = path.resolve()
    if FACE_DIR.resolve() not in path.parents or path.name != "features.npz":
        raise ValueError("기존 얼굴 데이터 경로가 올바르지 않습니다.")
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("기존 특징 데이터가 올바르지 않습니다.")
    return path, features


@app.get("/api/users")
def list_users():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, name, face_data_path FROM registered_users
            WHERE is_active = 1 ORDER BY id
        """).fetchall()
    return jsonify([{"id": row["id"], "name": row["name"]} for row in rows])


# 등록 시작

@app.route("/api/start", methods=["POST"])
def start_registration():
    global registration
    data = request.get_json(silent=True) or {}
    mode = str(data.get("mode", "new"))
    name = str(data.get("name", "")).strip()
    if mode not in ("new", "add"):
        return jsonify(success=False, message="등록 방식을 선택하세요."), 400
    if mode == "new" and (not name or len(name) > 50):
        return jsonify(success=False, message="이름을 1~50자로 입력해주세요."), 400

    with registration_lock:
        if registration is not None:
            return jsonify(success=False, message="이미 등록이 진행 중입니다."), 409

        user_id = None
        reference = None
        feature_path = None
        if mode == "add":
            try:
                user_id = int(data.get("user_id"))
                with get_connection() as conn:
                    row = conn.execute("""
                        SELECT id, name, face_label, face_data_path FROM registered_users
                        WHERE id = ? AND is_active = 1
                    """, (user_id,)).fetchone()
                if row is None:
                    raise ValueError("선택한 등록자가 없습니다.")
                feature_path, features = existing_features(row)
                normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
                reference = normalized.mean(axis=0)
                reference /= max(float(np.linalg.norm(reference)), 1e-8)
                name = row["name"]
                label = row["face_label"]
            except (ValueError, TypeError, KeyError, OSError) as exc:
                return jsonify(success=False, message=str(exc)), 400
        else:
            label = "person_" + uuid.uuid4().hex[:12]

        token = uuid.uuid4().hex
        registration = {
            "name": name, "label": label, "token": token,
            "mode": mode, "user_id": user_id,
            "feature_path": feature_path, "reference": reference,
            "photos": [], "features": [], "last_capture": 0,
        }

    return jsonify(success=True, token=token,
                   message=f"{name}님 {'기존 데이터 보강' if mode == 'add' else '신규 등록'} 시작")


# 얼굴 촬영

@app.route("/api/capture", methods=["POST"])
def capture_face():
    data = request.get_json(silent=True) or {}
    token = data.get("token")

    with registration_lock:
        info = registration

        if info is None or info["token"] != token:
            return jsonify({
                "success": False,
                "message": "등록 세션이 없습니다."
            }), 400

        count = len(info["photos"])

        if count >= TARGET_COUNT:
            return jsonify({
                "success": False,
                "message": "이미 20장을 촬영했습니다."
            }), 400

        if time.monotonic() - info["last_capture"] < 0.3:
            return jsonify({
                "success": False,
                "message": "잠시 후 다시 촬영해주세요."
            }), 429

        with frame_lock:
            frame = (
                latest_frame.copy()
                if latest_frame is not None
                else None
            )

        if frame is None:
            return jsonify({
                "success": False,
                "message": "카메라 영상이 없습니다."
            }), 503

        try:
            face, feature = extract_face(frame)

        except Exception as error:
            return jsonify({
                "success": False,
                "message": str(error)
            }), 400

        if info["mode"] == "add":
            # 등록한 본인과 현저히 다른 얼굴 데이터 혼입 방지
            similarity = float(np.dot(feature, info["reference"]))
            if similarity < 0.40:
                return jsonify(success=False,
                    message=f"기존 등록 얼굴과 유사도가 낮습니다 ({similarity:.2f}). 촬영 각도와 조명을 조정하세요."), 400

        info["photos"].append(face)
        info["features"].append(feature)
        info["last_capture"] = time.monotonic()

        count = len(info["photos"])

        print(
            f"[얼굴 촬영] {info['name']}: "
            f"{count}/{TARGET_COUNT}"
        )

        return jsonify({
            "success": True,
            "count": count,
            "message": f"{count}장 촬영 완료"
        })


# SQLite 등록 및 특징 데이터 보강

@app.route("/api/finish", methods=["POST"])
def finish_registration():
    global registration
    token = (request.get_json(silent=True) or {}).get("token")
    with registration_lock:
        info = registration
        if info is None or info["token"] != token:
            return jsonify(success=False, message="등록 세션이 없습니다."), 400
        minimum = MIN_EXTRA_COUNT if info["mode"] == "add" else MIN_COUNT
        if len(info["photos"]) < minimum:
            return jsonify(success=False, message=f"최소 {minimum}장을 촬영해주세요."), 400

        try:
            if info["mode"] == "add":
                # 기존 사용자 ID와 특징 파일을 그대로 유지하고 안전하게 추가
                with get_connection() as conn:
                    row = conn.execute("""
                        SELECT id, face_data_path FROM registered_users
                        WHERE id = ? AND is_active = 1
                    """, (info["user_id"],)).fetchone()
                if row is None:
                    raise ValueError("사용자가 더 이상 등록 상태가 아닙니다.")
                feature_path, previous = existing_features(row)
                if feature_path != info["feature_path"]:
                    raise ValueError("등록 중 특징 파일 경로가 변경됐습니다.")
                if previous.shape[1] != len(info["features"][0]):
                    raise ValueError("특징 벡터 크기가 다릅니다.")
                backup = feature_path.with_name("features_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".backup.npz")
                shutil.copy2(feature_path, backup)
                new_features = np.concatenate([previous, np.stack(info["features"])], axis=0)
                temporary = feature_path.with_name("features_" + uuid.uuid4().hex + ".tmp.npz")
                try:
                    np.savez_compressed(temporary, features=new_features)
                    os.replace(temporary, feature_path)
                finally:
                    temporary.unlink(missing_ok=True)
                folder = feature_path.parent
                next_index = 1
                while (folder / f"{next_index:03d}.jpg").exists():
                    next_index += 1
                for face in info["photos"]:
                    if not cv2.imwrite(str(folder / f"{next_index:03d}.jpg"), face):
                        raise RuntimeError("특징은 저장됐지만 일부 사진 저장에 실패했습니다. 백업을 확인하세요.")
                    next_index += 1
                count = len(info["photos"])
                registration = None
                print(f"[얼굴 보강 완료] {info['name']}: +{count}장, 누적 {len(new_features)}장")
                return jsonify(success=True, name=info["name"], count=count,
                               message=f"기존 사용자에 {count}장 추가 완료 (총 {len(new_features)}장). 메인 서버를 재시작해야 반영됩니다.")

            folder = FACE_DIR / info["label"]
            folder.mkdir(parents=True, exist_ok=True)
            for index, face in enumerate(info["photos"], 1):
                if not cv2.imwrite(str(folder / f"{index:03d}.jpg"), face):
                    raise RuntimeError("얼굴 사진 저장 실패")
            feature_path = folder / "features.npz"
            np.savez_compressed(feature_path, features=np.stack(info["features"]))
            with get_connection() as conn:
                conn.execute("""
                    INSERT INTO registered_users (name, face_label, face_data_path, is_active, registered_at)
                    VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                """, (info["name"], info["label"], str(feature_path.relative_to(BASE_DIR))))
                conn.commit()
            count = len(info["photos"])
            registration = None
            return jsonify(success=True, name=info["name"], count=count,
                           message=f"{info['name']}님 신규 등록 완료 ({count}장). 메인 서버를 재시작해야 반영됩니다.")
        except Exception as exc:
            return jsonify(success=False, message=str(exc)), 500


# 등록 취소

@app.route("/api/cancel", methods=["POST"])
def cancel_registration():
    global registration

    data = request.get_json(silent=True) or {}
    token = data.get("token")

    with registration_lock:
        if registration is None or registration["token"] != token:
            return jsonify({
                "success": False,
                "message": "취소할 등록이 없습니다."
            }), 400

        registration = None

    return jsonify({
        "success": True,
        "message": "등록이 취소됐습니다."
    })


# 웹페이지

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Guard Home - 얼굴 등록</title>

<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    background: #f2f7fc;
    font-family: 'Segoe UI','Malgun Gothic',sans-serif;
    color: #29445a;
}

.container {
    max-width: 780px;
    margin: 30px auto;
    padding: 16px;
}

.header {
    text-align: center;
    margin-bottom: 23px;
}

.header h1 {
    margin: 0 0 8px;
    font-size: 27px;
}

.header span { color: #529fda; }

.header p {
    color: #8196a8;
    font-size: 13px;
}

.card {
    padding: 24px;
    background: white;
    border-radius: 20px;
    box-shadow: 0 8px 30px #244d7514;
}

.video {
    overflow: hidden;
    border-radius: 14px;
    background: #172b3d;
}

.video img {
    width: 100%;
    display: block;
    aspect-ratio: 4/3;
    object-fit: contain;
}

.field { margin-top: 22px; }

label {
    display: block;
    font-weight: bold;
    margin-bottom: 9px;
    font-size: 14px;
}

input {
    width: 100%;
    padding: 15px;
    border: 1px solid #d8e6f1;
    border-radius: 12px;
    font-size: 15px;
}

button {
    border: 0;
    padding: 15px;
    border-radius: 12px;
    cursor: pointer;
    font-weight: bold;
    font-size: 14px;
}

button:disabled {
    opacity: .45;
    cursor: not-allowed;
}

.primary {
    background: #529fda;
    color: white;
}

.light {
    background: #e9f4ff;
    color: #397fbe;
}

.success {
    background: #e4f8ec;
    color: #228a53;
}

.cancel {
    background: #fff0f0;
    color: #be5260;
}

.full {
    width: 100%;
    margin-top: 13px;
}

.controls {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 12px;
}

.progress-wrap {
    background: #f0f7fd;
    padding: 17px;
    border-radius: 13px;
    margin-top: 20px;
}

.progress-top {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    font-weight: bold;
    margin-bottom: 12px;
}

.progress {
    height: 11px;
    background: #dce9f4;
    border-radius: 20px;
    overflow: hidden;
}

.bar {
    height: 100%;
    width: 0%;
    background: #529fda;
    transition: width .2s;
}

.message {
    padding: 15px;
    margin-top: 17px;
    background: #edf6ff;
    border-radius: 12px;
    font-size: 13px;
    line-height: 1.7;
}

.tip {
    margin-top: 18px;
    color: #8799aa;
    font-size: 12px;
    text-align: center;
    line-height: 1.7;
}
</style>
</head>

<body>
<div class="container">

<div class="header">
    <h1>Smart <span>Guard</span> Home</h1>
    <p>얼굴 등록 및 특징 학습</p>
</div>

<div class="card">

    <div class="video">
        <img src="/video_feed" alt="USB 카메라">
    </div>

    <div class="field">
        <label for="mode">등록 방식</label>
        <select id="mode" onchange="switchMode()" style="width:100%;padding:14px;border:1px solid #d8e6f1;border-radius:12px">
            <option value="add" selected>기존 등록자 사진 보강</option>
            <option value="new">새 사용자 등록</option>
        </select>
    </div>
    <div class="field" id="existingField">
        <label for="userId">기존 등록자</label>
        <select id="userId" style="width:100%;padding:14px;border:1px solid #d8e6f1;border-radius:12px"></select>
    </div>
    <div class="field" id="newField" style="display:none">
        <label for="name">사용자 이름</label>
        <input id="name"
               placeholder="등록할 이름을 입력하세요"
               maxlength="50">
    </div>

    <button id="start" class="primary full"
            onclick="startRegistration()">
        얼굴 등록 시작
    </button>

    <div class="progress-wrap">
        <div class="progress-top">
            <span>얼굴 사진 수집</span>
            <span id="count">0 / 20장</span>
        </div>
        <div class="progress">
            <div class="bar" id="bar"></div>
        </div>
    </div>

    <div class="controls">
        <button id="capture" class="light"
                onclick="captureFace()" disabled>
            얼굴 촬영
        </button>

        <button id="finish" class="success"
                onclick="finishRegistration()" disabled>
            학습 완료
        </button>
    </div>

    <button id="cancel" class="cancel full"
            onclick="cancelRegistration()" disabled>
        등록 취소
    </button>

    <div class="message" id="message">
        이름을 입력하고 얼굴 등록을 시작해주세요.
    </div>

    <div class="tip">
        정면, 좌측, 우측 등 얼굴 각도를 조금씩 변경하며 촬영하세요.
        <br>
        한 사람씩 촬영하고 얼굴을 가리지 마세요.
    </div>

</div>
</div>

<script>
let token = null;
let count = 0;
let mode = "add";

const $ = id => document.getElementById(id);

function message(text) {
    $("message").textContent = text;
}

function progress() {
    $("count").textContent = count + " / 20장";
    $("bar").style.width = (count / 20 * 100) + "%";

    $("capture").disabled = !token || count >= 20;
    $("finish").disabled = !token || count < (mode === "add" ? 5 : 15);
}

async function post(url, data) {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
        throw new Error(result.message || "오류 발생");
    }

    return result;
}

async function startRegistration() {
    const name = $("name").value.trim();

    if (mode === "new" && !name) {
        message("이름을 입력해주세요.");
        return;
    }

    $("start").disabled = true;

    try {
        const result = await post("/api/start", {
            name, mode, user_id: mode === "add" ? $("userId").value : null
        });

        token = result.token;
        count = 0;

        $("name").disabled = true;
        $("mode").disabled = true;
        $("userId").disabled = true;
        $("cancel").disabled = false;

        progress();
        message(result.message);

    } catch (error) {
        message(error.message);
        $("start").disabled = false;
    }
}

async function captureFace() {
    $("capture").disabled = true;

    try {
        const result = await post(
            "/api/capture",
            {token}
        );

        count = result.count;
        message(
            count + "장 촬영 완료. 얼굴 각도를 변경해주세요."
        );

    } catch (error) {
        message(error.message);

    } finally {
        progress();
    }
}

async function finishRegistration() {
    $("finish").disabled = true;
    $("capture").disabled = true;

    message("얼굴 특징 데이터 저장 중입니다...");

    try {
        const result = await post(
            "/api/finish",
            {token}
        );

        message(result.message);

        token = null;
        count = 0;

        $("name").disabled = false;
        $("mode").disabled = false;
        $("userId").disabled = false;
        loadUsers();
        $("name").value = "";
        $("start").disabled = false;
        $("cancel").disabled = true;

        progress();

    } catch (error) {
        message(error.message);
        progress();
    }
}

async function cancelRegistration() {
    if (!token) return;

    try {
        const result = await post(
            "/api/cancel",
            {token}
        );

        token = null;
        count = 0;

        $("name").disabled = false;
        $("mode").disabled = false;
        $("userId").disabled = false;
        $("start").disabled = false;
        $("cancel").disabled = true;

        progress();
        message(result.message);

    } catch (error) {
        message(error.message);
    }
}
function switchMode() {
    mode = $("mode").value;
    $("existingField").style.display = mode === "add" ? "block" : "none";
    $("newField").style.display = mode === "new" ? "block" : "none";
    progress();
}

async function loadUsers() {
    try {
        const response = await fetch("/api/users");
        const users = await response.json();
        $("userId").replaceChildren();
        for (const user of users) {
            const option = document.createElement("option");
            option.value = user.id;
            option.textContent = user.name + " (#" + user.id + ")";
            $("userId").append(option);
        }
        if (!users.length) {
            $("mode").value = "new";
            switchMode();
        }
    } catch (error) { message("등록 사용자 목록을 불러오지 못했습니다: " + error.message); }
}

loadUsers();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"}
    )

# 서버 실행

if __name__ == "__main__":
    init_database()

    thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )
    thread.start()

    print("[Smart Guard Home] 얼굴 등록 서버 시작")
    print("접속: http://192.168.0.206:5004")

    try:
        app.run(
            host="0.0.0.0",
            port=5004,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    finally:
        running = False
        thread.join(timeout=2)
