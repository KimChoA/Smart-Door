from threading import Thread, Event, Lock
import time

RGB_R_PIN = 17
RGB_G_PIN = 27
RGB_B_PIN = 22

TRIGGER_PIN = 23
ECHO_PIN = 24

WARNING_DISTANCE_CM = 20.0
EXIT_DISTANCE_CM = 25.0
EXIT_SECONDS = 2.0
DETECTION_SECONDS = 10.0
CHECK_INTERVAL = 0.1
BLINK_INTERVAL = 0.3


class SecurityMonitor:
    def __init__(
        self,
        enabled=False,
        on_warning=None,
        is_security_enabled=None,
        is_known_face=None,
        on_approach=None
    ):
        self.enabled = enabled
        self.on_warning = on_warning
        self.is_security_enabled = is_security_enabled or (lambda: True)
        self.is_known_face = is_known_face or (lambda: False)
        self.on_approach = on_approach or (lambda active: None)

        self.rgb = None
        self.sensor = None

        self.stop_event = Event()
        self.thread = None

        self.door_open = False
        self.warning_active = False
        self.status_lock = Lock()

        self.current_color = None

    def initialize(self):
        if not self.enabled:
            print("[보안 GPIO] 시뮬레이션 모드")
            return

        from gpiozero import Device, RGBLED, DistanceSensor
        from gpiozero.pins.lgpio import LGPIOFactory

        Device.pin_factory = LGPIOFactory()

        try:
            self.rgb = RGBLED(
                red=RGB_R_PIN,
                green=RGB_G_PIN,
                blue=RGB_B_PIN,
                active_high=True,
                initial_value=(0, 0, 0)
            )

            self.sensor = DistanceSensor(
                echo=ECHO_PIN,
                trigger=TRIGGER_PIN,
                max_distance=2.0
            )

            self.set_color("blue")

            print("[보안 GPIO] 초기화 완료")
            print("[초음파] TRIG 23 / ECHO 24")
            print("[RGB LED] R17 / G27 / B22")

        except Exception:
            self.close()
            raise

    def set_color(self, color):
        if self.rgb is None:
            return

        if self.current_color == color:
            return

        colors = {
            "red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "off": (0, 0, 0)
        }

        self.rgb.color = colors.get(color, (0, 0, 0))
        self.current_color = color

    def set_door_open(self, opened):
        with self.status_lock:
            self.door_open = bool(opened)

    def start(self):
        if not self.enabled:
            return

        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event.clear()

        self.thread = Thread(
            target=self.monitor,
            daemon=True,
            name="security-monitor"
        )
        self.thread.start()

    def notify_approach(self, active):
        try:
            self.on_approach(active)
        except Exception as exc:
            print("[접근 콜백 오류]", exc)

    def notify_warning(self, distance):
        if self.on_warning is None:
            return

        def worker():
            try:
                self.on_warning(distance)
            except Exception as exc:
                print("[보안 경고 처리 오류]", exc)

        Thread(
            target=worker,
            daemon=True,
            name="security-warning"
        ).start()

    def monitor(self):
        detected_since = None
        exit_since = None

        approach_active = False
        warning_saved = False

        blink_on = False
        last_blink = 0.0
        last_debug = 0.0

        while not self.stop_event.wait(CHECK_INTERVAL):
            try:
                now = time.monotonic()
                distance = self.sensor.distance * 100

                with self.status_lock:
                    door_open = self.door_open

                if not self.is_security_enabled():
                    if approach_active:
                        self.notify_approach(False)

                    approach_active = False
                    detected_since = None
                    exit_since = None
                    warning_saved = False
                    self.warning_active = False

                    self.set_color("off")
                    continue

                is_near = 0 < distance <= WARNING_DISTANCE_CM

                # 접근 시작
                if not approach_active and is_near:
                    approach_active = True
                    detected_since = now
                    exit_since = None
                    warning_saved = False
                    self.warning_active = False

                    print(
                        f"[초음파] 접근 감지: {distance:.1f}cm"
                    )

                    self.notify_approach(True)

                # 접근 종료 확인
                elif approach_active:
                    if distance > EXIT_DISTANCE_CM:
                        if exit_since is None:
                            exit_since = now

                        elif now - exit_since >= EXIT_SECONDS:
                            approach_active = False
                            detected_since = None
                            exit_since = None
                            warning_saved = False
                            self.warning_active = False
                            blink_on = False

                            print("[초음파] 접근 종료")
                            self.notify_approach(False)

                    else:
                        exit_since = None

                # 등록 사용자 인증 상태
                try:
                    known = bool(self.is_known_face())
                except Exception as exc:
                    print("[얼굴 인증 상태 오류]", exc)
                    known = False

                # 일정 간격으로 센서 상태 출력
                if now - last_debug >= 1.0:
                    elapsed = (
                        now - detected_since
                        if detected_since is not None
                        else 0.0
                    )

                    print(
                        f"[보안 상태] 거리={distance:.1f}cm "
                        f"접근={approach_active} "
                        f"인증={known} "
                        f"문열림={door_open} "
                        f"경고={self.warning_active} "
                        f"경과={elapsed:.1f}초"
                    )

                    last_debug = now

                # 문 열림 또는 등록자 인증 성공
                if door_open or (approach_active and known):
                    blink_on = False
                    self.set_color("green")
                    continue

                # 한 번 발생한 경고는 접근 종료까지 유지
                if self.warning_active:
                    if now - last_blink >= BLINK_INTERVAL:
                        blink_on = not blink_on
                        self.set_color(
                            "red" if blink_on else "off"
                        )
                        last_blink = now
                    continue

                # 10초 이상 근접 시 경고
                if (
                    approach_active
                    and is_near
                    and detected_since is not None
                    and now - detected_since >= DETECTION_SECONDS
                ):
                    self.warning_active = True

                    if not warning_saved:
                        warning_saved = True

                        print(
                            f"[보안 경고] 미인증 접근 "
                            f"{distance:.1f}cm / 10초 경과"
                        )

                        blink_on = True
                        last_blink = now

                        self.set_color("red")
                        self.notify_warning(distance)

                else:
                    self.set_color("blue")

            except Exception as exc:
                print("[보안 모니터링 오류]", exc)
                self.stop_event.wait(0.5)

    def close(self):
        self.stop_event.set()

        if (
            self.thread is not None
            and self.thread.is_alive()
            and self.thread is not __import__("threading").current_thread()
        ):
            self.thread.join(timeout=1.0)

        for device in (self.sensor, self.rgb):
            if device is not None:
                try:
                    device.close()
                except Exception as exc:
                    print("[보안 GPIO 정리 오류]", exc)

        self.sensor = None
        self.rgb = None
        self.current_color = None