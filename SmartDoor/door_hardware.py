
import os
from threading import Lock, Thread, Event
from time import sleep

SERVO_PIN = 13
HOUSE_SWITCH_PIN = 21
DOORBELL_PIN = 20
BUZZER_PIN = 14

# 현재 정상 동작한다고 공유된 값 그대로 유지
SERVO_MIN_PULSE = 0.5 / 1000
SERVO_MAX_PULSE = 2.5 / 1000
LOCK_POSITION = -0.5
UNLOCK_POSITION = 0.5
SERVO_MOVE_SECONDS = 0.3
OPEN_SECONDS = 2.0
ENABLE_JITTER_REDUCTION = True

class DoorHardware:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.servo = None
        self.house_switch = None
        self.doorbell = None
        self.buzzer = None
        self.bell_lock = Lock()
        self.current_position = None
        self.servo_lock = Lock()
        self.warning_stop = Event()
        self.warning_thread = None
        self.warning_lock = Lock()

    def initialize(self, on_indoor_open, on_doorbell):
        if not self.enabled:
            print('[도어락 GPIO] 시뮬레이션 모드')
            return
        from gpiozero import Device, Servo, Button, TonalBuzzer
        from gpiozero.pins.lgpio import LGPIOFactory

        Device.pin_factory = LGPIOFactory()
        try:
            self.servo = Servo(
                SERVO_PIN, min_pulse_width=SERVO_MIN_PULSE,
                max_pulse_width=SERVO_MAX_PULSE, initial_value=None
            )
            self.house_switch = Button(HOUSE_SWITCH_PIN, pull_up=False, bounce_time=0.1)
            self.doorbell = Button(DOORBELL_PIN, pull_up=False, bounce_time=0.1)
            self.buzzer = TonalBuzzer(BUZZER_PIN)

            # 초기 위치로 이동 후 PDF 예제대로 PWM 정지
            self.move(LOCK_POSITION)
            self.house_switch.when_pressed = on_indoor_open
            self.doorbell.when_pressed = on_doorbell
            print('[도어락 GPIO] 서보 13, 실내 스위치 21, 초인종 20, 부저 14')
            print('[서보] 지터 감소:', 'ON' if ENABLE_JITTER_REDUCTION else 'OFF')
        except Exception:
            self.close()
            raise

    
    
    
    def move(self, position):
        if not self.enabled:
            return

        if self.servo is None:
            raise RuntimeError("서보가 초기화되지 않았습니다.")

        self.servo.value = position
        sleep(0.3)
        self.servo.value = None

        print(f"[서보] 이동 완료: {position}, PWM OFF")




    def start_warning_buzzer(self):
        """미인증 접근 경고음. 초인종과 같은 GPIO14를 동시 제어하지 않는다."""
        if not self.enabled or self.buzzer is None:
            return
        with self.warning_lock:
            if self.warning_thread is not None and self.warning_thread.is_alive():
                return
            self.warning_stop.clear()
            self.warning_thread = Thread(
                target=self._warning_worker, daemon=True, name='warning-buzzer'
            )
            self.warning_thread.start()

    def _warning_worker(self):
        # 초인종이 먼저 재생 중이면 종료를 기다린 뒤 경고음을 시작한다.
        with self.bell_lock:
            try:
                from gpiozero.tones import Tone
                while not self.warning_stop.is_set():
                    self.buzzer.play(Tone('A5'))
                    if self.warning_stop.wait(0.3):
                        break
                    self.buzzer.stop()
                    if self.warning_stop.wait(0.3):
                        break
            except Exception as exc:
                print('[보안 부저 오류]', exc)
            finally:
                try:
                    self.buzzer.stop()
                except Exception as exc:
                    print('[보안 부저 정지 오류]', exc)

    def stop_warning_buzzer(self):
        # 긴 대기나 join으로 카메라/초음파/서보 스레드를 막지 않는다.
        self.warning_stop.set()

    def play_doorbell(self):
        if not self.enabled or self.buzzer is None:
            return
        if not self.bell_lock.acquire(blocking=False):
            return
        try:
            from gpiozero.tones import Tone
            for note, duration in [('C4', .20), ('E4', .20), ('G4', .20), ('C5', .45)]:
                self.buzzer.play(Tone(note))
                sleep(duration)
                self.buzzer.stop()
                sleep(.05)
        finally:
            try:
                self.buzzer.stop()
            finally:
                self.bell_lock.release()

    def close(self):
        self.stop_warning_buzzer()
        if self.warning_thread is not None and self.warning_thread.is_alive():
            self.warning_thread.join(timeout=2.0)
        for device in (self.house_switch, self.doorbell, self.buzzer, self.servo):
            if device is not None:
                try:
                    device.close()
                except Exception as exc:
                    print('[도어락 GPIO 정리 오류]', exc)
