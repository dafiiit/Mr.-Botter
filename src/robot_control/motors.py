try:
    import RPi.GPIO as GPIO
except ModuleNotFoundError:
    from Mock.GPIO import GPIO as GPIO
import time

class Servo:
    def __init__(self, pin, min_angle=0, max_angle=180, frequency=50):
        self.pin = pin
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.frequency = frequency
        self.pwm = None
        self._initialize()

    def _initialize(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, self.frequency)
        self.pwm.start(0)

    def _angle_to_duty_cycle(self, angle):
        duty_cycle = 2.5 + (12.0 * angle / 180.0)
        return duty_cycle

    def set_angle(self, angle):
        if angle < self.min_angle or angle > self.max_angle:
            raise ValueError(f"Winkel muss zwischen {self.min_angle} und {self.max_angle} Grad liegen.")
        duty_cycle = self._angle_to_duty_cycle(angle)
        self.pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # Zeit geben, damit sich der Servo bewegen kann
        self.pwm.ChangeDutyCycle(0)  # Stoppt das Senden des Signals

    def stop(self):
        self.pwm.stop()
        GPIO.cleanup()

    def __del__(self):
        self.stop()
        

if __name__ == "__main__":
    
    # Erstelle Servo-Instanzen für mehrere Pins
    servo1 = Servo(pin=17)
    servo2 = Servo(pin=18)
    servo3 = Servo(pin=27)

    try:
        # Setze den Winkel des ersten Servomotors auf 90 Grad
        print("Servo 1 auf 90 Grad einstellen.")
        servo1.set_angle(90)
        time.sleep(1)

        # Setze den Winkel des zweiten Servomotors auf 45 Grad
        print("Servo 2 auf 45 Grad einstellen.")
        servo2.set_angle(45)
        time.sleep(1)

        # Setze den Winkel des dritten Servomotors auf 135 Grad
        print("Servo 3 auf 135 Grad einstellen.")
        servo3.set_angle(135)
        time.sleep(1)

        # Zurücksetzen der Servos auf 0 Grad
        print("Zurücksetzen aller Servos auf 0 Grad.")
        servo1.set_angle(0)
        servo2.set_angle(0)
        servo3.set_angle(0)
        time.sleep(1)

    finally:
        # Beende PWM und räume GPIO-Pins auf
        servo1.stop()
        servo2.stop()
        servo3.stop()
        print("Servos gestoppt und GPIO bereinigt.")