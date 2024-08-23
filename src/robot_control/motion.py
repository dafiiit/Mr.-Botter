try:
    import RPi.GPIO as GPIO
except:
    import sys
    import os
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
    )
    from tests.mock_GPIO import MockGPIO as GPIO
import time


class MotionDetector:
    def __init__(self, pir_pin=4):
        """
        Initialisiert den Bewegungsmelder mit dem gegebenen GPIO-Pin.

        :param pir_pin: Der GPIO-Pin, an dem der SR501 PIR Sensor angeschlossen ist.
        """
        self.pir_pin = pir_pin
        self._setup()

    def _setup(self):
        """
        Richtet den GPIO-Pin für den PIR-Sensor ein.
        """
        GPIO.setmode(GPIO.BCM)  # BCM-Modus verwenden
        GPIO.setup(self.pir_pin, GPIO.IN)  # Setzt den Pin als Eingang

    def detect_movement(self):
        """
        Überwacht den PIR-Sensor und gibt True zurück, wenn eine Bewegung erkannt wird.

        :return: Boolean, ob Bewegung erkannt wurde.
        """
        movement_detected = GPIO.input(self.pir_pin)  # Bewegungszustand abfragen
        if movement_detected:
            print("Bewegung erkannt!")
        return movement_detected

    def cleanup(self):
        """
        Führt eine Bereinigung durch, um sicherzustellen, dass der GPIO-Pin freigegeben wird.
        """
        GPIO.cleanup()


# Beispielnutzung
if __name__ == "__main__":
    try:
        motion_detector = MotionDetector(pir_pin=4)
        while True:
            if motion_detector.detect_movement():
                # Hier kannst du weitere Aktionen ausführen, wenn Bewegung erkannt wird
                time.sleep(1)  # Kurze Pause, um Mehrfacherkennungen zu vermeiden
    except KeyboardInterrupt:
        print("Programm beendet.")
    finally:
        motion_detector.cleanup()
