import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from robot_control.motion import MotionDetector


class TestMotionDetector(unittest.TestCase):
    
    @patch("robot_control.motion.GPIO")
    def test_detect_movement(self, mock_gpio):
        # Simuliere die GPIO-Eingabe für Bewegungserkennung (Bewegung erkannt: 1, keine Bewegung: 0)
        mock_gpio.input.return_value = 1  # Bewegung erkannt

        detector = MotionDetector(pir_pin=4)
        self.assertTrue(
            detector.detect_movement()
        )  # Test sollte True zurückgeben, wenn Bewegung erkannt wird

        mock_gpio.input.return_value = 0  # Keine Bewegung
        self.assertFalse(
            detector.detect_movement()
        )  # Test sollte False zurückgeben, wenn keine Bewegung erkannt wird

    @patch("robot_control.motion.GPIO")
    def test_cleanup(self, mock_gpio):
        detector = MotionDetector(pir_pin=4)
        detector.cleanup()
        mock_gpio.cleanup.assert_called_once()  # Überprüfe, ob cleanup() genau einmal aufgerufen wurde


if __name__ == "__main__":
    unittest.main()
