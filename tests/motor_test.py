import unittest
from unittest.mock import patch, MagicMock
from Mock.GPIO import GPIO
# Mocking import


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from robot_control.motors import Servo

class TestServo(unittest.TestCase):

    @patch('RPi.GPIO.setup')
    @patch('RPi.GPIO.PWM')
    @patch('RPi.GPIO.setmode')
    def test_initialize(self, mock_setmode, mock_pwm, mock_setup):
        servo = Servo(pin=17)
        mock_setmode.assert_called_once_with(GPIO.BCM)
        mock_setup.assert_called_once_with(17, GPIO.OUT)
        mock_pwm.assert_called_once_with(17, 50)
        self.assertIsNotNone(servo.pwm)

    @patch('RPi.GPIO.PWM')
    def test_set_angle(self, mock_pwm):
        mock_pwm_instance = MagicMock()
        mock_pwm.return_value = mock_pwm_instance
        
        servo = Servo(pin=17)
        servo.set_angle(90)
        
        expected_duty_cycle = servo._angle_to_duty_cycle(90)
        mock_pwm_instance.ChangeDutyCycle.assert_called_with(expected_duty_cycle)

    @patch('RPi.GPIO.cleanup')
    @patch('RPi.GPIO.PWM')
    def test_stop(self, mock_pwm, mock_cleanup):
        mock_pwm_instance = MagicMock()
        mock_pwm.return_value = mock_pwm_instance
        
        servo = Servo(pin=17)
        servo.stop()
        
        mock_pwm_instance.stop.assert_called_once()
        mock_cleanup.assert_called_once()

if __name__ == '__main__':
    unittest.main()