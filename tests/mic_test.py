import unittest
from unittest.mock import patch, MagicMock
import wave
import pyaudio
import sys
import os

# Adjust the path for importing the Microphone class
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from robot_control.mic import Microphone


class TestMicrophone(unittest.TestCase):

    @patch("robot_control.mic.pyaudio.PyAudio")
    def test_start_recording(self, mock_pyaudio):
        mock_stream = MagicMock()
        mock_pyaudio.return_value.open.return_value = mock_stream

        mic = Microphone()
        mic.start_recording()

        mock_pyaudio.return_value.open.assert_called_once_with(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
        )
        self.assertIsNotNone(mic.stream)

    @patch("robot_control.mic.pyaudio.PyAudio")
    def test_record(self, mock_pyaudio):
        mock_stream = MagicMock()
        mock_pyaudio.return_value.open.return_value = mock_stream

        mic = Microphone()
        mic.start_recording()
        mic.record(duration=1)

        # Check the number of frames recorded
        self.assertEqual(
            len(mic.frames),
            43,  # Example value based on expected buffer size and duration
        )

    @patch("robot_control.mic.pyaudio.PyAudio")
    @patch("robot_control.mic.wave.open")
    def test_stop_and_save(self, mock_wave_open, mock_pyaudio):
        mock_stream = MagicMock()
        mock_pyaudio.return_value.open.return_value = mock_stream
        mock_wave_file = MagicMock()
        mock_wave_open.return_value = mock_wave_file

        mic = Microphone()
        mic.start_recording()

        # Manually populate frames with byte data
        mic.frames = [b"audio frame data"] * 43  # 43 frames as an example

        mic.stop_and_save("test.wav")

        # Verify the stream was properly stopped and closed
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()

        # Ensure wave file was opened and data written
        mock_wave_open.assert_called_once_with("test.wav", "wb")
        mock_wave_file.writeframes.assert_called_once_with(b"".join(mic.frames))
        mock_wave_file.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
