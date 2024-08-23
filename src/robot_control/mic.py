import pyaudio
import wave


class Microphone:
    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self):
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )
        self.frames = []

    def record(self, duration):
        if self.stream is None:
            raise RuntimeError("Recording has not been started.")
        print("Recording...")
        for _ in range(0, int(self.rate / self.frames_per_buffer * duration)):
            data = self.stream.read(self.frames_per_buffer)
            self.frames.append(data)
        print("Recording finished.")

    def stop_and_save(self, filename):
        if self.stream is None:
            raise RuntimeError("No recording to stop.")
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None

        wave_file = wave.open(filename, "wb")
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(self.rate)
        wave_file.writeframes(b"".join(self.frames))
        wave_file.close()

    def __del__(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


# Beispielnutzung
if __name__ == "__main__":
    mic = Microphone()
    mic.start_recording()
    mic.record(duration=5)  # 5 Sekunden aufnehmen
    mic.stop_and_save("output.wav")
