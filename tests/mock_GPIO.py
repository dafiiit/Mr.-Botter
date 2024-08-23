class MockGPIO:
    BCM = "BCM"
    BOARD = "BOARD"
    IN = "IN"
    OUT = "OUT"
    HIGH = 1
    LOW = 0

    def __init__(self):
        self.mode = None
        self.pins = {}

    def setmode(self, mode):
        print(f"Set mode to {mode}")
        self.mode = mode

    def setup(self, pin, mode, initial=None):
        print(f"Setting up pin {pin} as {'OUTPUT' if mode == self.OUT else 'INPUT'}")
        self.pins[pin] = {"mode": mode, "state": initial}
        if initial is not None:
            self.output(pin, initial)

    def output(self, pin, state):
        if pin in self.pins and self.pins[pin]["mode"] == self.OUT:
            print(f"Setting pin {pin} to {'HIGH' if state else 'LOW'}")
            self.pins[pin]["state"] = state
        else:
            print(f"Error: Pin {pin} is not set up as an output")

    def input(self, pin):
        if pin in self.pins and self.pins[pin]["mode"] == self.IN:
            print(f"Reading pin {pin}")
            return self.pins[pin].get("state", self.LOW)
        else:
            print(f"Error: Pin {pin} is not set up as an input")
            return self.LOW

    def cleanup(self):
        print("Cleaning up GPIO")
        self.pins = {}

# Beispiel für die Verwendung der Mock-Klasse
#GPIO = MockGPIO()
