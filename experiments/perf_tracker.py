import os
import signal
import subprocess
import time


class PerfTracker:
    """Tracks Linux hardware perf events for the current Python process."""

    def __init__(self, events: list[str]):
        self.events = events
        self.process = None
        self.measurements: dict[str, int | None] = {}

    def __enter__(self):
        # -p: attach to the current Python process ID
        # -e: comma-separated list of hardware events
        # -x: output in CSV format with ',' as delimiter for easy parsing
        cmd = [
            "perf",
            "stat",
            "-p",
            str(os.getpid()),
            "-e",
            ",".join(self.events),
            "-x",
            ",",
        ]

        # perf stat writes its results to stderr
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )

        # CRITICAL QUIRK: `perf` takes a few milliseconds to wake up and attach
        # to the PID. If we don't sleep briefly, the math will start before
        # perf is actually counting!
        time.sleep(0.10)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.send_signal(signal.SIGINT)
        _, stderr = self.process.communicate()

        for line in stderr.strip().split("\n"):
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                value_str = parts[0].strip()
                event_name = parts[2].strip()

                try:
                    self.measurements[event_name] = int(float(value_str))
                except ValueError:
                    self.measurements[event_name] = None
                    # --- NEW DEBUG LINE ---
                    print(
                        f"[DEBUG] Perf missed {event_name}. Raw output: '{value_str}'"
                    )
