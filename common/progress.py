"""One-line ASCII progress (claude.md 10.1) and run.log logger."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


def _fmt_time(sec: float) -> str:
    if sec != sec or sec == float("inf"):
        return "--:--"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ProgressLine:
    """Single line updated in place with '\\r'. ASCII only."""

    def __init__(self, tag: str, total: int, unit: str = "", quiet: bool = False,
                 width: int = 20, stream=None):
        self.tag = tag
        self.total = max(1, int(total))
        self.unit = unit
        self.quiet = quiet
        self.width = width
        self.stream = stream or sys.stdout
        self.t0 = time.perf_counter()
        self._last_len = 0

    def update(self, done: int, extra: str = "") -> None:
        if self.quiet:
            return
        frac = min(1.0, done / self.total)
        n = int(round(frac * self.width))
        bar = "#" * n + "-" * (self.width - n)
        el = time.perf_counter() - self.t0
        eta = el / done * (self.total - done) if done > 0 else float("inf")
        unit = f" {self.unit}" if self.unit else ""
        line = (f"[{self.tag:<8}] {done}/{self.total}{unit} |{bar}| {100 * frac:5.1f}%  "
                f"{_fmt_time(el)}<{_fmt_time(eta)}  {extra}")
        line = line.encode("ascii", "replace").decode("ascii")
        pad = max(0, self._last_len - len(line))
        self.stream.write("\r" + line + " " * pad)
        self.stream.flush()
        self._last_len = len(line)

    def close(self, summary: str = "") -> None:
        if not self.quiet:
            self.stream.write("\n")
            if summary:
                self.stream.write(summary.encode("ascii", "replace").decode("ascii") + "\n")
            self.stream.flush()


def setup_run_logger(out_dir: Path, name: str = "run") -> logging.Logger:
    """File logger writing <out_dir>/run.log (no carriage returns)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}:{out_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger
