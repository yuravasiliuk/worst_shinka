from __future__ import annotations

import os
import sys
import logging
import time
import multiprocessing
from pathlib import Path
from .branding import print_logo, print_racket
from .integrations import play_candidate
from .terminal import print_play_start

log = logging.getLogger(__name__)

def _quit_pressed() -> bool:
    if not sys.stdin.isatty():
        return False
    import select

    readable, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(readable) and sys.stdin.read(1).lower() == "q"

def run_game(model_path: Path) -> int:
    process = multiprocessing.Process(target = play_candidate, kwargs={"model_path": str(model_path)}, name="worst-shinka-atari-tennis")
    process.start()

    print_logo()
    
    print_play_start(model_path = model_path, process_id = process.pid)
    print_racket()
    log.info("Atari Tennis started (PID %s)\n", process.pid)
    log.info("Press Q to quit.")

    stopped_by_user = False

    try:
        while process.is_alive():
            if _quit_pressed():
                stopped_by_user = True
                process.terminate()
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        stopped_by_user = True
        process.terminate()
    finally:
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join()

    if stopped_by_user:
        log.info("Atari Tenis gamme has been stopped by an user.")
        return 0
    exit_code = process.exitcode or 0
    if exit_code == 0:
        log.info("Game finished")
    else:
        log.info("Atari Tennis Proces exited with code %s.", exit_code)
    return exit_code