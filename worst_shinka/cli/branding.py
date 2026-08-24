from __future__ import annotations

import os
import shutil
import sys
from typing import TextIO

CLR_BOLD = "\033[1m"
CLR_RESET = "\033[0m"

COLOR_STOPS: list[tuple[int, int, int]] = [
    (0, 240, 255),   # Cyan
    (80, 100, 255),  # Blue
    (240, 50, 240),  # Magenta
]

WORST_LINES = [
    r"██╗   ██╗    ██╗  ██████╗  ██████╗   ███████╗ ████████╗",
    r"██║   ██║    ██║ ██╔═══██╗ ██╔══██╗  ██╔════╝ ╚══██╔══╝",
    r"██║   ██║    ██║ ██║   ██║ ██████╔╝  ███████╗    ██║   ",
    r"██║   ██║    ██║ ██║   ██║ ██╔══██╗  ╚════██║    ██║   ",
    r"╚██████╔██████╔╝ ╚██████╔╝ ██║  ██║  ███████║    ██║   ",
    r" ╚═════╝ ╚════╝   ╚═════╝  ╚═╝  ╚═╝  ╚══════╝    ╚═╝   ",
]

SUBTITLE_WORD = "     S H I N K A     "

def _interpolate_color(
    start_rgb: tuple[int, int, int], end_rgb: tuple[int, int, int], factor: float
) -> tuple[int, int, int]:
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * factor)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * factor)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * factor)
    return r, g, b


def _get_gradient_rgb(ratio: float) -> tuple[int, int, int]:
    ratio = max(0.0, min(1.0, ratio))
    num_segments = len(COLOR_STOPS) - 1
    segment = min(int(ratio * num_segments), num_segments - 1)

    seg_ratio = (ratio - (segment / num_segments)) * num_segments
    return _interpolate_color(COLOR_STOPS[segment], COLOR_STOPS[segment + 1], seg_ratio)


def _colorize_gradient(text: str, bold: bool = True) -> str:
    if not text:
        return ""

    length = len(text)
    colored_chars: list[str] = []
    bold_code = CLR_BOLD if bold else ""

    for i, char in enumerate(text):
        ratio = i / (length - 1) if length > 1 else 0.0
        r, g, b = _get_gradient_rgb(ratio)
        colored_chars.append(f"\033[38;2;{r};{g};{b}m{bold_code}{char}")

    return "".join(colored_chars) + CLR_RESET


def _generate_logo(colored: bool = True, term_width: int = 80) -> str:
    output: list[str] = []

    top_line_raw = "─" * term_width

    if colored:
        output.append(_colorize_gradient(top_line_raw, bold=True))
    else:
        output.append(top_line_raw)

    output.append("")

    max_logo_width = max(len(line) for line in WORST_LINES)
    worst_indent = " " * max(0, (term_width - max_logo_width) // 2)

    num_lines = len(WORST_LINES)
    for idx, line in enumerate(WORST_LINES):
        if colored:
            ratio = idx / (num_lines - 1) if num_lines > 1 else 0.0
            r, g, b = _get_gradient_rgb(ratio)
            output.append(f"{worst_indent}\033[38;2;{r};{g};{b}m{CLR_BOLD}{line}{CLR_RESET}")
        else:
            output.append(f"{worst_indent}{line}")

    output.append("")
    total_line_len = max(0, term_width - len(SUBTITLE_WORD))
    left_len = total_line_len // 2
    right_len = total_line_len - left_len

    bottom_line_raw = f"{'─' * left_len}{SUBTITLE_WORD}{'─' * right_len}"

    if colored:
        output.append(_colorize_gradient(bottom_line_raw, bold=True))
    else:
        output.append(bottom_line_raw)

    output.append("\n")
    return "\n".join(output)


def print_logo(file: TextIO | None = None) -> None:
    target_file = file if file is not None else sys.stdout

    is_tty = getattr(target_file, "isatty", lambda: False)()
    no_color = bool(os.getenv("NO_COLOR"))
    colored = is_tty and not no_color

    term_width = shutil.get_terminal_size((80, 20)).columns if is_tty else 80

    logo_text = _generate_logo(colored=colored, term_width=term_width)
    target_file.write(logo_text)
    target_file.flush()

RACKET_LINES = [
r"                               ▄▄██████▄▄      ",
r"                            ▄██▀░░░░░░░░▀██▄   ",
r"                         ▄██▀░░░░░░░░░░░░▀██▄  ",
r"    ▄████▄             ▄██▀░░░░░░░░░░░░░░░░██  ",
r"   ██░░▒▒██           ██▀░░░░░░░░░░░░░░░░░▄██  ",
r"   ██▒▒░░██          ██░░░░░░░░░░░░░░░░░░▄██   ",
r"    ▀████▀          ██░░░░░░░░░░░░░░░░░░▄██    ",
r"                   ██░░░░░░░░░░░░░░░░░░▄██     ",
r"                  ██░░░░░░░░░░░░░░░░░▄██       ",
r"                  ██░░░░░░░░░░░░░░░▄██▀        ",
r"                  ▀█▄░░░░░░░░░░░▄██▀           ",
r"                   ▀████████████▀              ",
r"                    ▀██  ▄██▀                  ",
r"                     ██▄██▀                    ",
r"                    ▄██▀                       ",
r"                  ▄██▀                         ",
r"                 ▄██▀                          ",
r"                ▄██                            ",
r"               ██▀                             ",
r"              ██▀                              ",
]

def _generate_racket(colored: bool = True, term_width: int = 80) -> str:
    output: list[str] = []
    max_racket_width = max(len(line) for line in RACKET_LINES)
    indent = " " * max(0, (term_width - max_racket_width) // 2)
    num_lines = len(RACKET_LINES)

    for idx, line in enumerate(RACKET_LINES):
        if colored:
            ratio = idx / (num_lines - 1) if num_lines > 1 else 0.0
            r, g, b = _get_gradient_rgb(ratio)
            output.append(f"{indent}\033[38;2;{r};{g};{b}m{CLR_BOLD}{line}{CLR_RESET}")
        else:
            output.append(f"{indent}{line}")

    return "\n".join(output)


def print_racket(file: TextIO | None = None) -> None:
    target_file = file if file is not None else sys.stdout

    is_tty = getattr(target_file, "isatty", lambda: False)()
    no_color = bool(os.getenv("NO_COLOR"))
    colored = is_tty and not no_color

    term_width = shutil.get_terminal_size((80, 20)).columns if is_tty else 80

    racket_text = _generate_racket(colored=colored, term_width=term_width)
    target_file.write(racket_text + "\n")
    target_file.flush()