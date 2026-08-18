from __future__ import annotations

import os
import sys
from typing import TextIO

CLR_CYAN = "\033[1;36m"
CLR_BLUE = "\033[1;34m"
CLR_MAGENTA = "\033[1;35m"
CLR_BOLD = "\033[1m"
CLR_RESET = "\033[0m"

LOGO = (
    f"{CLR_CYAN} __        _____  ____  ____ _____ {CLR_RESET}\n"
    f"{CLR_CYAN} \\ \\      / / _ \\|  _ \\/ ___|_   _|{CLR_RESET}\n"
    f"{CLR_BLUE}  \\ \\ /\\ / / | | | |_) \\___ \\ | |  {CLR_RESET}\n"
    f"{CLR_MAGENTA}   \\ V  V /| |_| |  _ < ___) || |  {CLR_RESET}\n"
    f"{CLR_MAGENTA}    \\_/\\_/  \\___/|_| \\_\\____/ |_|  {CLR_RESET}\n\n"
    f"{CLR_BOLD}{CLR_MAGENTA}         ───  S H I N K A  ───{CLR_RESET}\n"
)

_PLAIN_LOGO = (
    " __        _____  ____  ____ _____\n"
    " \\ \\      / / _ \\|  _ \\/ ___|_   _|\n"
    "  \\ \\ /\\ / / | | | |_) \\___ \\ | |\n"
    "   \\ V  V /| |_| |  _ < ___) || |\n"
    "    \\_/\\_/  \\___/|_| \\_\\____/ |_|\n\n"
    "         ───  S H I N K A  ───\n"
)


def print_logo(file: TextIO | None = None) -> None:
    """Wypisuje kolorowe logo WORST-SHINKA w terminalu."""
    target_file = file if file is not None else sys.stdout

    is_tty = getattr(target_file, "isatty", lambda: False)()
    no_color = bool(os.getenv("NO_COLOR"))

    output_logo = LOGO if (is_tty and not no_color) else _PLAIN_LOGO
    target_file.write(output_logo)
    target_file.flush()