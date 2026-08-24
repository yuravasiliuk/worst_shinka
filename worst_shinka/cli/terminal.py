from __future__ import annotations
import logging
import os
import platform
import re
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import TextIO
from .config import RunConfig, DEFAULT_INITIAL_MODEL

RESET = "\033[0m"
PEACH = "\033[38;2;255;190;140m"
MINT = "\033[38;2;152;255;204m"
BLUE = "\033[94m"
GREEN = "\033[92m"
WHITE = "\033[97m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
PURPLE = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"

GRADIENT_BLUE = (70, 130, 255)
GRADIENT_GREEN = (57, 220, 160)

_LEVEL_COLORS = {
    logging.DEBUG: DIM,
    logging.INFO: BLUE,
    logging.WARNING: YELLOW,
    logging.ERROR: RED,
    logging.CRITICAL: RED + BOLD
}

def supports_color(stream: TextIO | None = None) -> bool:
    target = stream if stream is not None else sys.stdout
    return not os.getenv("NO_COLOR") and bool(getattr(target, "isatty", lambda: False)())

def styled(value: object, color: str, *, enabled: bool) -> str:
    text = str(value)
    return f"{color}{text}{RESET}" if enabled else text

def gradient_text(value: object, *, enabled: bool, start: tuple[int, int, int] = GRADIENT_BLUE, end: tuple[int, int, int] = GRADIENT_GREEN) -> str:
    text = str(value)
    if not enabled or not text:
        return text

    denominator = max(len(text) - 1, 1)
    result = []

    for idx, char in enumerate(text):
        ratio = idx / denominator
        red = round(start[0] + (end[0] - start[0]) * ratio)
        green = round(start[1] + (end[1] - start[1]) * ratio)
        blue = round(start[2] + (end[2] - start[2]) * ratio)

        result.append(f"\033[38;2;{red};{green};{blue}m{char}")

    return "".join(result) + RESET                

def _visible_length(value: str) -> int:
    return len(re.compile(r"\033\[[0-?]*[ -/]*[@-~]").sub("", value))

def _get_color(ratio: float, start: tuple[int, int, int], end: tuple[int, int, int]) -> str:
    r = round(start[0] + (end[0] - start[0]) * ratio)
    g = round(start[1] + (end[1] - start[1]) * ratio)
    b = round(start[2] + (end[2] - start[2]) * ratio)
    return f"\033[38;2;{r};{g};{b}m"

def _info_box(title: str, rows: list[tuple[str, str]], *, enabled: bool, width: int = 62) -> str:
    content_width = max(
        width - 2,
        _visible_length(title) + 4,
        max((_visible_length(rendered) + 2 for _, rendered in rows), default=0),
    )
    width = content_width + 2
    total_lines = len(rows) + 2
    top_right_len = max(width - _visible_length(title) - 5, 0)
    lines = []
    if enabled:
        c1 = _get_color(0.0, GRADIENT_BLUE, GRADIENT_GREEN)
        c2 = _get_color(1.0 / (total_lines - 1), GRADIENT_BLUE, GRADIENT_GREEN)
        top_left = f"{c1}┌─{RESET} "
        top_right = f" {c2}" + ("─" * top_right_len) + f"┐{RESET}"
    else:
        top_left = "┌─ "
        top_right = " " + ("─" * top_right_len) + "┐"
    lines.append(f"{top_left}{styled(title, WHITE + BOLD, enabled=enabled)}{top_right}")

    for idx, (_, rendered) in enumerate(rows, start=1):
        padding = " " * max(content_width - _visible_length(rendered) - 2, 0)
        
        if enabled:
            c = _get_color(idx / (total_lines - 1), GRADIENT_BLUE, GRADIENT_GREEN)
            side = f"{c}│{RESET}"
        else:
            side = "│"
            
        lines.append(f"{side} {rendered}{padding} {side}")

    if enabled:
        c_bot = _get_color(1.0, GRADIENT_BLUE, GRADIENT_GREEN)
        bot_left = f"{c_bot}└{RESET}"
        horizontal = f"{c_bot}" + "─" * content_width + f"{RESET}"
        bot_right = f"{c_bot}┘{RESET}"
    else:
        bot_left, horizontal, bot_right = "└", "─" * content_width, "┘"

    lines.append(f"{bot_left}{horizontal}{bot_right}")

    return "\n".join(lines)

def format_config_status(*, config_directory: Path, source: str | None, masked_key: str | None, enabled: bool | None = None) -> str:
    color = supports_color(sys.stdout) if enabled is None else enabled

    def label(value: str) -> str:
        return styled(f"{value:<19}", WHITE, enabled=color)

    def row(name: str, value: str, color_code: str = BLUE) -> tuple[str, str]:
        raw = f"{name:<19}{value}"
        colored_value = styled(value, color_code, enabled=color)

        return raw, f"{label(name)}{colored_value}"
    connected = source is not None
    status = "CONNECTED" if connected else "NOT CONNECTED"

    rows = [
        ("", ""),
        row("Status", status, (GREEN if connected else RED) + BOLD),
        row("API key", masked_key or "Not configured", BLUE if masked_key else RED+BOLD),
        row("Config directory", str(config_directory), PURPLE)
    ]
    if source:
        rows.append(row("Credential source", source, PURPLE))
    rows.extend([
        ("", ""),
        ("Default settings", styled("Default settings", WHITE + BOLD, enabled=color)),
        row("Initial model", str(DEFAULT_INITIAL_MODEL), PURPLE),
        row("Mode", "medium"),
        row("Generations", 10, BLUE),
        row("Workers", 1, BLUE),
        row("Parents", 4, BLUE),
        row("Results directory", "results", PURPLE),
        ("", "")
    ])
    if not connected:
        command = "worst-shinka config login"
        rows.extend([
            ("To connect, run", styled("To connect, run:", MINT, enabled=color)),
            (f" {command}", f"  {styled(command, MINT+BOLD, enabled=color)}"),
            ("", "")
        ])

    return _info_box("Current configuration", rows, enabled=color)

def print_login_intro(save_path: Path, *, file: TextIO | None = None) -> None:
    target = file if file is not None else sys.stdout
    color = supports_color(target)
    url = "https://openrouter.ai/settings/keys"
    rows = [
        ("", ""),
        ("An OpenRouter API key is required to run evolution.", styled("An OpenRouter API key is required to run evolution.", WHITE, enabled=color)),
        ("", ""),
        ("Create or manage a key at:", styled("Create or manage a key at:", WHITE, enabled=color)),
        (f" {url}", f"  {styled(url, PURPLE+UNDERLINE, enabled=color)}"),
        ("", ""),
        ("IMPORTANT", styled("IMPORTANT", YELLOW+BOLD, enabled=color)),
        ("API key requests may generate costs on your OpenRouter account.", styled("API key requests may generate costs on your OpenRouter account.", WHITE, enabled=color)),
        ("Review model pricing and set spending limits before continuing!", styled("Review model pricing and set spending limits before continuing!", WHITE, enabled=color)),
        ("",""),
        (f" Save path: {save_path}", f"    Save path: {styled(save_path, PURPLE, enabled=color)}"),
        ("", "")
    ]
    target.write(_info_box("Connect to OpenRouter", rows, enabled=color) + "\n\n")
    target.flush()

def print_login_success(path: Path, *, file: TextIO | None = None) -> None:
    target = file if file is not None else sys.stdout
    color = supports_color(target)
    target.write(
        f"\n{styled('SUCCESS', GREEN+BOLD, enabled=color)}  {styled('OpenRouter connected succesfully', MINT, enabled=color)}\n"
        f"  {styled('Credential saved in:', MINT, enabled=color)} {styled(path, PURPLE, enabled=color)}"
        f"  {styled('Start an evolution:', MINT, enabled=color)} {styled('worst-shinka run', GRAY, enabled=color)}\n"
    )
    target.flush()

def print_logout_result(*, removed: bool, still_connected: bool, file: TextIO | None = None) -> None:
    target = file if file is not None else sys.stdout
    color = supports_color(target)
    if still_connected:
        tone = RED
        heading_text = "WARNING"
        message = (
            "Local credentials removed, but OpenRouter is still connected!"
            if removed else "OpenRouter is still connected through the environment."
        )
        detail = "The OPENROUTER_API_KEY environment variable is set."
    elif removed:
        tone = GREEN
        heading_text = "SUCCESS"
        message = "OpenRouter credentials removed."
        detail = "To run evolutions connect again with: worst-shinka config login"
    else:
        tone = BLUE
        heading_text = "INFO"
        message = "No loccaly saved OpenRouter credentials were found."
        detail = "The application is already logged out."
    heading = styled(heading_text, tone + BOLD, enabled=color)
    target.write(
        f"\n{heading}   {styled(message, tone, enabled=color)}\n"
        f"  {styled(detail, tone, enabled=color)}\n"
    )
    target.flush()

def _terminal_width(target: TextIO) -> int:
    is_tyy = getattr(target, "isatty", lambda: False)()
    if is_tyy:
        return shutil.get_terminal_size((80, 20)).columns
    return 80

def _crop(value: object, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(width - 3, 0)] + "..."

def _render_table(
    headers: tuple[str, ...], 
    rows: list[tuple[object, ...]], 
    *, 
    enabled: bool, 
    border_color: str, 
    numeric_columns: set[int] | None = None, 
    status_column: int | None = None,
    target_stream: TextIO | None = None
) -> str:
    normalized = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in normalized:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(val))
    term_w = _terminal_width(target_stream or sys.stdout)
    target_table_width = int(term_w * 0.8)
    fixed_decorations_width = (len(headers) + 1) + (len(headers) * 2)
    available_content_width = target_table_width - fixed_decorations_width

    current_content_width = sum(widths)
    if current_content_width < available_content_width:
        extra_space = available_content_width - current_content_width
        add_per_col = extra_space // len(headers)
        remainder = extra_space % len(headers)

        for i in range(len(widths)):
            widths[i] += add_per_col + (1 if i < remainder else 0)

    top_border = styled("┌" + "┬".join("─" * (w + 2) for w in widths) + "┐", border_color, enabled=enabled)
    mid_border = styled("├" + "┼".join("─" * (w + 2) for w in widths) + "┤", border_color, enabled=enabled)
    bot_border = styled("└" + "┴".join("─" * (w + 2) for w in widths) + "┘", border_color, enabled=enabled)
    side = styled("│", border_color, enabled=enabled)

    def table_row(values: list[str], *, header: bool = False) -> str:
        cells = []
        for idx, val in enumerate(values):
            visible = _crop(val, widths[idx])
            padded = f" {visible:<{widths[idx]}} "
            tone = MINT
            if not header and numeric_columns and idx in numeric_columns and visible != "-":
                tone = BLUE
            if not header and status_column == idx and visible.lower() == "incorrect":
                tone = RED + BOLD
            elif header:
                tone = MINT + BOLD
            cells.append(styled(padded, tone, enabled=enabled))
        return side + side.join(cells) + side

    output = [top_border, table_row(list(headers), header=True), mid_border]
    output.extend(table_row(row) for row in normalized)
    output.append(bot_border)

    return "\n".join(output)

def print_gen_header(*, generation: int, completed_generation: int | None = None, total_cost: float | None = None, file: TextIO | None = None) -> None:
    target = file if file is not None else sys.stdout
    color = supports_color(target)
    term_width = _terminal_width(target)

    title = f"GENERATION #{generation}"
    horizontal_line = styled("─" * term_width, YELLOW + BOLD, enabled=color)
    centered_title = styled(title.center(term_width), YELLOW + BOLD, enabled=color)

    output = ["", horizontal_line, centered_title, horizontal_line, ""]

    target.write("\n".join(output) + "\n")
    target.flush()
    
def print_gen_metadata(*, generation: int, name: str, parent_ids: list[str], mode: str, file: TextIO | None = None) -> None:
    target = file if file is not None else sys.stdout
    color = supports_color(target)
    parents = "; ".join(parent_ids) if parent_ids else "-"
    table = _render_table(
        ("ID", "NAME", "PARENTS NO.", "PARENTS ID", "MODE"),
        [(generation, name, len(parent_ids), parents, mode)],
        enabled=color,
        border_color=PURPLE,
        numeric_columns={0, 2}
    )
    target.write(table + "\n")
    target.flush()

def print_gen_results(
        rows: list[dict[str, object]],
        *, 
        generation: int,
        heading: str = "GENERATION RESULT",
        file: TextIO | None = None
) -> None:
    target = file if file else sys.stdout
    color = supports_color(target)

    def check_val(value: object) -> object:
        return "-" if value is None else value

    values = [
        (
            check_val(row.get("generation", generation)),
            row.get("status") or "pending",
            check_val(row.get("score", "-")),
            check_val(row.get("cost", "-")),
            check_val(row.get("elo", "-")),
            check_val(row.get("time", "-"))

        ) for row in rows
    ]
    if not values:
        values = [(generation, "pending", "-", "-", "-", "-")]

    target.write(styled(heading, YELLOW + BOLD, enabled=color) + "\n")
    for value in values:
        status = str(value[1]).lower()
        border_color = GREEN if status == "correct" else RED if status == "incorrect" else PURPLE
        target.write(
            _render_table(
                ("GEN", "STATUS", "SCORE", "COST", "ELO", "TIME"),
                [value],
                enabled=color,
                border_color=border_color,
                numeric_columns={0, 2, 3, 4, 5},
                status_column=1
            ) + "\n"
        )

    target.flush()

class ColoredLogFormatter(logging.Formatter):
    """logging formatter with colored date, time and level"""

    def __init__(self, *, use_color: bool) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).astimezone()
        date = timestamp.strftime("%Y-%m-%d")
        time = timestamp.strftime("%H:%M:%S")
        level = f"{record.levelname:<8}"
        if self.use_color:
            date = styled(date, PEACH, enabled=True)
            time = styled(time, MINT, enabled=True)
            level = styled(level, _LEVEL_COLORS.get(record.levelno, RESET), enabled=True)

        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        if record.stack_info:
            message += "\n" + self.formatStack(record.stack_info)

        return f"{date} {time} - {level} - {message}"

def configure_logging(*, level: int = logging.INFO, stream: TextIO | None = None) -> None:
    target = stream if stream is not None else sys.stderr
    handler = logging.StreamHandler(target)
    handler.setFormatter(ColoredLogFormatter(use_color=supports_color(target)))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    gib = value / (1024 ** 3)
    return f"{gib:.1f} GiB"

def _total_memory() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None

def _available_memory() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None

def _cpu_frequency_mhz() -> float | None:
    try:
        cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^cpu MHz\s*:\s*([\d.]+)", cpu_info, re.MULTILINE)
        return float(match.group(1)) if match else None
    except (OSError, ValueError):
        return None

def _cpu_details() -> tuple[str, str]:
    try:
        cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "unknown", "unknown"

    vendor_match = re.search(r"^vendor_id\s*:\s*(.+)$", cpu_info, re.MULTILINE)
    model_match = re.search(r"^model name\s*:\s*(.+)$", cpu_info, re.MULTILINE)
    vendor = vendor_match.group(1).strip() if vendor_match else "unknown"
    model = model_match.group(1).strip() if model_match else "unknown"
    return vendor, model

def _known_gpu_memory(name: str) -> int | None:
    known_memory = {
        "RTX 500 Ada Generation Laptop GPU": 4 * 1024 ** 3,
    }
    for model, memory in known_memory.items():
        if model in name:
            return memory
    return None

def _gpu_details() -> tuple[str, int | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, _, memory = result.stdout.strip().splitlines()[0].partition(",")
            memory_mib = int(memory.strip()) if memory.strip().isdigit() else None
            return name.strip(), memory_mib * 1024 * 1024 if memory_mib is not None else None
    except (OSError, ValueError):
        pass

    try:
        result = subprocess.run(
            ["lspci", "-mm"],
            capture_output=True,
            text=True,
            check=False,
        )
        candidates: list[tuple[str, str]] = []
        for line in result.stdout.splitlines():
            if "VGA compatible controller" not in line and "3D controller" not in line:
                continue
            fields = re.findall(r'"([^"]*)"', line)
            if fields:
                vendor = fields[1] if len(fields) > 1 else ""
                name = fields[2] if len(fields) > 2 else fields[-1]
                candidates.append((vendor, name))

        candidates.sort(key=lambda candidate: "NVIDIA" not in candidate[0])
        vram_paths = Path("/sys/class/drm").glob("card*/device/mem_info_vram_total")
        for vendor, name in candidates:
            full_name = f"{vendor} {name}".strip()
            for vram_path in vram_paths:
                try:
                    return full_name, int(vram_path.read_text(encoding="ascii").strip())
                except (OSError, ValueError):
                    continue
            return full_name, _known_gpu_memory(full_name)
    except OSError:
        pass

    return "unknown", None

@dataclass(frozen=True)
class SystemInfo:
    operating_system: str
    python_version: str
    cpu: str
    cpu_vendor: str
    cpu_model: str
    logical_cores: int | None
    cpu_frequency_mhz: float | None
    total_memory: int | None
    available_memory: int | None
    gpu: str
    gpu_memory: int | None

def collect_system_info() -> SystemInfo:
    cpu_vendor, cpu_model = _cpu_details()
    gpu, gpu_memory = _gpu_details()
    return SystemInfo(
        operating_system=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=platform.python_version(),
        cpu=platform.machine() or "unknown",
        cpu_vendor=cpu_vendor,
        cpu_model=cpu_model,
        logical_cores=os.cpu_count(),
        cpu_frequency_mhz=_cpu_frequency_mhz(),
        total_memory=_total_memory(),
        available_memory=_available_memory(),
        gpu=gpu,
        gpu_memory=gpu_memory,
    )

def print_startup_info(config: RunConfig, run_dir: Path, *, file: TextIO | None = None, system_info: SystemInfo | None = None) -> None:
    target = file if file is not None else sys.stdout
    color = supports_color(target)
    info = system_info or collect_system_info()

    def number(value: object) -> str:
        return styled(value, BLUE, enabled=color)

    def path(value: Path | str) -> str:
        return styled(value, PURPLE, enabled=color)

    models = config.resolved_models()
    frequency = "unknown" if info.cpu_frequency_mhz is None else f"{info.cpu_frequency_mhz:.0f} MHz"
    cores = "unknown" if info.logical_cores is None else str(info.logical_cores)
    gpu_memory = _format_bytes(info.gpu_memory)
    lines = [
        styled("🔄 Running configuration", BOLD, enabled=color),
        f"  Run directory:      {path(run_dir)}",
        f"  Results root:       {path(config.results_dir.expanduser().resolve())}",
        f"  Initial model:      {path(config.initial_model)}",
        f"  Mode:               {config.mode}",
        f"  Generations:        {number(config.generations)}",
        f"  Workers:            {number(config.workers)}",
        f"  Parents:            {number(config.parents)}",
        f"  Models count:       {number(len(models))}",
        "",
        "",
        "",
        styled("🖥️ System information", BOLD, enabled=color),
        f"  Operating system:   {info.operating_system}",
        f"  Python:             {number(info.python_version)}",
        f"  Processor:          {info.cpu_model} {info.cpu}",
        f"  Logical cores:      {number(cores)}",
        f"  CPU frequency:      {number(frequency)}",
        f"  Total RAM:          {number(_format_bytes(info.total_memory))}",
        f"  Free memory:        {number(_format_bytes(info.available_memory))}",
        f"  Detected GPU:       {number(info.gpu)}",
        f"  GPU VRAM:           {number(gpu_memory)}",
        "",
        "",
        ""
    ]
    target.write("\n".join(lines))
    target.flush()


def print_play_start(*, model_path: Path, process_id: int, file: TextIO | None = None) -> None:
    target = file if file is not None else sys.stdout
    color = supports_color(target)

    def label(value: str) -> str:
        return styled(f"{value:<19}", MINT, enabled=color)

    rows = [
        ("",""),
        (
            "GAME",
            f"{label('GAME')}{styled('ATARI TENNIS', YELLOW+BOLD, enabled=color)}"
        ),
        (
            "Process ID",
            f"{label('Process ID')}{styled(process_id, BLUE+BOLD, enabled=color)}"
        ),
        ("",""),
        ("",""),
        (
            "Press Q to quit",
            styled("Press Q to quit", RED+BOLD, enabled=color)
        ),
        ("","")
    ]
    target.write(_info_box("Starting game", rows, enabled=color)+"\n")
    target.flush()