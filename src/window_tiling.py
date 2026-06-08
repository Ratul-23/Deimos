from pathlib import Path

from wizwalker.constants import user32
from wizwalker.utils import get_wiz_install

TASKBAR_CLASS: str = "Shell_TrayWnd"

WALL_DIMENSIONS: dict[int, list[tuple[float, float]]] = {
    1: [(0, 0)],
    2: [(0, 0), (0, 0.5)],
    3: [(0, 0), (0.5, 0), (0, 0.5)],
    4: [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)],
    5: [(0, 0), (0.3333333, 0), (0.6666666, 0), (0, 0.5), (0.3333333, 0.5)],
    6: [
        (0, 0),
        (0.3333333, 0),
        (0.6666666, 0),
        (0, 0.5),
        (0.3333333, 0.5),
        (0.6666666, 0.5),
    ],
}


def set_window_position(hwnd: int, x: int, y: int, width: int, height: int) -> None:
    gwl_style: int = -16
    ws_overlappedwindow: int = 0x00CF0000
    ws_popup: int = 0x80000000

    current_style: int = int(user32.GetWindowLongW(hwnd, gwl_style))
    new_style: int = (current_style & ~ws_overlappedwindow) | ws_popup
    user32.SetWindowLongW(hwnd, gwl_style, new_style)
    user32.SetWindowPos(hwnd, 0, x, y, width, height, 0x0040 | 0x0020)


def set_window_size(width: int, height: int) -> None:
    preferences: Path = get_wiz_install() / "Bin" / "preferences.xml"
    with open(preferences) as config:
        data: list[str] = config.readlines()

    data[56] = '    <IsFullscreen TYPE="INT">0</IsFullscreen>'
    data[57] = f'    <Resolution TYPE="STR">{width}x{height}</Resolution>'

    with open(preferences, "w") as config:
        config.writelines(data)


def get_tile_dimensions(count: int) -> tuple[int, int]:
    user32.SetProcessDPIAware()
    screen_w: int = user32.GetSystemMetrics(0)
    screen_h: int = user32.GetSystemMetrics(1)

    match count:
        case 1:
            return screen_w, screen_h
        case 2:
            return int(screen_w / 1.6), int(screen_h / 2)
        case 3 | 4:
            return int(screen_w / 2), int(screen_h / 2)
        case 5 | 6:
            return int(screen_w / 3), int(screen_h / 2)
        case _:
            return int(screen_w / 2), int(screen_h / 2)


def hide_taskbar() -> None:
    taskbar_hwnd: int = user32.FindWindowW(TASKBAR_CLASS, None)

    if taskbar_hwnd:
        # SWP_NOSIZE | SWP_NOMOVE | SWP_HIDEWINDOW
        user32.SetWindowPos(taskbar_hwnd, 0, 0, 0, 0, 0, 0x0083)


def restore_taskbar() -> None:
    taskbar_hwnd: int = user32.FindWindowW(TASKBAR_CLASS, None)

    if taskbar_hwnd:
        # SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW
        user32.SetWindowPos(taskbar_hwnd, 0, 0, 0, 0, 0, 0x0043)


def tile_windows(handles: list[int], hide_taskbar_setting: bool = False) -> None:
    count: int = len(handles)

    if count not in WALL_DIMENSIONS:
        return

    user32.SetProcessDPIAware()
    screen_w: int = user32.GetSystemMetrics(0)
    screen_h: int = user32.GetSystemMetrics(1)

    modifiers: list[tuple[float, float]] = WALL_DIMENSIONS[count]
    positions: list[tuple[int, int]] = [
        (int(m[0] * screen_w), int(m[1] * screen_h)) for m in modifiers
    ]
    width: int
    height: int
    width, height = get_tile_dimensions(count)

    set_window_size(width, height)

    if hide_taskbar_setting:
        hide_taskbar()

    for i, handle in enumerate(handles):
        x: int
        y: int
        x, y = positions[i]

        if count == 2:
            set_window_position(handle, x + 1, y, width, height)

        elif count > 2:
            set_window_position(handle, x, y, width, height)
