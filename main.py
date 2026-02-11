import csv
import io
import math
import threading
import time
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pygame


# -----------------------------
# Config
# -----------------------------

FPS = 60
FULLSCREEN = True
WINDOW_W, WINDOW_H = 1100, 700

# Phase durations (seconds)
PHASES = [
    ("IMAGINE", 2.0),
    ("MARGIN", 2.0),
    ("PREP", 4.0),
    ("MANUAL", 2.0),
    ("MOVE", 4.0),
    ("REST", 2.0),
]

TOTAL_SEC = sum(d for _, d in PHASES)
MOVE_SEC = next(d for name, d in PHASES if name == "MOVE")

COUNTDOWN_SEC = 3.0
N_TRIALS = 999999  # effectively endless; stop with ESC

# GSI (Geospatial Information Authority of Japan) tiles
USE_GSI = True
GSI_URL_TEMPLATE = "https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png"
GSI_ZOOM = 16
GSI_CENTER_LAT = 34.702485  # Osaka Station
GSI_CENTER_LON = 135.495951
GSI_ATTRIBUTION = "出典: 国土地理院"
GSI_TILE_SIZE = 256
GSI_CELL_TILES = 1  # 1 tile per move
GSI_PREFETCH_ENABLED = True
GSI_PREFETCH_RADIUS = 5  # tiles (square radius)
MAX_TILE_CACHE = 512
TILE_FETCH_TIMEOUT = 4.0

ASSET_DIR = Path("assets")
GSI_CACHE_DIR = ASSET_DIR / "gsi_cache"

# Visual cue (phase flash)
PHASE_FLASH_SEC = 0.9
PHASE_FLASH_BG_ALPHA = 120
PHASE_FLASH_TEXT_ALPHA = 255
PHASE_FLASH_SHADOW_ALPHA = 140

# Photodiode patch
ENABLE_PD = True
PD_FLASH_SEC = 0.20
PD_SIZE = 168
PD_PADDING = 10
PD_FLASH_PHASES = {"IMAGINE"}

# Logging
LOG_DIR = Path("logs")
SNAP_DIR = Path("snapshots")
LOG_DIR.mkdir(exist_ok=True)
SNAP_DIR.mkdir(exist_ok=True)

# Keys
MOVE_KEYS = {
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
    pygame.K_w: (0, -1),
    pygame.K_s: (0, 1),
    pygame.K_a: (-1, 0),
    pygame.K_d: (1, 0),
}

PHASE_COLORS = {
    "IMAGINE": (30, 90, 170),
    "MARGIN": (90, 90, 120),
    "PREP": (170, 140, 20),
    "MANUAL": (170, 40, 40),
    "MOVE": (40, 140, 90),
    "REST": (50, 50, 60),
}

PHASE_LABELS_JP = {
    "IMAGINE": "イメージ(2s)",
    "MARGIN": "マージン(2s)",
    "PREP": "運動準備(4s)",
    "MANUAL": "手動入力(2s)",
    "MOVE": "システム動作(4s)",
    "REST": "安静(2s)",
}

PHASE_BIG_LABELS_JP = {
    "IMAGINE": "イメージ",
    "MARGIN": "マージン",
    "PREP": "運動準備",
    "MANUAL": "手動入力",
    "MOVE": "システム動作",
    "REST": "安静",
}


@dataclass
class Command:
    dx: int
    dy: int
    key_name: str


class SessionLog:
    def __init__(self) -> None:
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.start_ts = time.perf_counter()
        self.csv_path = LOG_DIR / f"session_{self.session_id}.csv"
        self.csv_f = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_f,
            fieldnames=[
                "session_id", "session_start_ts",
                "t", "trial_idx", "phase", "phase_t",
                "manual_key", "label",
                "move_dx", "move_dy",
                "view_x", "view_y",
                "event",
            ],
        )
        self.writer.writeheader()
        self.current_trial_idx: Optional[int] = None
        self.buffer: list[dict[str, str | int]] = []

    def t(self) -> float:
        return time.perf_counter() - self.start_ts

    def log_row(
        self,
        trial_idx: int,
        phase: str,
        phase_t: float,
        manual_key: str,
        label: str,
        move_dx: int,
        move_dy: int,
        view_x: float,
        view_y: float,
        event: str = "",
    ) -> None:
        if self.current_trial_idx is None:
            self.current_trial_idx = trial_idx
        elif trial_idx != self.current_trial_idx:
            self.flush_current()
            self.current_trial_idx = trial_idx

        self.buffer.append(
            {
                "session_id": self.session_id,
                "session_start_ts": f"{self.start_ts:.6f}",
                "t": f"{self.t():.6f}",
                "trial_idx": trial_idx,
                "phase": phase,
                "phase_t": f"{phase_t:.6f}",
                "manual_key": manual_key,
                "label": label,
                "move_dx": move_dx,
                "move_dy": move_dy,
                "view_x": f"{view_x:.2f}",
                "view_y": f"{view_y:.2f}",
                "event": event,
            }
        )

    def flush_current(self) -> None:
        if not self.buffer:
            return
        self.writer.writerows(self.buffer)
        self.buffer.clear()

    def discard_current(self) -> None:
        self.buffer.clear()

    def close(self, discard_current: bool = False) -> None:
        if discard_current:
            self.discard_current()
        else:
            self.flush_current()
        self.csv_f.flush()
        self.csv_f.close()


class PhaseSchedule:
    def __init__(self) -> None:
        self.boundaries = []
        t = 0.0
        for name, dur in PHASES:
            self.boundaries.append((name, t, t + dur))
            t += dur

    def phase_of(self, t: float) -> tuple[str, float]:
        for name, t0, t1 in self.boundaries:
            if t < t1:
                return name, t - t0
        return PHASES[-1][0], PHASES[-1][1]


def clamp_lat(lat: float) -> float:
    return max(-85.05112878, min(85.05112878, lat))


def latlon_to_world_px(lat: float, lon: float, zoom: int, tile_size: int) -> tuple[float, float]:
    lat = clamp_lat(lat)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n * tile_size
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n * tile_size
    return x, y


class GsiTileProvider:
    def __init__(self, zoom: int, tile_size: int, cache_dir: Path) -> None:
        self.zoom = zoom
        self.tile_size = tile_size
        self.cache_dir = cache_dir / f"z{zoom}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mem_cache: OrderedDict[tuple[int, int], pygame.Surface] = OrderedDict()
        self.placeholder = pygame.Surface((tile_size, tile_size))
        self.placeholder.fill((200, 200, 200))

    def _tile_path(self, x: int, y: int) -> Path:
        return self.cache_dir / f"{x}_{y}.png"

    def _load_surface_from_bytes(self, data: bytes) -> pygame.Surface:
        img = pygame.image.load(io.BytesIO(data))
        return img.convert() if img.get_alpha() is None else img.convert_alpha()

    def _load_surface_from_file(self, path: Path) -> pygame.Surface:
        img = pygame.image.load(str(path))
        return img.convert() if img.get_alpha() is None else img.convert_alpha()

    def _fetch_tile(self, x: int, y: int) -> Optional[bytes]:
        url = GSI_URL_TEMPLATE.format(z=self.zoom, x=x, y=y)
        try:
            with urllib.request.urlopen(url, timeout=TILE_FETCH_TIMEOUT) as resp:
                return resp.read()
        except Exception:
            return None

    def get_tile(self, x: int, y: int) -> pygame.Surface:
        key = (x, y)
        if key in self.mem_cache:
            self.mem_cache.move_to_end(key)
            return self.mem_cache[key]

        path = self._tile_path(x, y)
        surf: Optional[pygame.Surface] = None
        if path.exists():
            try:
                surf = self._load_surface_from_file(path)
            except Exception:
                surf = None

        if surf is None:
            data = self._fetch_tile(x, y)
            if data:
                try:
                    path.write_bytes(data)
                except Exception:
                    pass
                try:
                    surf = self._load_surface_from_bytes(data)
                except Exception:
                    surf = None

        if surf is None:
            surf = self.placeholder

        self.mem_cache[key] = surf
        if len(self.mem_cache) > MAX_TILE_CACHE:
            self.mem_cache.popitem(last=False)
        return surf

    def prefetch_tile(self, x: int, y: int) -> None:
        path = self._tile_path(x, y)
        if path.exists():
            return
        data = self._fetch_tile(x, y)
        if data:
            try:
                path.write_bytes(data)
            except Exception:
                pass


class Prefetcher:
    def __init__(self, provider: GsiTileProvider, center_tx: int, center_ty: int, radius: int) -> None:
        self.provider = provider
        self.coords = []
        max_tile = (2 ** provider.zoom) - 1
        for ty in range(center_ty - radius, center_ty + radius + 1):
            if ty < 0 or ty > max_tile:
                continue
            for tx in range(center_tx - radius, center_tx + radius + 1):
                if tx < 0 or tx > max_tile:
                    continue
                self.coords.append((tx, ty))

        self.total = len(self.coords)
        self._done = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        for tx, ty in self.coords:
            self.provider.prefetch_tile(tx, ty)
            with self._lock:
                self._done += 1

    def progress(self) -> tuple[int, int]:
        with self._lock:
            return self._done, self.total

    def is_done(self) -> bool:
        return not self._thread.is_alive()


class MapWorld:
    def __init__(self) -> None:
        self.tile_provider = GsiTileProvider(GSI_ZOOM, GSI_TILE_SIZE, GSI_CACHE_DIR)
        self.map_w = GSI_TILE_SIZE * (2 ** GSI_ZOOM)
        self.map_h = self.map_w
        self.cell_size = GSI_TILE_SIZE * GSI_CELL_TILES
        self.init_view_x, self.init_view_y = latlon_to_world_px(
            GSI_CENTER_LAT,
            GSI_CENTER_LON,
            GSI_ZOOM,
            GSI_TILE_SIZE,
        )
        self.source = f"GSI std z{GSI_ZOOM}"

    def render(self, target: pygame.Surface, view_x: float, view_y: float) -> None:
        left = view_x - WINDOW_W / 2
        top = view_y - WINDOW_H / 2
        right = view_x + WINDOW_W / 2
        bottom = view_y + WINDOW_H / 2

        tile_size = self.tile_provider.tile_size
        max_tile = (2 ** self.tile_provider.zoom) - 1

        start_tx = int(math.floor(left / tile_size))
        end_tx = int(math.floor((right - 1) / tile_size))
        start_ty = int(math.floor(top / tile_size))
        end_ty = int(math.floor((bottom - 1) / tile_size))

        for ty in range(start_ty, end_ty + 1):
            if ty < 0 or ty > max_tile:
                continue
            for tx in range(start_tx, end_tx + 1):
                if tx < 0 or tx > max_tile:
                    continue
                tile = self.tile_provider.get_tile(tx, ty)
                dst_x = int(tx * tile_size - left)
                dst_y = int(ty * tile_size - top)
                target.blit(tile, (dst_x, dst_y))


class Beeper:
    def __init__(self) -> None:
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        self.cache: dict[tuple[int, float], pygame.mixer.Sound] = {}

    def beep(self, freq: int, duration: float = 0.14, volume: float = 0.3) -> None:
        key = (freq, duration)
        if key not in self.cache:
            sample_rate = 44100
            n_samples = int(sample_rate * duration)
            buf = bytearray()
            for i in range(n_samples):
                t = i / sample_rate
                val = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
                buf += int(val).to_bytes(2, byteorder="little", signed=True)
            snd = pygame.mixer.Sound(buffer=bytes(buf))
            self.cache[key] = snd
        snd = self.cache[key]
        snd.set_volume(volume)
        snd.play()


class TrialController:
    def __init__(
        self,
        map_w: int,
        map_h: int,
        cell_size: int,
        move_sec: float,
        init_view_x: float,
        init_view_y: float,
    ) -> None:
        self.schedule = PhaseSchedule()
        self.trial_start_ts = time.perf_counter()
        self.trial_idx = 0
        self.manual_command: Optional[Command] = None
        self.move_dx = 0
        self.move_dy = 0
        self.map_w = map_w
        self.map_h = map_h
        self.cell_size = cell_size
        self.move_sec = move_sec
        self.view_x = init_view_x
        self.view_y = init_view_y
        self.move_start_x = self.view_x
        self.move_start_y = self.view_y

    def t_in_trial(self) -> float:
        return time.perf_counter() - self.trial_start_ts

    def phase(self) -> tuple[str, float]:
        return self.schedule.phase_of(self.t_in_trial())

    def advance_if_needed(self) -> bool:
        if self.t_in_trial() >= TOTAL_SEC:
            self.trial_idx += 1
            self.trial_start_ts = time.perf_counter()
            self.manual_command = None
            self.move_dx = 0
            self.move_dy = 0
            self.move_start_x = self.view_x
            self.move_start_y = self.view_y
            return True
        return False

    def accept_key(self, key: int) -> None:
        if self.manual_command is not None:
            return
        if key not in MOVE_KEYS:
            return
        dx, dy = MOVE_KEYS[key]
        self.manual_command = Command(dx=dx, dy=dy, key_name=pygame.key.name(key))

    def finalize_manual(self) -> None:
        if self.manual_command is None:
            self.move_dx, self.move_dy = 0, 0
        else:
            self.move_dx, self.move_dy = self.manual_command.dx, self.manual_command.dy

        self.move_start_x = self.view_x
        self.move_start_y = self.view_y

    def _clamp_view(self) -> None:
        if self.map_w <= WINDOW_W:
            self.view_x = self.map_w / 2
        else:
            self.view_x = max(WINDOW_W / 2, min(self.map_w - WINDOW_W / 2, self.view_x))

        if self.map_h <= WINDOW_H:
            self.view_y = self.map_h / 2
        else:
            self.view_y = max(WINDOW_H / 2, min(self.map_h - WINDOW_H / 2, self.view_y))

    def update_view(self, phase: str, phase_t: float) -> None:
        if phase != "MOVE":
            return
        target_x = self.move_start_x + self.move_dx * self.cell_size
        target_y = self.move_start_y + self.move_dy * self.cell_size
        alpha = min(1.0, phase_t / self.move_sec)
        self.view_x = self.move_start_x + (target_x - self.move_start_x) * alpha
        self.view_y = self.move_start_y + (target_y - self.move_start_y) * alpha
        self._clamp_view()

    def finalize_move(self) -> None:
        target_x = self.move_start_x + self.move_dx * self.cell_size
        target_y = self.move_start_y + self.move_dy * self.cell_size
        self.view_x = target_x
        self.view_y = target_y
        self._clamp_view()


def draw_timeline(screen: pygame.Surface, t_in_trial: float) -> None:
    x, y = 14, 52
    w, h = 540, 18
    pygame.draw.rect(screen, (50, 50, 55), (x, y, w, h), border_radius=6)

    t = 0.0
    for name, dur in PHASES:
        seg_w = int(w * dur / TOTAL_SEC)
        col = PHASE_COLORS.get(name, (80, 80, 80))
        pygame.draw.rect(screen, col, (x + int(w * t / TOTAL_SEC), y, seg_w, h), border_radius=6)
        t += dur

    marker_x = int(w * t_in_trial / TOTAL_SEC)
    pygame.draw.line(screen, (240, 240, 240), (x + marker_x, y - 3), (x + marker_x, y + h + 3), 2)


def label_from_command(cmd: Optional[Command]) -> str:
    if cmd is None:
        return "静止"
    if cmd.dx == 0 and cmd.dy == -1:
        return "前"
    if cmd.dx == 0 and cmd.dy == 1:
        return "後"
    if cmd.dx == -1 and cmd.dy == 0:
        return "左"
    if cmd.dx == 1 and cmd.dy == 0:
        return "右"
    return "不明"


def save_snapshot(surface: pygame.Surface, trial_idx: int) -> None:
    fname = SNAP_DIR / f"map_trial_{trial_idx:04d}.png"
    pygame.image.save(surface, str(fname))


def save_map_view(world: MapWorld, view_x: float, view_y: float, trial_idx: int) -> None:
    snap = pygame.Surface((WINDOW_W, WINDOW_H))
    world.render(snap, view_x, view_y)
    save_snapshot(snap, trial_idx)


def draw_phase_flash(
    screen: pygame.Surface,
    phase: str,
    alpha: float,
    font_huge: pygame.font.Font,
) -> None:
    if alpha <= 0.0:
        return
    col = PHASE_COLORS.get(phase, (80, 80, 80))
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((*col, int(PHASE_FLASH_BG_ALPHA * alpha)))
    screen.blit(overlay, (0, 0))

    text = PHASE_BIG_LABELS_JP.get(phase, phase)
    shadow = font_huge.render(text, True, (0, 0, 0))
    shadow.set_alpha(int(PHASE_FLASH_SHADOW_ALPHA * alpha))
    label = font_huge.render(text, True, (255, 255, 255))
    label.set_alpha(int(PHASE_FLASH_TEXT_ALPHA * alpha))

    x = WINDOW_W // 2 - label.get_width() // 2
    y = WINDOW_H // 2 - label.get_height() // 2
    screen.blit(shadow, (x + 4, y + 4))
    screen.blit(label, (x, y))


def draw_photodiode(screen: pygame.Surface, active: bool) -> None:
    if not ENABLE_PD:
        return
    x = PD_PADDING
    y = WINDOW_H - PD_PADDING - PD_SIZE
    pygame.draw.rect(screen, (0, 0, 0), (x, y, PD_SIZE, PD_SIZE))
    if active:
        pygame.draw.rect(screen, (255, 255, 255), (x, y, PD_SIZE, PD_SIZE))


def draw_crosshair(screen: pygame.Surface) -> None:
    cx = WINDOW_W // 2
    cy = WINDOW_H // 2
    col = (30, 30, 30)
    col2 = (230, 230, 230)
    size = 18
    gap = 6
    pygame.draw.line(screen, col, (cx - size, cy), (cx - gap, cy), 3)
    pygame.draw.line(screen, col, (cx + gap, cy), (cx + size, cy), 3)
    pygame.draw.line(screen, col, (cx, cy - size), (cx, cy - gap), 3)
    pygame.draw.line(screen, col, (cx, cy + gap), (cx, cy + size), 3)
    pygame.draw.line(screen, col2, (cx - size, cy), (cx - gap, cy), 1)
    pygame.draw.line(screen, col2, (cx + gap, cy), (cx + size, cy), 1)
    pygame.draw.line(screen, col2, (cx, cy - size), (cx, cy - gap), 1)
    pygame.draw.line(screen, col2, (cx, cy + gap), (cx, cy + size), 1)


def main() -> None:
    pygame.init()
    flags = pygame.FULLSCREEN if FULLSCREEN else 0
    screen = pygame.display.set_mode((0, 0), flags)
    global WINDOW_W, WINDOW_H
    WINDOW_W, WINDOW_H = screen.get_size()
    pygame.display.set_caption("Map-BCI Experiment 1")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Meiryo", 22)
    font_small = pygame.font.SysFont("Meiryo", 18)
    font_big = pygame.font.SysFont("Meiryo", 36)
    font_huge = pygame.font.SysFont("Meiryo", 64)
    font_mono = pygame.font.SysFont("consolas", 18)

    world = MapWorld()
    beeper = Beeper()

    controller: Optional[TrialController] = None
    logger: Optional[SessionLog] = None
    prefetcher: Optional[Prefetcher] = None
    abort_discard = False

    state = "IDLE"  # IDLE -> COUNTDOWN -> PREFETCH -> RUNNING
    countdown_start = 0.0

    running = True
    last_phase_name = ""
    flash_phase = ""
    flash_start = 0.0
    pd_flash_start: Optional[float] = None

    while running:
        now = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                abort_discard = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    abort_discard = True
                if state == "IDLE" and event.key == pygame.K_SPACE:
                    state = "COUNTDOWN"
                    countdown_start = now
                if state == "RUNNING" and controller is not None:
                    phase, _ = controller.phase()
                    if phase == "MANUAL":
                        controller.accept_key(event.key)
                    if event.key == pygame.K_s:
                        save_map_view(world, controller.view_x, controller.view_y, controller.trial_idx)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if state == "IDLE":
                    state = "COUNTDOWN"
                    countdown_start = now

        if state == "COUNTDOWN":
            if now - countdown_start >= COUNTDOWN_SEC:
                if USE_GSI and GSI_PREFETCH_ENABLED:
                    center_tx = int(world.init_view_x // world.tile_provider.tile_size)
                    center_ty = int(world.init_view_y // world.tile_provider.tile_size)
                    prefetcher = Prefetcher(world.tile_provider, center_tx, center_ty, GSI_PREFETCH_RADIUS)
                    state = "PREFETCH"
                else:
                    controller = TrialController(
                        world.map_w,
                        world.map_h,
                        world.cell_size,
                        MOVE_SEC,
                        world.init_view_x,
                        world.init_view_y,
                    )
                    logger = SessionLog()
                    state = "RUNNING"
                    last_phase_name = ""
        elif state == "PREFETCH":
            if prefetcher is not None and prefetcher.is_done():
                controller = TrialController(
                    world.map_w,
                    world.map_h,
                    world.cell_size,
                    MOVE_SEC,
                    world.init_view_x,
                    world.init_view_y,
                )
                logger = SessionLog()
                state = "RUNNING"
                last_phase_name = ""
        elif state == "RUNNING" and controller is not None and logger is not None:
            phase, phase_t = controller.phase()

            if phase != last_phase_name:
                last_phase_name = phase
                flash_phase = phase
                flash_start = now

                if phase in PD_FLASH_PHASES:
                    pd_flash_start = now

                if phase == "IMAGINE":
                    beeper.beep(660)
                elif phase == "MARGIN":
                    beeper.beep(520)
                elif phase == "PREP":
                    beeper.beep(440)
                elif phase == "MANUAL":
                    beeper.beep(880)
                elif phase == "MOVE":
                    controller.finalize_manual()
                    beeper.beep(320)
                elif phase == "REST":
                    beeper.beep(260)
                    controller.finalize_move()
                    save_map_view(world, controller.view_x, controller.view_y, controller.trial_idx)

                logger.log_row(
                    controller.trial_idx,
                    phase,
                    phase_t,
                    controller.manual_command.key_name if controller.manual_command else "",
                    label_from_command(controller.manual_command),
                    controller.move_dx,
                    controller.move_dy,
                    controller.view_x,
                    controller.view_y,
                    event="PHASE_START",
                )

            controller.update_view(phase, phase_t)

            logger.log_row(
                controller.trial_idx,
                phase,
                phase_t,
                controller.manual_command.key_name if controller.manual_command else "",
                label_from_command(controller.manual_command),
                controller.move_dx,
                controller.move_dy,
                controller.view_x,
                controller.view_y,
            )

            controller.advance_if_needed()

            if controller.trial_idx >= N_TRIALS:
                running = False

        screen.fill((18, 18, 22))

        if state == "IDLE":
            title = font_big.render("Map-BCI Experiment", True, (240, 240, 240))
            screen.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 180))
            hint = font.render("SPACE or click to start", True, (200, 200, 200))
            screen.blit(hint, (WINDOW_W // 2 - hint.get_width() // 2, 240))
            guide = font_small.render("Manual input only during '手動入力(2s)'", True, (180, 180, 180))
            screen.blit(guide, (WINDOW_W // 2 - guide.get_width() // 2, 280))
            src = font_small.render(f"Map source: {world.source}", True, (160, 160, 160))
            screen.blit(src, (WINDOW_W // 2 - src.get_width() // 2, 320))
            esc = font_small.render("ESC: 現在の試行を破棄して終了", True, (160, 160, 160))
            screen.blit(esc, (WINDOW_W // 2 - esc.get_width() // 2, 350))
        elif state == "COUNTDOWN":
            elapsed = now - countdown_start
            secs_left = max(1, math.ceil(COUNTDOWN_SEC - elapsed))
            label = font_big.render(str(secs_left), True, (240, 240, 240))
            screen.blit(label, (WINDOW_W // 2 - label.get_width() // 2, WINDOW_H // 2 - label.get_height() // 2))
        elif state == "PREFETCH":
            msg = font_big.render("地図タイルを事前取得中...", True, (240, 240, 240))
            screen.blit(msg, (WINDOW_W // 2 - msg.get_width() // 2, 240))
            if prefetcher is not None:
                done, total = prefetcher.progress()
                sub = font.render(f"{done}/{total}", True, (200, 200, 200))
                screen.blit(sub, (WINDOW_W // 2 - sub.get_width() // 2, 300))
        elif state == "RUNNING" and controller is not None:
            world.render(screen, controller.view_x, controller.view_y)
            draw_crosshair(screen)

            phase, phase_t = controller.phase()
            col = PHASE_COLORS.get(phase, (80, 80, 80))
            pygame.draw.rect(screen, col, (0, 0, WINDOW_W, 42))
            label = f"Trial {controller.trial_idx + 1}  Phase: {phase}  (t_in_phase={phase_t:0.2f}s)"
            screen.blit(font_mono.render(label, True, (240, 240, 240)), (14, 10))

            t_in_trial = controller.t_in_trial()
            rem = max(0.0, TOTAL_SEC - t_in_trial)
            screen.blit(font_mono.render(f"{rem:0.1f}s", True, (240, 240, 240)), (WINDOW_W - 120, 10))

            draw_timeline(screen, t_in_trial)

            jp = PHASE_LABELS_JP.get(phase, phase)
            phase_label = font.render(jp, True, (240, 240, 240))
            screen.blit(phase_label, (14, 86))

            cmd_txt = "なし"
            if controller.manual_command is not None:
                cmd_txt = controller.manual_command.key_name
            cmd_label = font_small.render(f"手動入力: {cmd_txt}", True, (30, 30, 30))
            screen.blit(cmd_label, (14, 120))

            snap_hint = font_small.render("Sキーで地図画像保存", True, (30, 30, 30))
            screen.blit(snap_hint, (14, 146))
            esc = font_small.render("ESC: 現在の試行を破棄して終了", True, (30, 30, 30))
            screen.blit(esc, (14, 172))

            if USE_GSI:
                attr = font_small.render(GSI_ATTRIBUTION, True, (255, 255, 255))
                screen.blit(attr, (WINDOW_W - attr.get_width() - 12, WINDOW_H - attr.get_height() - 12))

            if flash_phase:
                flash_t = now - flash_start
                if flash_t <= PHASE_FLASH_SEC:
                    alpha = 1.0 - (flash_t / PHASE_FLASH_SEC)
                    draw_phase_flash(screen, flash_phase, alpha, font_huge)

            if ENABLE_PD and pd_flash_start is not None:
                draw_photodiode(screen, (now - pd_flash_start) <= PD_FLASH_SEC)

        pygame.display.flip()
        clock.tick(FPS)

    if logger is not None:
        logger.close(discard_current=abort_discard)
    pygame.quit()


if __name__ == "__main__":
    main()
