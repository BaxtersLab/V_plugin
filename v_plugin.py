"""
v_plugin.py — Vision plugin for SOC Ultralight
================================================

Adds a 4th agent slot powered by any local vision-capable GGUF model served by
an OpenAI-compatible llama-server endpoint (default: GGUF Chatbox on port 8082).

Provides:
  - Agent 4 floating window (chat + live screen capture + region selector)
  - Mission dispatch: any agent can send `To Agent4 / ... / end message now`
  - Auto-routing of Agent 4 responses back to other agents
  - JSONL session logging of every VLM call (training dataset for later tuning)

Entry point:
    v_plugin.load(socu, config) -> VPlugin instance

The plugin is fully optional. SOCU runs identically without it; with it loaded
SOCU gains an `_vplugin` attribute and `_route_text()` extends to digit "4".
"""
from __future__ import annotations

import base64
import io
import json as _json
import re
import time
import threading
from datetime import datetime
from pathlib import Path

import requests
import tkinter as tk
from tkinter import scrolledtext

import pyperclip
from PIL import Image

# Platform seam (S8): window-under-point + cursor queries go through the SOC
# platform layer (win32 today, X11 on Linux). When run standalone (not via
# SOC), the SOC root two levels up is added to sys.path to find it.
try:
    from platform_layer import get_platform
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from platform_layer import get_platform
PLATFORM = get_platform()

# mss is optional — we fall back to PIL.ImageGrab if not present
try:
    import mss
    _mss_ctor = getattr(mss, "MSS", None) or getattr(mss, "mss", None)
    _MSS_OK = _mss_ctor is not None
except ImportError:
    _MSS_OK = False
    _mss_ctor = None

from PIL import ImageGrab as _PILGrab

# pyautogui is optional — action execution degrades gracefully without it
try:
    import pyautogui as _pag
    _pag.FAILSAFE = True   # move mouse to top-left corner to abort any sequence
    _pag.PAUSE    = 0.05
    _PYAUTOGUI_OK = True
except ImportError:
    _pag = None
    _PYAUTOGUI_OK = False

# ── Action block parsing ──────────────────────────────────────────────────────
# Agent4 can emit a "To Actions … end message now" block to control the desktop.
# Blocks coexist with routing blocks — actions execute first, then route.
_ACT_BLOCK_RE  = re.compile(
    r"(?i)\bto\s+actions?\b(.+?)end\s+message\s+now", re.DOTALL)
_ACT_CLICK_RE  = re.compile(r"(?i)^CLICK\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_ACT_RCLICK_RE = re.compile(r"(?i)^RCLICK\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_ACT_MOVE_RE   = re.compile(r"(?i)^MOVE\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_ACT_TYPE_RE   = re.compile(r'(?i)^TYPE\s*\((.+)\)\s*$')
_ACT_HOTKEY_RE = re.compile(r'(?i)^HOTKEY\s*\((.+)\)\s*$')
_ACT_SCDN_RE   = re.compile(r"(?i)^SCROLL_?DOWN\s*\(\s*(\d+)(?:\s*,\s*(-?\d+)\s*,\s*(-?\d+))?\s*\)")
_ACT_SCUP_RE   = re.compile(r"(?i)^SCROLL_?UP\s*\(\s*(\d+)(?:\s*,\s*(-?\d+)\s*,\s*(-?\d+))?\s*\)")
_ACT_WAIT_RE   = re.compile(r"(?i)^WAIT\s*\(\s*([0-9.]+)\s*\)")
_ACT_SHOT_RE   = re.compile(r"(?i)^SCREENSHOT\s*\(\s*\)")
_ACT_REASON_RE = re.compile(r'(?i)^REASON\s*\((.+)\)\s*$')

# Windows shell classes that are hard-blocked (taskbar, start menu)
_TASKBAR_CLASSES = frozenset({
    "Shell_TrayWnd",
    "Windows.UI.Core.CoreWindow",
    "DV2ControlHost",
    "StartMenuExperienceHost",
    "LauncherTipWnd",
})
# Windows shell classes that are the bare desktop (require permission)
_DESKTOP_CLASSES = frozenset({
    "Progman",
    "WorkerW",
    "SHELLDLL_DefView",
    "SysListView32",
})


# ── Defaults (overridable via config) ─────────────────────────────────────────
DEFAULTS = {
    "vlm_server_url": "http://localhost:8080/v1/chat/completions",
    "vlm_model":      "vision",   # llama-server ignores this; matches GGUF Chatbox convention
    "vlm_timeout":    300.0,   # local reasoning model can think for a while — patient-wait, don't cut it off early
    "vlm_max_tokens": 4096,    # local = no per-token cost; a reasoning model needs room to think
                               # THEN answer (400 truncated it mid-thought → empty content). Bounded
                               # only for sanity; the model stops naturally (finish_reason=stop).
    "vlm_temperature": 0.3,
}

# Ordered list of fallback VLM endpoints to probe at startup.
# Port 8080 = GGUF Chatbox main model proxy (preferred — loads whatever model is in the tray).
# Port 8082 = standalone vision server (start_vlm_server.py or GGUF Chatbox vision server).
_FALLBACK_URLS = [
    "http://localhost:8080/v1/chat/completions",
    "http://localhost:8082/v1/chat/completions",
]


def _probe_port(port: int) -> bool:
    """Return True if something is listening on localhost:port (TCP connect, 0.5 s timeout)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _endpoint_vision_capable(chat_url: str, timeout: float = 2.0) -> bool:
    """True if this endpoint's /v1/models reports a vision-capable model.

    Lets A4 auto-find the vision model whether it is loaded in the GGUF Chatbox
    *main* slot (:8080, use_main_for_vision) or a dedicated *vision* port
    (:8082) — the resolver prefers whichever endpoint actually serves vision,
    instead of just whichever port happens to be listening.
    """
    import json as _json
    import urllib.request
    from urllib.parse import urlparse
    try:
        p = urlparse(chat_url)
        models_url = f"{p.scheme or 'http'}://{p.hostname or '127.0.0.1'}:{p.port or 8080}/v1/models"
        with urllib.request.urlopen(models_url, timeout=timeout) as r:
            data = _json.load(r)
    except Exception:
        return False
    # Accept both GGUF Chatbox ({"models":[...]}) and OpenAI ({"data":[...]}) shapes.
    models = data.get("models") or data.get("data") or []
    for m in models:
        caps = m.get("capabilities") or []
        if any(("multimodal" in str(c).lower() or "vision" in str(c).lower())
               for c in caps):
            return True
    return False

# Routing system prompt — instructs the VLM how to delegate findings back into
# the SOC routing protocol. Model-agnostic — works with any instruction-tuned
# vision GGUF (qwen2-vl, llava, minicpm-v, etc.).
AGENT4_SYSTEM_PROMPT = (
    "You are Agent 4 — the visual intelligence and desktop-control agent in a multi-agent system.\n"
    "You can see the screen live and issue actions to control the mouse and keyboard.\n\n"

    "ROUTING FORMAT — send findings back into the agent loop:\n"
    "  To Agent1\n"
    "  [your findings or instructions]\n"
    "  end message now\n\n"
    "Use To Agent1 (planner/context), To Agent2 (builder/implementer), or "
    "To Agent3 (orchestrator/auditor) depending on who needs the information.\n\n"

    "ACTION FORMAT — control the desktop:\n"
    "  To Actions\n"
    "  REASON(one sentence explaining why this action is needed)\n"
    "  CLICK(x, y)          — left-click at screen coordinates\n"
    "  RCLICK(x, y)         — right-click at screen coordinates\n"
    "  MOVE(x, y)           — move mouse without clicking\n"
    "  TYPE(text)           — type text at current focus\n"
    "  HOTKEY(ctrl, c)      — press key combination (comma-separated)\n"
    "  SCROLL_DOWN(n)           — scroll down n clicks at current mouse position\n"
    "  SCROLL_DOWN(n, x, y)     — scroll down n clicks at screen coordinate (x,y)\n"
    "  SCROLL_UP(n)             — scroll up n clicks at current mouse position\n"
    "  SCROLL_UP(n, x, y)       — scroll up n clicks at screen coordinate (x,y)\n"
    "  SCREENSHOT()         — capture screen and attach to your next reply\n"
    "  WAIT(seconds)        — pause (max 10s)\n"
    "  end message now\n\n"

    "SANDBOX RULES — you must follow these exactly:\n"
    "  1. Always include REASON(...) before any CLICK or RCLICK that targets the desktop, "
    "files, folders, or icons. The user will see this reason in an approval dialog.\n"
    "  2. NEVER attempt to click the Windows taskbar, Start button, or system tray — "
    "these are hard off-limits and will be blocked automatically.\n"
    "  3. Clicks inside application windows (agent panels, browsers, editors) do not "
    "require special permission but REASON is still good practice.\n"
    "  4. If you are unsure whether a coordinate is safe, use SCREENSHOT() first to "
    "verify what is visible before acting.\n\n"

    "WORKFLOW SEQUENCE — the 4-agent loop you are embedded in:\n"
    "  Agent 1 (Copilot) → Agent 2 (VS Code Claude Code) → Agent 3 (Claude.ai) → Agent 1\n\n"
    "  Each agent writes a response in the format:\n"
    "    To Agent<N>\n"
    "    [message body]\n"
    "    end message now\n"
    "  SOC reads the response via OCR or clipboard, routes it to the next agent.\n\n"

    "STALL RECOVERY — when SOC cannot complete a step it dispatches you with a stall name.\n"
    "Always SCREENSHOT() first to see the current screen state, then act.\n\n"

    "  copy_button (agent2 — VS Code / Claude Code panel):\n"
    "    The copy button is a small clipboard or overlapping-pages icon that appears\n"
    "    on hover at the bottom-right corner of the last AI response block.\n"
    "    Action: hover over the bottom portion of the response text to reveal the icon,\n"
    "    then CLICK it. The response text will be copied to the clipboard.\n\n"

    "  clipboard_empty (agent1 — Copilot in Edge/Chrome browser):\n"
    "    A click was attempted but clipboard is still empty. The copy icon\n"
    "    appears near the 'end message now' sentinel at the bottom of the response.\n"
    "    It looks like a clipboard or two overlapping document pages.\n"
    "    Action: MOVE to the bottom of the Copilot response area, wait for hover-reveal,\n"
    "    then CLICK the copy icon precisely. If not visible, try a small hover sweep\n"
    "    across the bottom 60px of the response.\n\n"

    "  send_button (any agent):\n"
    "    Text was pasted into the agent input box but the send button was not found.\n"
    "    For Copilot (Edge): a blue filled circle with a right-pointing arrow, appears\n"
    "    at the right end of the input bar after text is entered.\n"
    "    For Claude Code (VS Code): a paper-plane or arrow icon at the right of the input.\n"
    "    For Claude.ai: an upward-arrow button at the right of the input field.\n"
    "    Action: locate the send button and CLICK it. If unsure, SCREENSHOT() first.\n\n"

    "After completing a stall recovery action, route a brief status:\n"
    "  To Agent2\n"
    "  stall resolved: <stall_name> on <agent_id>\n"
    "  end message now\n\n"

    "Action and routing blocks can both appear in a single response — "
    "actions execute first, then the routing block is dispatched.\n\n"
    "If the user is talking to you directly, respond conversationally — "
    "no special format needed unless you want to act or dispatch."
)


# ── Utilities ────────────────────────────────────────────────────────────────
def _img_to_b64(img: Image.Image, max_px: int = 1280) -> str:
    """Encode a PIL image as a base64 PNG string for VLM API calls.
    Downscales if either dimension exceeds max_px to keep visual token count low."""
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _grab_full_or_region(region: tuple | None) -> Image.Image:
    """Capture full desktop, or a sub-region if provided. Uses mss when
    available (faster, multi-monitor aware), else PIL.ImageGrab."""
    if _MSS_OK:
        with _mss_ctor() as sct:
            if region:
                x0, y0, x1, y1 = region
                raw = sct.grab({"left": x0, "top": y0,
                                "width": x1 - x0, "height": y1 - y0})
            else:
                raw = sct.grab(sct.monitors[0])  # all monitors combined
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    # Fallback: PIL.ImageGrab
    if region:
        return _PILGrab.grab(bbox=region)
    return _PILGrab.grab()


# ── Data logger ──────────────────────────────────────────────────────────────
class DataLogger:
    """Logs every VLM call + outcome to a JSONL session file.
    Good detections (routed successfully) vs bad (failed / no route) are
    flagged so the dataset can be split for fine-tuning later."""

    def __init__(self, base_dir: Path):
        self._dir = base_dir / "data_log"
        self._img_dir = self._dir / "images"
        self._dir.mkdir(exist_ok=True)
        self._img_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session = ts
        self._file = self._dir / f"session_{ts}.jsonl"
        self._seq = 0

    def log(self, agent_id: str, prompt: str, response: str,
            image: Image.Image | None, action: str,
            outcome: str = "", inference_ms: float = 0.0,
            extra: dict | None = None):
        """Write one entry. action: 'chat'|'mission'|'observation'|'actions'|'error'.
        outcome (when action=='route'): 'success'|'fail'."""
        self._seq += 1
        img_path = ""
        if image is not None:
            try:
                fname = f"{self._session}_{agent_id}_{self._seq:05d}.png"
                img_path = str(self._img_dir / fname)
                image.save(img_path)
            except Exception:
                img_path = ""
        entry = {
            "ts":           datetime.now().isoformat(timespec="milliseconds"),
            "session":      self._session,
            "seq":          self._seq,
            "agent":        agent_id,
            "prompt":       prompt[:2000],
            "response":     response[:4000],
            "action":       action,
            "outcome":      outcome,
            "inference_ms": round(inference_ms, 1),
            "image":        img_path,
            "quality":      "good" if outcome == "success" else ("bad" if outcome == "fail" else ""),
        }
        if extra:
            entry["extra"] = extra
        try:
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry) + "\n")
        except Exception:
            pass


# ── Permission gate ───────────────────────────────────────────────────────────
class _PermissionGate:
    """Modal permission dialog for desktop-scope actions.

    Background threads call ask() and block until the user responds.
    Auto-denies after TIMEOUT seconds so a hallucinating model cannot hang
    the session indefinitely.
    """
    TIMEOUT = 60

    BG     = "#1e1e1e"
    BG2    = "#2d2d2d"
    FG     = "#d4d4d4"
    ORANGE = "#ce9178"
    RED    = "#f44747"
    GREEN  = "#4ec9b0"

    def __init__(self, parent: tk.Toplevel):
        self._parent  = parent
        self._event   = threading.Event()
        self._result  = False
        self._win: tk.Toplevel | None = None

    def ask(self, action_desc: str, reason: str) -> bool:
        """Show gate and block until user allows or denies. Returns True = allowed."""
        self._event.clear()
        self._result = False
        self._parent.after(0, lambda: self._build(action_desc, reason))
        granted = self._event.wait(timeout=self.TIMEOUT)
        return self._result if granted else False

    def _build(self, action_desc: str, reason: str):
        w = tk.Toplevel(self._parent)
        w.title("Agent 4 — Action Permission Required")
        w.configure(bg=self.BG)
        w.geometry("440x230")
        w.attributes("-topmost", True)
        w.resizable(False, False)
        w.protocol("WM_DELETE_WINDOW", self._deny)
        self._win = w

        tk.Label(w, text="⚠  Agent 4 is requesting desktop access",
                 bg=self.BG, fg=self.ORANGE,
                 font=("Segoe UI", 10, "bold")).pack(pady=(14, 6))

        tk.Label(w, text=action_desc,
                 bg=self.BG2, fg=self.FG,
                 font=("Consolas", 9), wraplength=400,
                 relief="flat", padx=8, pady=4).pack(fill="x", padx=16)

        tk.Label(w, text=reason or "No reason provided.",
                 bg=self.BG, fg="#aaaaaa",
                 font=("Segoe UI", 8, "italic"),
                 wraplength=400).pack(pady=(6, 10))

        btn_row = tk.Frame(w, bg=self.BG)
        btn_row.pack()
        tk.Button(btn_row, text="  Allow  ", command=self._allow,
                  bg="#1a3d1a", fg=self.GREEN,
                  font=("Segoe UI", 9, "bold"), relief="flat",
                  cursor="hand2", padx=14, pady=5).pack(side="left", padx=(0, 14))
        tk.Button(btn_row, text="  Deny  ", command=self._deny,
                  bg="#3d1a1a", fg=self.RED,
                  font=("Segoe UI", 9, "bold"), relief="flat",
                  cursor="hand2", padx=14, pady=5).pack(side="left")

        tk.Label(w, text=f"Auto-denies in {self.TIMEOUT}s with no response.",
                 bg=self.BG, fg="#444444",
                 font=("Segoe UI", 7, "italic")).pack(pady=(8, 0))
        w.grab_set()

    def _allow(self):
        self._result = True
        if self._win:
            self._win.destroy()
        self._event.set()

    def _deny(self):
        self._result = False
        if self._win:
            self._win.destroy()
        self._event.set()


def _classify_click(x: int, y: int, scope: list | None = None) -> str:
    """Classify a screen coordinate for action sandboxing.

    Returns:
      'blocked'          — taskbar / start bar (hard off-limits, always)
      'needs_permission' — bare desktop, icons, shell views, or out-of-scope windows
      'allowed'          — matches a user-registered scope window (or no scope set)
    """
    try:
        got = PLATFORM.window_from_point(x, y)
        if not got:
            return "needs_permission"
        _root, title, cls, _rect = got
        if cls in _TASKBAR_CLASSES:
            return "blocked"
        if cls in _DESKTOP_CLASSES:
            return "needs_permission"
        # Scope list populated → only registered windows are autonomous
        if scope:
            for entry in scope:
                if entry.get("class") == cls:
                    return "allowed"
                stored = entry.get("title", "").lower()
                if stored and stored in title.lower():
                    return "allowed"
            return "needs_permission"
        # No scope configured → any app window is allowed (original behaviour)
        return "allowed"
    except Exception:
        return "needs_permission"


# ── Agent 4 floating window ──────────────────────────────────────────────────
class Agent4Window:
    """Floating vision chat — the eyes of the V plugin.

    Receives missions from the routing loop (any agent can send 'To Agent4').
    Grabs a live screenshot, queries the local VLM, and routes findings back.
    Also usable as a direct chat interface by the user.
    """

    BG    = "#1e1e1e"
    BG2   = "#2d2d2d"
    FG    = "#d4d4d4"
    GREEN = "#4ec9b0"
    ORANGE = "#ce9178"
    YELLOW = "#dcdcaa"
    ACCENT = "#569cd6"
    RED   = "#f44747"

    def __init__(self, parent: tk.Tk, socu_app, plugin: "VPlugin"):
        self.app = socu_app
        self.plugin = plugin
        self._win = tk.Toplevel(parent)
        self._win.title("Agent 4 · Vision")
        self._win.configure(bg=self.BG)
        # Taskbar/titlebar icon: the yellow "A4V" mark (replaces the default
        # python icon). Best-effort — a missing asset must never break the plugin.
        try:
            _ico = Path(__file__).resolve().parent / "assets" / "a4v_icon.ico"
            if _ico.exists():
                self._win.iconbitmap(default=str(_ico))
        except Exception:
            pass
        # Arrive TIGHT: the window opens at its floor width and a height with
        # ~1/3 of the old transcript pane removed (scroll covers the rest) —
        # tight arrival makes multi-window screen layout painless.
        self._win.geometry("380x355")
        # Floor the size so the history can never squeeze the input/send row out of
        # view — below this the window simply won't shrink further.
        self._win.minsize(380, 340)
        self._win.attributes("-topmost", True)
        self._win.protocol("WM_DELETE_WINDOW", self.hide)
        self._win.withdraw()

        self._conversation: list[dict] = []
        self._vision_region: tuple | None = None
        self._last_response: str = ""
        self._pending_screenshot: "Image.Image | None" = None
        self._busy = False
        self._perm_gate = _PermissionGate(self.app.root)

        # Scope: windows agent4 may click autonomously
        _base = Path(getattr(plugin.app, "BASE_DIR",
                              Path(__file__).resolve().parent.parent))
        self._scope_file: Path = _base / "agent4_scope.json"
        self._scope: list[dict] = []
        self._scope_visible = False   # collapsible panel state
        self._scope_list_frame: tk.Frame | None = None

        self._load_scope()
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        W = self._win
        hdr = tk.Frame(W, bg=self.BG2, pady=4)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"👁  Agent 4 · {self.plugin.cfg['vlm_model']}",
                 bg=self.BG2, fg=self.GREEN,
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=10)
        self._status_lbl = tk.Label(hdr, text="● idle",
                                    bg=self.BG2, fg="#555555",
                                    font=("Segoe UI", 8))
        self._status_lbl.pack(side="right", padx=10)

        hist_frame = tk.Frame(W, bg=self.BG)
        hist_frame.pack(fill="both", expand=True, padx=6, pady=(4, 0))
        self._history = scrolledtext.ScrolledText(
            hist_frame, bg=self.BG, fg=self.FG,
            font=("Consolas", 9), wrap="word", height=8,
            relief="flat", state="disabled",
            insertbackground=self.FG)
        self._history.pack(fill="both", expand=True)
        self._history.tag_config("user",     foreground=self.ACCENT)
        self._history.tag_config("agent4",   foreground=self.GREEN)
        self._history.tag_config("mission",  foreground=self.YELLOW)
        self._history.tag_config("system",   foreground="#666666")
        self._history.tag_config("err",      foreground=self.RED)

        route_frame = tk.Frame(W, bg=self.BG, pady=2)
        route_frame.pack(fill="x", padx=6)
        tk.Label(route_frame, text="Route last →",
                 bg=self.BG, fg="#555555",
                 font=("Segoe UI", 8)).pack(side="left")
        for aid, label in [("agent1", "A1"), ("agent2", "A2"), ("agent3", "A3")]:
            tk.Button(
                route_frame, text=f"→ {label}",
                command=lambda a=aid: self._route_last_to(a),
                bg=self.BG2, fg=self.ACCENT,
                font=("Segoe UI", 8), relief="flat",
                cursor="hand2", padx=5, pady=1
            ).pack(side="left", padx=(3, 0))
        tk.Button(
            route_frame, text="📋 Copy", command=self._copy_last,
            bg=self.BG2, fg=self.FG, font=("Segoe UI", 8), relief="flat",
            cursor="hand2", padx=5, pady=1
        ).pack(side="left", padx=(6, 0))
        tk.Button(
            route_frame, text="🗑 Clear", command=self._clear_history,
            bg=self.BG2, fg="#666666", font=("Segoe UI", 8), relief="flat",
            cursor="hand2", padx=5, pady=1
        ).pack(side="right", padx=(0, 2))

        region_frame = tk.Frame(W, bg=self.BG, pady=2)
        region_frame.pack(fill="x", padx=6)
        tk.Label(region_frame, text="👁 Region:",
                 bg=self.BG, fg="#555555",
                 font=("Segoe UI", 8)).pack(side="left")
        self._region_lbl = tk.Label(
            region_frame, text="full desktop",
            bg=self.BG, fg="#555555",
            font=("Segoe UI", 8, "italic"))
        self._region_lbl.pack(side="left", padx=(4, 0))
        tk.Button(
            region_frame, text="✎ Set Region", command=self._set_region,
            bg=self.BG2, fg=self.ORANGE, font=("Segoe UI", 8), relief="flat",
            cursor="hand2", padx=5, pady=1
        ).pack(side="left", padx=(8, 0))
        tk.Button(
            region_frame, text="✕ Clear", command=self._clear_region,
            bg=self.BG2, fg="#666666", font=("Segoe UI", 8), relief="flat",
            cursor="hand2", padx=4, pady=1
        ).pack(side="left", padx=(3, 0))

        # ── Autonomous scope panel (collapsible) ──────────────────────────────
        scope_outer = tk.Frame(W, bg=self.BG2)
        scope_outer.pack(fill="x", padx=6, pady=(4, 0))

        scope_hdr = tk.Frame(scope_outer, bg=self.BG2)
        scope_hdr.pack(fill="x")
        self._scope_toggle = tk.Label(
            scope_hdr, text="▶ Autonomous Scope",
            bg=self.BG2, fg=self.ORANGE,
            font=("Segoe UI", 8, "bold"), cursor="hand2")
        self._scope_toggle.pack(side="left", padx=(6, 0), pady=2)
        self._scope_count_lbl = tk.Label(
            scope_hdr, text="(unconfigured — all app windows allowed)",
            bg=self.BG2, fg="#555555", font=("Segoe UI", 7, "italic"))
        self._scope_count_lbl.pack(side="left", padx=(4, 0))
        tk.Button(
            scope_hdr, text="+ Add Window",
            command=self._add_scope_window,
            bg=self.BG2, fg=self.GREEN,
            font=("Segoe UI", 8), relief="flat",
            cursor="hand2", padx=5, pady=1
        ).pack(side="right", padx=(0, 4))

        self._scope_body = tk.Frame(scope_outer, bg=self.BG2)
        # body starts hidden; toggled by header click
        self._scope_countdown_lbl = tk.Label(
            self._scope_body, text="",
            bg=self.BG2, fg=self.YELLOW,
            font=("Segoe UI", 8, "italic"))
        self._scope_countdown_lbl.pack(fill="x", padx=6, pady=(0, 2))
        self._scope_list_frame = tk.Frame(self._scope_body, bg=self.BG2)
        self._scope_list_frame.pack(fill="x", padx=6, pady=(0, 4))

        self._scope_toggle.bind("<Button-1>", lambda e: self._toggle_scope_panel())
        self._refresh_scope_list()

        input_frame = tk.Frame(W, bg=self.BG, pady=4)
        input_frame.pack(fill="x", padx=6, pady=(2, 6))
        self._input = tk.Text(
            input_frame, bg=self.BG2, fg=self.FG,
            font=("Segoe UI", 9), height=3, relief="flat", wrap="word",
            insertbackground=self.FG)
        self._input.pack(fill="x", expand=True)
        self._input.bind("<Control-Return>", lambda e: self._on_send(vision=True))
        self._input.bind("<Shift-Return>",   lambda e: self._on_send(vision=False))
        # Send buttons on their own full-width row so neither label is clipped.
        # Left (green, 👁) = send WITH a live screenshot (vision); right = text only.
        btn_row = tk.Frame(input_frame, bg=self.BG)
        btn_row.pack(fill="x", pady=(4, 0))
        tk.Button(
            btn_row, text="👁  Send + Screenshot",
            command=lambda: self._on_send(vision=True),
            bg=self.BG2, fg=self.GREEN,
            font=("Segoe UI", 9, "bold"), relief="flat",
            cursor="hand2", pady=5
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        tk.Button(
            btn_row, text="Send (text only)",
            command=lambda: self._on_send(vision=False),
            bg=self.BG2, fg=self.FG,
            font=("Segoe UI", 9), relief="flat",
            cursor="hand2", pady=5
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        self._append_history(
            "system",
            "Agent 4 ready. Ctrl+Enter = send with live screenshot. "
            "Shift+Enter = text only.\n"
            "Other agents dispatch missions via:  To Agent4 / task / end message now\n")

    # ── Show / hide ───────────────────────────────────────────────────────────
    def show(self):
        self._win.deiconify()
        self._win.lift()

    def hide(self):
        self._win.withdraw()

    def toggle(self):
        if self._win.state() == "withdrawn":
            self.show()
        else:
            self.hide()

    # ── History ───────────────────────────────────────────────────────────────
    def _append_history(self, tag: str, text: str):
        def _do():
            self._history.config(state="normal")
            prefix = {"user": "User:   ", "agent4": "VLM:    ",
                      "mission": "Mission:", "system": "──────  ",
                      "err": "Error:  "}.get(tag, "        ")
            self._history.insert("end", f"{prefix} {text}\n", tag)
            # Per-response one-click copy for VLM answers — grab any response without
            # hand-selecting the read-only history.
            if tag == "agent4" and text.strip():
                self._copy_seq = getattr(self, "_copy_seq", 0) + 1
                ctag = f"copybtn_{self._copy_seq}"
                self._history.insert("end", "         📋 copy\n", (ctag, "copylink"))
                self._history.tag_config("copylink", foreground=self.ACCENT, underline=True)
                self._history.tag_bind(ctag, "<Button-1>", lambda e, t=text: self._copy_text(t))
                self._history.tag_bind(ctag, "<Enter>", lambda e: self._history.config(cursor="hand2"))
                self._history.tag_bind(ctag, "<Leave>", lambda e: self._history.config(cursor=""))
            self._history.config(state="disabled")
            self._history.see("end")
        self._win.after(0, _do)

    def _copy_text(self, text: str):
        """Copy one specific response to the clipboard (per-response copy link)."""
        try:
            pyperclip.copy(text)
            self._set_status("● copied ✓", self.GREEN)
            self._win.after(1200, lambda: self._set_status("● idle", "#555555"))
        except Exception as e:
            self._append_history("err", f"copy failed: {e}")

    def _set_status(self, text: str, color: str | None = None):
        def _do():
            self._status_lbl.config(text=text, fg=color or "#555555")
        self._win.after(0, _do)

    # ── Screen grab ───────────────────────────────────────────────────────────
    def _grab_screen(self) -> Image.Image:
        return _grab_full_or_region(self._vision_region)

    # ── Response sanitizer ────────────────────────────────────────────────────
    @staticmethod
    def _sanitize_response(text: str) -> str:
        """Truncate runaway repetition loops.
        GGUF models sometimes lock into repeating a line indefinitely.
        Keep at most 2 occurrences of any single non-empty line, then stop."""
        lines = text.split("\n")
        seen: dict[str, int] = {}
        out: list[str] = []
        for line in lines:
            key = line.strip()
            if key:
                count = seen.get(key, 0) + 1
                seen[key] = count
                if count > 2:
                    break
            out.append(line)
        return "\n".join(out)

    @staticmethod
    def _build_messages(system_prompt, history, user_content):
        """Assemble chat messages with STRICTLY alternating user/assistant roles.
        Strict chat templates (e.g. gemma) raise HTTP 400 on ANY two consecutive
        same-role turns, so consecutive same-role history turns are merged and a
        stale trailing user turn is dropped before the current user turn is added."""
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if (len(messages) > 1 and messages[-1]["role"] == role
                    and role in ("user", "assistant")
                    and isinstance(messages[-1]["content"], str)
                    and isinstance(content, str)):
                messages[-1]["content"] += "\n" + content   # merge same-role turn
            else:
                messages.append({"role": role, "content": content})
        if messages[-1]["role"] == "user":     # avoid user,user with the current turn
            messages.pop()
        messages.append({"role": "user", "content": user_content})
        return messages

    # ── Inference awareness (graceful-wait) ─────────────────────────────────────
    def _slots_url(self) -> str:
        """Derive the llama-server /slots URL from the configured chat endpoint."""
        from urllib.parse import urlparse
        p = urlparse(self.plugin.cfg["vlm_server_url"])
        return f"{p.scheme or 'http'}://{p.hostname or '127.0.0.1'}:{p.port or 8080}/slots"

    @staticmethod
    def _parse_slots(slots_json) -> tuple[bool, int, int] | None:
        """(is_processing, prompt_tokens_processed, prompt_tokens) from a /slots
        payload, or None if unparseable. Pure logic → unit-testable."""
        try:
            s = slots_json[0]
            return (bool(s.get("is_processing")),
                    int(s.get("n_prompt_tokens_processed") or 0),
                    int(s.get("n_prompt_tokens") or 0))
        except Exception:
            return None

    def _poll_inference_state(self) -> tuple[bool, int, int] | None:
        """Query /slots once. Returns (is_processing, processed, total) or None."""
        try:
            r = requests.get(self._slots_url(), timeout=2)
            return self._parse_slots(r.json())
        except Exception:
            return None

    def _monitor_inference(self, stop_event):
        """While a (blocking) VLM call runs, poll /slots and surface live progress
        so the operator SEES the server working — never a frozen 'thinking…'. This
        is the graceful-wait the operator requires; degrades silently if /slots is
        unavailable."""
        while not stop_event.wait(2.0):
            st = self._poll_inference_state()
            if st is None:
                continue
            busy, done, total = st
            if busy and total and done < total:
                self._set_status(f"● inferencing — prefill {int(done * 100 / max(total, 1))}%",
                                 self.ACCENT)
            elif busy:
                self._set_status("● inferencing — generating…", self.ACCENT)

    # ── VLM call ──────────────────────────────────────────────────────────────
    def _call_vlm(self, prompt: str, image: Image.Image | None = None) -> str:
        """POST prompt (+ optional screenshot) to llama-server. Returns response text."""
        user_content: list = []
        if image is not None:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_img_to_b64(image)}"}
            })
        user_content.append({"type": "text", "text": prompt})

        messages = self._build_messages(
            AGENT4_SYSTEM_PROMPT, self._conversation[-12:], user_content)

        payload = {
            "model":         self.plugin.cfg["vlm_model"],
            "messages":      messages,
            "max_tokens":    int(self.plugin.cfg["vlm_max_tokens"]),
            "temperature":   float(self.plugin.cfg["vlm_temperature"]),
            "repeat_penalty": float(self.plugin.cfg.get("vlm_repeat_penalty", 1.3)),
        }
        # Start the inference-awareness monitor so the blocking call below shows
        # live server progress instead of a frozen status (operator's graceful-wait
        # requirement). Always stopped in `finally`.
        _stop_mon = threading.Event()
        _mon = threading.Thread(target=self._monitor_inference, args=(_stop_mon,), daemon=True)
        _mon.start()
        try:
            resp = requests.post(
                self.plugin.cfg["vlm_server_url"],
                json=payload,
                timeout=float(self.plugin.cfg["vlm_timeout"]),
            )
        except requests.exceptions.ConnectionError as e:
            url = self.plugin.cfg["vlm_server_url"]
            raise RuntimeError(
                f"Vision server not reachable at {url}.\n"
                f"  • Start GGUF Chatbox\n"
                f"  • Open the Server tray → Vision Server section\n"
                f"  • Set model + mmproj paths, click Start Vision Server\n"
                f"  • Confirm port 8082 is free\n"
                f"(underlying error: {e.__class__.__name__})"
            ) from e
        except requests.exceptions.Timeout as e:
            # Graceful: if the server is STILL inferencing, it's slow, not dead —
            # say so, and don't imply failure. Otherwise report the plain timeout.
            st = self._poll_inference_state()
            if st and st[0]:
                raise RuntimeError(
                    f"Vision server is STILL inferencing after "
                    f"{self.plugin.cfg['vlm_timeout']}s — it is working, just slow "
                    f"(not stalled). Raise vlm_timeout if this recurs."
                ) from e
            raise RuntimeError(
                f"Vision server timed out after "
                f"{self.plugin.cfg['vlm_timeout']}s with no active inference. The "
                f"model may be loading or the prompt may be too long. Increase "
                f"vlm_timeout in config.json if this persists."
            ) from e
        finally:
            _stop_mon.set()
        resp.raise_for_status()
        data = resp.json()
        if "choices" not in data:
            # llama-server returns HTTP 200 with an {"error": {...}} body when it
            # can't serve the request (e.g. no mmproj loaded → "image input is not
            # supported"). Surface that message instead of a cryptic KeyError 'choices'.
            err = data.get("error")
            emsg = err.get("message") if isinstance(err, dict) else (err or data)
            raise RuntimeError(
                f"Vision server returned an error instead of a completion: {emsg}")
        return data["choices"][0]["message"]["content"].strip()

    # ── Action executor ───────────────────────────────────────────────────────
    def _execute_actions(self, block: str) -> list[str]:
        """Parse and execute one To Actions block. Returns list of completed steps.
        Runs on the background _send thread — never call from the Tk main thread.

        Sandbox rules enforced here:
          - Taskbar / start bar coordinates are hard-blocked, action skipped + logged.
          - Desktop coordinates (icons, shell views) require explicit user approval
            via _PermissionGate before the click proceeds.
          - REASON(...) lines set the explanation shown in the permission dialog.
        """
        if not _PYAUTOGUI_OK:
            self._append_history("err", "action execution unavailable — pyautogui not installed")
            return []

        sw, sh     = _pag.size()
        executed:  list[str] = []
        cur_reason = "No reason provided."

        def _clamp(v, lo, hi):
            return max(lo, min(v, hi))

        def _gate_click(x: int, y: int, label: str) -> bool:
            """Return True if the click is allowed to proceed."""
            scope = _classify_click(x, y, self._scope if self._scope else None)
            if scope == "blocked":
                msg = f"BLOCKED — taskbar/start bar is off-limits: {label}"
                self._append_history("err", msg)
                try:
                    self.app._log(f"[agent4] {msg}")
                except Exception:
                    pass
                return False
            if scope == "needs_permission":
                granted = self._perm_gate.ask(
                    action_desc=label,
                    reason=cur_reason,
                )
                if not granted:
                    msg = f"DENIED by user: {label}"
                    self._append_history("system", msg)
                    try:
                        self.app._log(f"[agent4] {msg}")
                    except Exception:
                        pass
                    return False
            return True

        for raw in block.strip().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # REASON — sets context for the next permission dialog
            m = _ACT_REASON_RE.match(line)
            if m:
                cur_reason = m.group(1).strip()
                continue

            m = _ACT_CLICK_RE.match(line)
            if m:
                x, y = _clamp(int(m.group(1)), 0, sw-1), _clamp(int(m.group(2)), 0, sh-1)
                if _gate_click(x, y, f"CLICK({x}, {y})"):
                    _pag.click(x, y)
                    executed.append(f"CLICK({x},{y})")
                    time.sleep(0.15)
                continue

            m = _ACT_RCLICK_RE.match(line)
            if m:
                x, y = _clamp(int(m.group(1)), 0, sw-1), _clamp(int(m.group(2)), 0, sh-1)
                if _gate_click(x, y, f"RCLICK({x}, {y})"):
                    _pag.rightClick(x, y)
                    executed.append(f"RCLICK({x},{y})")
                    time.sleep(0.15)
                continue

            m = _ACT_MOVE_RE.match(line)
            if m:
                x, y = _clamp(int(m.group(1)), 0, sw-1), _clamp(int(m.group(2)), 0, sh-1)
                _pag.moveTo(x, y)
                executed.append(f"MOVE({x},{y})")
                time.sleep(0.1)
                continue

            m = _ACT_TYPE_RE.match(line)
            if m:
                text = m.group(1).strip().strip('"\'')
                _pag.write(text, interval=0.03)
                executed.append(f"TYPE({text!r})")
                continue

            m = _ACT_HOTKEY_RE.match(line)
            if m:
                keys = [k.strip() for k in m.group(1).split(",")]
                _pag.hotkey(*keys)
                executed.append(f"HOTKEY({'+'.join(keys)})")
                time.sleep(0.1)
                continue

            m = _ACT_SCDN_RE.match(line)
            if m:
                n = int(m.group(1))
                if m.group(2) and m.group(3):
                    x, y = _clamp(int(m.group(2)), 0, sw-1), _clamp(int(m.group(3)), 0, sh-1)
                    _pag.scroll(-n, x=x, y=y)
                    executed.append(f"SCROLL_DOWN({n},{x},{y})")
                else:
                    _pag.scroll(-n)
                    executed.append(f"SCROLL_DOWN({n})")
                time.sleep(0.1)
                continue

            m = _ACT_SCUP_RE.match(line)
            if m:
                n = int(m.group(1))
                if m.group(2) and m.group(3):
                    x, y = _clamp(int(m.group(2)), 0, sw-1), _clamp(int(m.group(3)), 0, sh-1)
                    _pag.scroll(n, x=x, y=y)
                    executed.append(f"SCROLL_UP({n},{x},{y})")
                else:
                    _pag.scroll(n)
                    executed.append(f"SCROLL_UP({n})")
                time.sleep(0.1)
                continue

            m = _ACT_WAIT_RE.match(line)
            if m:
                secs = min(float(m.group(1)), 10.0)
                time.sleep(secs)
                executed.append(f"WAIT({secs})")
                continue

            if _ACT_SHOT_RE.match(line):
                try:
                    self._pending_screenshot = _grab_full_or_region(self._vision_region)
                    executed.append("SCREENSHOT()")
                except Exception as se:
                    self._append_history("err", f"screenshot failed: {se}")
                continue

            try:
                self.app._log(f"[agent4] unknown action line: {line!r}")
            except Exception:
                pass

        return executed

    # ── Vision query (headless, synchronous) ──────────────────────────────────
    def query_vision(self, prompt: str, region: tuple | None = None,
                     attach_screenshot: bool = True) -> str:
        """Blocking vision query — callable by other SOCU components.
        Returns the raw VLM response string, or an error message prefixed 'ERROR:'."""
        img = None
        if attach_screenshot:
            try:
                img = _grab_full_or_region(region or self._vision_region)
            except Exception as e:
                return f"ERROR: screenshot failed — {e}"
        try:
            return self._call_vlm(prompt, img)
        except Exception as e:
            return f"ERROR: {e}"

    # ── Send ──────────────────────────────────────────────────────────────────
    def _on_send(self, vision: bool = True):
        prompt = self._input.get("1.0", "end").strip()
        if not prompt:
            return
        self._input.delete("1.0", "end")
        threading.Thread(
            target=self._send, args=(prompt, vision), daemon=True).start()

    def _send(self, prompt: str, vision: bool = True,
              source_agent: str | None = None, auto_route: bool = False):
        if self._busy:
            self._append_history("system", "⏳ busy — previous call still running")
            return
        self._busy = True
        self._set_status("● thinking…", self.YELLOW)

        # Reset context per ping-pong: each exchange starts fresh. A4 looks at the
        # screen anew every time, so it needs no memory of prior exchanges — and
        # carrying them (especially a prior full-screen image) would grow the
        # prompt until a follow-up action gets truncated. The action/observe loop
        # WITHIN this exchange still builds context below; it is cleared again on
        # the next send. (Manual "clear history" also resets it.)
        self._conversation.clear()

        img = None
        if vision:
            try:
                img = self._grab_screen()
            except Exception as e:
                self._append_history("err", f"screen grab failed: {e}")

        tag = "mission" if source_agent else "user"
        label = f"[from {source_agent}] " if source_agent else ""
        self._append_history(tag, f"{label}{prompt}" + (" 📷" if img else ""))
        # The current user turn is recorded AFTER a successful call (below), NOT
        # here: _call_vlm already appends the current prompt as a user turn, so
        # adding it here too created TWO consecutive user turns — which strict chat
        # templates (gemma) reject with HTTP 400 "roles must alternate user/assistant".

        t0 = time.time()
        try:
            response = self._call_vlm(prompt, img)
        except Exception as e:
            self._append_history("err", str(e))
            self._set_status("● error", self.RED)
            self.plugin.logger.log(
                "agent4", prompt, "", img, "error",
                outcome="fail",
                inference_ms=(time.time() - t0) * 1000.0,
                extra={"exception": str(e)})
            self._busy = False
            return
        inference_ms = (time.time() - t0) * 1000.0

        # Sanitize repetition loops — GGUF models can lock into repeating the
        # same line indefinitely. Truncate at the third occurrence of any line.
        response = self._sanitize_response(response)

        self._last_response = response
        # Record the turn only now that it succeeded — user first, then assistant,
        # preserving strict user/assistant alternation for the next call. History
        # keeps the text form; the screenshot itself is not stored.
        self._conversation.append({
            "role": "user",
            "content": prompt + (" [screenshot attached]" if img else ""),
        })
        self._conversation.append({"role": "assistant", "content": response})
        self._append_history("agent4", response)
        self._set_status("● idle", "#555555")
        self._busy = False

        action_type = "mission" if source_agent else "chat"
        self.plugin.logger.log(
            "agent4", prompt, response, img, action_type,
            inference_ms=inference_ms)

        # ── Execute any action blocks first ───────────────────────────────────
        act_match = _ACT_BLOCK_RE.search(response)
        if act_match:
            self._set_status("● acting…", self.ORANGE)
            try:
                steps = self._execute_actions(act_match.group(1))
                if steps:
                    summary = " → ".join(steps)
                    self._append_history("system", f"⚡ {len(steps)} action(s): {summary}")
                    try:
                        self.app._log(f"[agent4] actions executed: {steps}")
                    except Exception:
                        pass
                    self.plugin.logger.log(
                        "agent4", prompt, response, img, "actions",
                        outcome="success", inference_ms=inference_ms,
                        extra={"steps": steps})
            except Exception as ae:
                self._append_history("err", f"action error: {ae}")
                try:
                    self.app._log(f"[agent4] action execution error: {ae}")
                except Exception:
                    pass

            # If SCREENSHOT() was called during the action block, do a follow-up
            # observation pass so the model can see what changed after its actions.
            if self._pending_screenshot is not None:
                follow_img = self._pending_screenshot
                self._pending_screenshot = None
                self._set_status("● observing…", self.ACCENT)
                try:
                    obs_response = self._call_vlm(
                        "You just executed a set of actions. Observe the current screen "
                        "and report what changed. Route findings or issue further actions as needed.",
                        follow_img)
                    self._last_response = obs_response
                    self._conversation.append({"role": "assistant", "content": obs_response})
                    self._append_history("agent4", f"[post-action] {obs_response}")
                    self.plugin.logger.log(
                        "agent4", "[post-action observation]", obs_response,
                        follow_img, "observation", inference_ms=0.0)
                    if auto_route:
                        obs_m = re.search(
                            r"(?i)\bto\s+agent\s*([1-4])\b(.+?)end\s+message\s+now",
                            obs_response, re.DOTALL)
                        if obs_m and obs_m.group(1) != "4":
                            self._route_last_to(
                                f"agent{obs_m.group(1)}", body=obs_m.group(2).strip())
                except Exception as oe:
                    self._append_history("err", f"post-action observation error: {oe}")

            self._set_status("● idle", "#555555")

        # ── Auto-route any routing block ──────────────────────────────────────
        if auto_route:
            m = re.search(
                r"(?i)\bto\s+agent\s*([1-4])\b(.+?)end\s+message\s+now",
                response, re.DOTALL)
            if m:
                digit = m.group(1)
                body = m.group(2).strip()
                target = f"agent{digit}"
                if target == "agent4":
                    self._append_history(
                        "system", "auto-route skipped — refusing self-route")
                else:
                    self.app._log(
                        f"[agent4] auto-routing response to {target} "
                        f"({len(body)} chars)")
                    self._route_last_to(target, body=body)

    # ── Routing ───────────────────────────────────────────────────────────────
    def receive_mission(self, mission: str, source_agent: str, silent: bool = False):
        """Called by SOC routing when a message is addressed 'To Agent4'."""
        if not silent:
            self.show()
        self._append_history("system",
            f"── mission received from {source_agent} ──")
        threading.Thread(
            target=self._send,
            args=(mission, True, source_agent, True),
            daemon=True).start()

    def _route_last_to(self, agent_id: str, body: str | None = None):
        text = body or self._last_response
        if not text:
            self._append_history("system", "nothing to route yet")
            return
        # Self-modification gate (if present on host app)
        gate = getattr(self.app, "_self_mod_gate", None)
        if gate is not None:
            ok = gate.check_and_prompt(
                source_agent="agent4",
                dest_agent=agent_id,
                body=text)
            if not ok:
                self._append_history(
                    "system",
                    f"→ {agent_id} BLOCKED by self-modification gate")
                return
        self.app._log(f"[agent4] routing to {agent_id}: {len(text)} chars")
        threading.Thread(
            target=self.app._inject_to_agent,
            args=(agent_id, text), daemon=True).start()
        self._append_history("system", f"→ routed to {agent_id}")

    def _copy_last(self):
        """Copy the last response — or, when there is none (fresh window,
        cleared history, failed call), the full window transcript. Never a
        silent no-op: it fills the clipboard or says why it couldn't."""
        text = self._last_response
        label = "last response"
        if not text:
            text = self._history.get("1.0", "end").strip()
            label = "window text"
        if not text:
            self._append_history("system", "nothing to copy yet")
            return
        try:
            pyperclip.copy(text)
            self._append_history("system",
                                 f"{label} copied to clipboard ({len(text)} chars)")
        except Exception as e:
            self._append_history("err", f"copy failed: {e}")

    def _clear_history(self):
        self._history.config(state="normal")
        self._history.delete("1.0", "end")
        self._history.config(state="disabled")
        self._conversation.clear()
        self._last_response = ""
        self._append_history("system", "history cleared")

    # ── Scope: load / save ────────────────────────────────────────────────────
    def _load_scope(self):
        try:
            if self._scope_file.exists():
                self._scope = _json.loads(self._scope_file.read_text(encoding="utf-8"))
        except Exception:
            self._scope = []

    def _save_scope(self):
        try:
            self._scope_file.write_text(
                _json.dumps(self._scope, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── Scope: UI helpers ─────────────────────────────────────────────────────
    def _toggle_scope_panel(self):
        self._scope_visible = not self._scope_visible
        if self._scope_visible:
            self._scope_body.pack(fill="x")
            self._scope_toggle.config(text="▼ Autonomous Scope")
        else:
            self._scope_body.pack_forget()
            self._scope_toggle.config(text="▶ Autonomous Scope")

    def _refresh_scope_list(self):
        if self._scope_list_frame is None:
            return
        for w in self._scope_list_frame.winfo_children():
            w.destroy()
        if not self._scope:
            self._scope_count_lbl.config(
                text="(unconfigured — all app windows allowed)", fg="#555555")
            return
        n = len(self._scope)
        self._scope_count_lbl.config(
            text=f"({n} window{'s' if n != 1 else ''} — strict mode active)",
            fg=self.GREEN)
        for i, entry in enumerate(self._scope):
            row = tk.Frame(self._scope_list_frame, bg=self.BG2)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=entry.get("display", entry.get("class", "?")),
                     bg=self.BG2, fg=self.FG,
                     font=("Segoe UI", 8), anchor="w").pack(side="left", fill="x", expand=True)
            tk.Button(row, text="✕",
                      command=lambda idx=i: self._remove_scope(idx),
                      bg=self.BG2, fg="#666666",
                      font=("Segoe UI", 8), relief="flat",
                      cursor="hand2", padx=4).pack(side="right")

    def _remove_scope(self, idx: int):
        if 0 <= idx < len(self._scope):
            removed = self._scope.pop(idx)
            self._save_scope()
            self._refresh_scope_list()
            self._append_history("system", f"scope removed: {removed.get('display', '?')}")

    # ── Scope: calibration ────────────────────────────────────────────────────
    def _add_scope_window(self):
        if not self._scope_visible:
            self._toggle_scope_panel()
        threading.Thread(target=self._add_scope_thread, daemon=True).start()

    def _add_scope_thread(self):
        for i in range(3, 0, -1):
            self._win.after(0, lambda n=i: self._scope_countdown_lbl.config(
                text=f"Move cursor to target window…  {n}"))
            time.sleep(1)
        self._win.after(0, lambda: self._scope_countdown_lbl.config(text=""))
        try:
            x, y = PLATFORM.cursor_pos()
            got = PLATFORM.window_from_point(x, y)
            if not got:
                self._win.after(0, lambda: self._scope_countdown_lbl.config(
                    text="No window found at cursor position."))
                return
            root, title, cls, _rect = got

            if cls in _TASKBAR_CLASSES or cls in _DESKTOP_CLASSES:
                self._win.after(0, lambda: self._scope_countdown_lbl.config(
                    text="Cannot add system shell — point to an app window."))
                return

            # Don't add the agent4 window itself. Win32-only best-effort
            # (ctypes.windll is absent on Linux → AttributeError → skipped);
            # harmless everywhere, no platform-layer equivalent needed for v1.
            try:
                import ctypes
                own = ctypes.windll.user32.GetParent(int(self._win.wm_frame(), 16))
                if root == own:
                    self._win.after(0, lambda: self._scope_countdown_lbl.config(
                        text="Cannot add this window to its own scope."))
                    return
            except Exception:
                pass

            short = title[:50] + "…" if len(title) > 50 else title
            display = f"{short}  [{cls}]"
            entry = {"class": cls, "title": title[:80], "display": display}

            for existing in self._scope:
                if existing.get("class") == cls and existing.get("title") == entry["title"]:
                    self._win.after(0, lambda: self._scope_countdown_lbl.config(
                        text="Already in scope."))
                    return

            self._scope.append(entry)
            self._save_scope()
            self._win.after(0, self._refresh_scope_list)
            self._append_history("system", f"scope added: {display}")
        except Exception as e:
            self._win.after(0, lambda: self._scope_countdown_lbl.config(
                text=f"Capture error: {e}"))

    # ── Region selector ──────────────────────────────────────────────────────
    def _set_region(self):
        self._append_history("system",
            "Click and drag to set the vision focus region…")
        threading.Thread(target=self._draw_region_thread, daemon=True).start()

    def _draw_region_thread(self):
        try:
            overlay = tk.Toplevel(self._win)
            overlay.attributes("-fullscreen", True)
            overlay.attributes("-alpha", 0.25)
            overlay.attributes("-topmost", True)
            overlay.configure(bg="#000030")
            canvas = tk.Canvas(overlay, cursor="crosshair",
                               bg="#000030", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            rect_id = [None]
            start = [0, 0]

            def _press(e):
                start[0], start[1] = e.x_root, e.y_root
                if rect_id[0]:
                    canvas.delete(rect_id[0])

            def _drag(e):
                if rect_id[0]:
                    canvas.delete(rect_id[0])
                rx = canvas.winfo_rootx()
                ry = canvas.winfo_rooty()
                rect_id[0] = canvas.create_rectangle(
                    start[0] - rx, start[1] - ry,
                    e.x_root - rx, e.y_root - ry,
                    outline="#4ec9b0", width=2, fill="#4ec9b020")

            def _release(e):
                x0 = min(start[0], e.x_root)
                y0 = min(start[1], e.y_root)
                x1 = max(start[0], e.x_root)
                y1 = max(start[1], e.y_root)
                overlay.destroy()
                if x1 - x0 > 20 and y1 - y0 > 20:
                    self._vision_region = (x0, y0, x1, y1)
                    w, h = x1 - x0, y1 - y0
                    self._region_lbl.config(
                        text=f"{w}×{h}px ({x0},{y0})", fg=self.GREEN)
                    self._append_history(
                        "system", f"vision region set: {w}×{h}px at ({x0},{y0})")

            canvas.bind("<ButtonPress-1>",   _press)
            canvas.bind("<B1-Motion>",       _drag)
            canvas.bind("<ButtonRelease-1>", _release)
            overlay.bind("<Escape>", lambda e: overlay.destroy())
        except Exception as ex:
            self._append_history("err", f"region draw failed: {ex}")

    def _clear_region(self):
        self._vision_region = None
        self._region_lbl.config(text="full desktop", fg="#555555")
        self._append_history("system", "vision region cleared — using full desktop")


# ── Plugin entry ─────────────────────────────────────────────────────────────
class VPlugin:
    """Container for plugin state attached to SOCU as `socu._vplugin`."""

    name = "v_plugin"
    version = "0.2.0"

    def __init__(self, socu_app, config: dict):
        self.app = socu_app
        # Merge defaults with provided config
        self.cfg = dict(DEFAULTS)
        for k, v in (config or {}).items():
            if k in DEFAULTS and v is not None:
                self.cfg[k] = v
        # Resolve the best available VLM endpoint at startup.
        # If the saved/configured URL's port isn't listening, try fallback ports
        # so that loading Phi-4 in the GGUF Chatbox main model tray (port 8080)
        # is detected automatically without requiring the standalone vision server.
        self.cfg["vlm_server_url"] = self._resolve_vlm_url()
        base = Path(getattr(socu_app, "BASE_DIR", Path(__file__).resolve().parent.parent))
        if not isinstance(base, Path):
            base = Path(base)
        self.logger = DataLogger(base)
        # Build UI window
        self.agent4_window = Agent4Window(socu_app.root, socu_app, self)
        try:
            socu_app._log(
                f"[v_plugin] loaded · model={self.cfg['vlm_model']} "
                f"endpoint={self.cfg['vlm_server_url']}")
        except Exception:
            pass

    def _resolve_vlm_url(self) -> str:
        """Return the best reachable VLM endpoint, preferring a vision-capable one.

        Order: configured URL first, then the fallback list. A candidate wins if
        it is listening AND its /v1/models reports a vision model — so A4 finds
        the model whether it is loaded in the main slot (:8080) or a dedicated
        vision port (:8082). If nothing reports vision yet (model not loaded),
        falls back to the first listening port; the lazy error at call time
        guides the user. Called at load and again when the A4 window is raised.
        """
        from urllib.parse import urlparse
        configured = self.cfg["vlm_server_url"]
        try:
            configured_port = urlparse(configured).port or 8080
        except Exception:
            configured_port = 8080

        # Ordered, de-duplicated candidates: configured first, then fallbacks.
        candidates = [configured]
        for url in _FALLBACK_URLS:
            if url not in candidates:
                candidates.append(url)

        # 1) Prefer an endpoint that actually serves vision (main OR vision port).
        for url in candidates:
            try:
                port = urlparse(url).port or 8080
            except Exception:
                continue
            if _probe_port(port) and _endpoint_vision_capable(url):
                if url != configured:
                    try:
                        self.app._log(f"[v_plugin] vision model found on {url} — using it")
                    except Exception:
                        pass
                return url

        # 2) None report vision yet — use the first listening port so a later
        #    model load still works; the lazy connection error will guide setup.
        for url in candidates:
            try:
                port = urlparse(url).port or 8080
            except Exception:
                continue
            if _probe_port(port):
                if url != configured:
                    try:
                        self.app._log(
                            f"[v_plugin] no vision endpoint live yet — "
                            f"auto-selected listening {url}")
                    except Exception:
                        pass
                return url

        return configured

    def route_to_agent4(self, body: str, source_agent: str | None = None,
                        silent: bool = False) -> bool:
        """Called by SOCU's _route_text when destination digit == '4'."""
        try:
            self.agent4_window.receive_mission(body, source_agent or "unknown",
                                               silent=silent)
            return True
        except Exception as e:
            try:
                self.app._log(f"[v_plugin] route_to_agent4 error: {e}")
            except Exception:
                pass
            return False

    def nudge_stall(self, stall_name: str, agent_id: str, context: str = "") -> bool:
        """Dispatch Agent 4 to visually recover a stalled workflow step.

        stall_name: short label matching one of the STALL RECOVERY entries in the
                    system prompt (e.g. 'copy_button', 'clipboard_empty', 'send_button').
        agent_id:   which agent stalled (e.g. 'agent1', 'agent2').
        context:    optional extra detail about what was on screen / what was tried.
        Returns True if the dispatch succeeded (Agent 4 loaded and not busy).
        """
        win = getattr(self, "agent4_window", None)
        if win is None:
            return False
        if getattr(win, "_busy", False):
            try:
                self.app._log(
                    f"[v_plugin] nudge_stall({stall_name}) skipped — Agent4 already busy")
            except Exception:
                pass
            return False
        mission = (
            f"STALL: {stall_name} on {agent_id}\n\n"
            f"{context}\n\n"
            "Take a SCREENSHOT() to see the current screen state, identify the "
            "element described above, then perform the action to unblock the sequence."
        )
        try:
            self.app._log(
                f"[v_plugin] nudge_stall({stall_name}, {agent_id}) — dispatching Agent4")
        except Exception:
            pass
        return self.route_to_agent4(mission, source_agent="soc_stall_handler",
                                    silent=True)

    def toggle_window(self):
        self.agent4_window.toggle()

    def refresh_endpoint(self) -> str:
        """Re-detect the vision endpoint (main :8080 or dedicated vision :8082)
        and update cfg. Called when the A4 window is raised (e.g. from the master
        widget) so a model loaded after startup is picked up without a restart."""
        url = self._resolve_vlm_url()
        self.cfg["vlm_server_url"] = url
        return url

    def show_window(self):
        """Bring the A4 window to front and re-resolve the endpoint. This is the
        target of the master widget's 'Show A4' control (via SOC's signal poll)."""
        try:
            self.refresh_endpoint()
        except Exception:
            pass
        try:
            self.agent4_window.show()
        except Exception:
            # Fall back to toggle if a dedicated show() isn't available.
            try:
                self.agent4_window.toggle()
            except Exception:
                pass


def load(socu_app, config: dict | None = None) -> VPlugin:
    """Plugin entry point. Returns a VPlugin instance bound to `socu_app`."""
    return VPlugin(socu_app, config or {})
