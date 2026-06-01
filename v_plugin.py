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
import threading
from datetime import datetime
from pathlib import Path

import requests
import tkinter as tk
from tkinter import scrolledtext

import pyperclip
from PIL import Image

# mss is optional — we fall back to PIL.ImageGrab if not present
try:
    import mss
    _mss_ctor = getattr(mss, "MSS", None) or getattr(mss, "mss", None)
    _MSS_OK = _mss_ctor is not None
except ImportError:
    _MSS_OK = False
    _mss_ctor = None

from PIL import ImageGrab as _PILGrab


# ── Defaults (overridable via config) ─────────────────────────────────────────
DEFAULTS = {
    "vlm_server_url": "http://localhost:8082/v1/chat/completions",
    "vlm_model":      "vision",   # llama-server ignores this; matches GGUF Chatbox convention
    "vlm_timeout":    30.0,
    "vlm_max_tokens": 1024,
    "vlm_temperature": 0.3,
}

# Routing system prompt — instructs the VLM how to delegate findings back into
# the SOC routing protocol. Model-agnostic — works with any instruction-tuned
# vision GGUF (qwen2-vl, llava, minicpm-v, etc.).
AGENT4_SYSTEM_PROMPT = (
    "You are Agent 4 — the visual intelligence agent in a 4-agent system.\n"
    "You can see the screen live. When sent on a mission by another agent, "
    "observe what is visible, analyse it carefully, and report your findings.\n\n"
    "ROUTING FORMAT — use this when sending results back into the agent loop:\n"
    "  To Agent1\n"
    "  [your findings or instructions]\n"
    "  end message now\n\n"
    "Use To Agent1 (planner/context), To Agent2 (builder/implementer), or "
    "To Agent3 (orchestrator/auditor) depending on who needs the information.\n"
    "If the user is talking to you directly, respond conversationally — "
    "no routing format needed unless you want to dispatch to another agent."
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
        """Write one entry. action: 'chat'|'mission'|'route'|'error'.
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
        self._win.geometry("520x580")
        self._win.attributes("-topmost", True)
        self._win.protocol("WM_DELETE_WINDOW", self.hide)
        self._win.withdraw()

        self._conversation: list[dict] = []
        self._vision_region: tuple | None = None
        self._last_response: str = ""
        self._busy = False

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
            font=("Consolas", 9), wrap="word",
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

        input_frame = tk.Frame(W, bg=self.BG, pady=4)
        input_frame.pack(fill="x", padx=6, pady=(2, 6))
        self._input = tk.Text(
            input_frame, bg=self.BG2, fg=self.FG,
            font=("Segoe UI", 9), height=3, relief="flat", wrap="word",
            insertbackground=self.FG)
        self._input.pack(side="left", fill="x", expand=True)
        self._input.bind("<Control-Return>", lambda e: self._on_send(vision=True))
        self._input.bind("<Shift-Return>",   lambda e: self._on_send(vision=False))
        btn_col = tk.Frame(input_frame, bg=self.BG)
        btn_col.pack(side="left", padx=(4, 0))
        tk.Button(
            btn_col, text="Send 👁",
            command=lambda: self._on_send(vision=True),
            bg=self.BG2, fg=self.GREEN,
            font=("Segoe UI", 9, "bold"), relief="flat",
            cursor="hand2", padx=8, pady=4
        ).pack(fill="x")
        tk.Button(
            btn_col, text="Send",
            command=lambda: self._on_send(vision=False),
            bg=self.BG2, fg=self.FG,
            font=("Segoe UI", 8), relief="flat",
            cursor="hand2", padx=8, pady=2
        ).pack(fill="x", pady=(3, 0))

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
            prefix = {"user": "You:    ", "agent4": "VLM:    ",
                      "mission": "Mission:", "system": "──────  ",
                      "err": "Error:  "}.get(tag, "        ")
            self._history.insert("end", f"{prefix} {text}\n", tag)
            self._history.config(state="disabled")
            self._history.see("end")
        self._win.after(0, _do)

    def _set_status(self, text: str, color: str | None = None):
        def _do():
            self._status_lbl.config(text=text, fg=color or "#555555")
        self._win.after(0, _do)

    # ── Screen grab ───────────────────────────────────────────────────────────
    def _grab_screen(self) -> Image.Image:
        return _grab_full_or_region(self._vision_region)

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

        messages = [{"role": "system", "content": AGENT4_SYSTEM_PROMPT}]
        for turn in self._conversation[-12:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model":       self.plugin.cfg["vlm_model"],
            "messages":    messages,
            "max_tokens":  int(self.plugin.cfg["vlm_max_tokens"]),
            "temperature": float(self.plugin.cfg["vlm_temperature"]),
        }
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
            raise RuntimeError(
                f"Vision server timed out after "
                f"{self.plugin.cfg['vlm_timeout']}s. The model may be loading "
                f"or the prompt may be too long. Increase vlm_timeout in "
                f"config.json if this persists."
            ) from e
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

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

        img = None
        if vision:
            try:
                img = self._grab_screen()
            except Exception as e:
                self._append_history("err", f"screen grab failed: {e}")

        tag = "mission" if source_agent else "user"
        label = f"[from {source_agent}] " if source_agent else ""
        self._append_history(tag, f"{label}{prompt}" + (" 📷" if img else ""))
        self._conversation.append({
            "role": "user",
            "content": prompt + (" [screenshot attached]" if img else ""),
        })

        import time as _t
        t0 = _t.time()
        try:
            response = self._call_vlm(prompt, img)
        except Exception as e:
            self._append_history("err", str(e))
            self._set_status("● error", self.RED)
            self.plugin.logger.log(
                "agent4", prompt, "", img, "error",
                outcome="fail",
                inference_ms=(_t.time() - t0) * 1000.0,
                extra={"exception": str(e)})
            self._busy = False
            return
        inference_ms = (_t.time() - t0) * 1000.0

        self._last_response = response
        self._conversation.append({"role": "assistant", "content": response})
        self._append_history("agent4", response)
        self._set_status("● idle", "#555555")
        self._busy = False

        action = "mission" if source_agent else "chat"
        self.plugin.logger.log(
            "agent4", prompt, response, img, action,
            inference_ms=inference_ms)

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
    def receive_mission(self, mission: str, source_agent: str):
        """Called by SOC routing when a message is addressed 'To Agent4'."""
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
        if self._last_response:
            pyperclip.copy(self._last_response)
            self._append_history("system", "last response copied to clipboard")

    def _clear_history(self):
        self._history.config(state="normal")
        self._history.delete("1.0", "end")
        self._history.config(state="disabled")
        self._conversation.clear()
        self._last_response = ""
        self._append_history("system", "history cleared")

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
    version = "0.1.0"

    def __init__(self, socu_app, config: dict):
        self.app = socu_app
        # Merge defaults with provided config
        self.cfg = dict(DEFAULTS)
        for k, v in (config or {}).items():
            if k in DEFAULTS and v is not None:
                self.cfg[k] = v
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

    def route_to_agent4(self, body: str, source_agent: str | None = None) -> bool:
        """Called by SOCU's _route_text when destination digit == '4'."""
        try:
            self.agent4_window.receive_mission(body, source_agent or "unknown")
            return True
        except Exception as e:
            try:
                self.app._log(f"[v_plugin] route_to_agent4 error: {e}")
            except Exception:
                pass
            return False

    def toggle_window(self):
        self.agent4_window.toggle()


def load(socu_app, config: dict | None = None) -> VPlugin:
    """Plugin entry point. Returns a VPlugin instance bound to `socu_app`."""
    return VPlugin(socu_app, config or {})
