# V Plugin — Vision module for SOC Ultralight

Adds a 4th agent slot ("Agent 4") to [SOC Ultralight](https://github.com/BaxtersLab/SOC_Ultralight), powered by any local vision-capable GGUF model served by [GGUF Chatbox](https://github.com/BaxtersLab/GGUF-Chatbox)'s vision server on `127.0.0.1:8082`.

Kept as a separate plugin/repository so SOC Ultralight stays lightweight — users without a vision-capable GPU don't pay the dependency cost.

## What it does

- Floating Agent 4 chat window inside SOCU
- Region selector — drag a rectangle on screen, send region + prompt to the VLM
- Auto-routing — when the VLM replies in SOCU's `To AgentN ... end message now` format, the message is routed via SOCU's normal routing pipeline (including the self-modification gate)
- Session logging — JSONL per session + saved image captures under `data_log/`
- Model-agnostic — works with any vision GGUF (Qwen2-VL, LLaVA, MiniCPM-V, etc.) loaded by GGUF Chatbox's vision server

## Requirements

- [SOC Ultralight](https://github.com/BaxtersLab/SOC_Ultralight) installed
- [GGUF Chatbox](https://github.com/BaxtersLab/GGUF-Chatbox) installed and running
- A vision-capable GGUF model + matching `mmproj` projector file
- ~6-12 GB VRAM depending on model (CPU works but is slow)
- Python: `requests`, `Pillow`, optional `mss` (all already present if SOCU is installed)

## Installation

### Recommended — via SOCU installer

```powershell
cd SOC_Ultralight
python install.py
# Answer "y" when asked "Install V plugin?"
# Installer clones this repo into SOC_Ultralight/plugins/v_plugin/
# and verifies GGUF Chatbox is set up.
```

### Manual

```powershell
cd SOC_Ultralight/plugins
git clone https://github.com/BaxtersLab/V_plugin.git v_plugin
```

SOCU's `_load_plugins()` discovers the plugin automatically at startup. Restart SOCU — the `👁 A4` button appears in the control row when the plugin loads.

## Plugin contract

`v_plugin.load(socu_app, config) -> VPlugin`

`VPlugin` exposes:

| attribute | purpose |
|---|---|
| `.cfg` | dict of VLM settings (URL, model, timeout, max_tokens, temperature) |
| `.logger` | `DataLogger` writing `data_log/session_*.jsonl` + images |
| `.agent4_window` | floating Tkinter chat window |
| `.route_to_agent4(body, source_agent)` | called by SOCU when a `To Agent4` block is detected |
| `.toggle_window()` | show/hide the A4 window |

## Configuration

Read from SOCU's `config.json` (all optional — defaults shown):

```json
{
  "vlm_server_url":  "http://localhost:8082/v1/chat/completions",
  "vlm_model":       "vision",
  "vlm_timeout":     30.0,
  "vlm_max_tokens":  1024,
  "vlm_temperature": 0.3
}
```

`vlm_model` is sent in the request payload but llama-server (under GGUF Chatbox) serves whatever model the user loaded — the field is informational only. The literal string `"vision"` matches GGUF Chatbox's own convention.

## Self-modification gate

Every routed message — including Agent 4 replies — passes through SOCU's `SelfModGate`. The plugin never bypasses this. Agent 4 cannot trigger edits to protected SOC files (e.g. `soc_ultralight.py`, `v_plugin.py`, `config.json`, `pc.py`) without explicit user approval via the modal popup.

## Origin

Extracted from `SOCQ`, a Qwen2-VL-specific fork of SOCU. With this plugin live, SOCQ becomes a deprecated/shelved snapshot — V supersedes it and is model-agnostic.

## License

MIT — see [LICENSE](./LICENSE).
