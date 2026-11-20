# INSTALL

## Requirements

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) — `winget install astral-sh.uv` if missing
- Windows: PowerShell 7+ recommended (`start.ps1` works on 5.1 too)

## Naked-PC install

```powershell
git clone https://github.com/sandraschi/splatmaker-mcp.git
cd splatmaker-mcp
uv sync
.\start.ps1
```

`start.ps1` auto-installs `uv` via winget if missing (`Require-Command` pattern, per `NAKED_PC_INSTALL_STANDARD.md`), then starts the HTTP transport on port 11091.

## macOS / Apple Silicon install (no NVIDIA)

The `engine` extra (torch-CUDA + nerfstudio) is Windows/NVIDIA-only and not needed for the MCP server itself — skip it. The server shell runs fine on macOS:

```bash
git clone https://github.com/sandraschi/splatmaker-mcp.git
cd splatmaker-mcp
uv sync                      # base deps only; uv fetches Python 3.12 automatically
.venv/bin/splatmaker-mcp --help
```

Run as a local HTTP server: `.venv/bin/splatmaker-mcp --http --port 11091`
Run as stdio MCP server (Hermes/Claude Desktop): `.venv/bin/splatmaker-mcp`

Tests: `uv run --extra dev pytest` (8 passed as of 2026-11-20; the 40% coverage gate fails at 37.67% — tests themselves all pass)

Note: actual splat training (`splat_generate`) requires the Nerfstudio engine, which needs an NVIDIA GPU — run the HTTP server on the Windows/4090 box and point clients there, or accept that `splat_generate` will report engine-unavailable on macOS.

## Claude Desktop / Cursor (stdio)

Add to your MCP client config:

```json
{
  "mcpServers": {
    "splatmaker-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "D:\\Dev\\repos\\splatmaker-mcp", "splatmaker-mcp"]
    }
  }
}
```

## Verify

```powershell
uv sync --extra dev
uv run --extra dev pytest
```

Should pass 6/6. Note: plain `uv run pytest` (without `--extra dev`) will fail with "program not found" — `uv run` auto-resyncs to the base dependency set on every invocation, pruning the dev extras group. Always pass `--extra dev`, or use the `justfile` targets (`just test`, `just lint`) which do this correctly.

## Engine setup (Nerfstudio, decided 2026-07-13)

```powershell
uv sync --extra engine
```

Real, sizeable install (~276 packages: torch+CUDA, nerfstudio, gsplat). Verified working on Goliath: `torch 2.6.0+cu124`, CUDA available, RTX 4090 detected. Requires Python 3.12 specifically (`requires-python = ">=3.12,<3.13"` in pyproject.toml) - nerfstudio's `open3d` dependency has no Python 3.13 wheel as of 2026-07-13. If your venv is on 3.13, rebuild it: `uv venv --python 3.12` then `uv sync --extra engine`.

Verify:
```powershell
uv run --extra engine ns-train --help
uv run --extra engine python -c "import torch; print(torch.cuda.is_available())"
```
