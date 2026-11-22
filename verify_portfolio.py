"""Check Parlay Guard portfolio through ASGI (actual URL), not app.routes.

Only if /portfolio genuinely returns 404, install an idempotent end-of-file
router registration with a timestamped backup. No game data, models, or
paper trades are modified. No background refresh is started by this probe.
"""
from __future__ import annotations

import asyncio
import importlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = ROOT / "src" / "api.py"
MARKER = "# PG_PORTFOLIO_ROUTE_REPAIR_V2"
URLS = ["/portfolio", "/portfolio-core.js", "/portfolio-ui.js"]


async def get_status(app, path: str) -> int | None:
    """Dispatch an in-process HTTP GET without starting FastAPI lifespan tasks."""
    status = None

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        nonlocal status
        if message.get("type") == "http.response.start":
            status = int(message["status"])

    scope = {
        "type": "http", "http_version": "1.1", "method": "GET",
        "scheme": "http", "root_path": "", "path": path,
        "raw_path": path.encode("ascii"), "query_string": b"",
        "headers": [], "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return status


def probe(app) -> dict[str, int | None]:
    return {path: asyncio.run(get_status(app, path)) for path in URLS}


def main() -> int:
    if not API.is_file():
        print("Missing src/api.py: run from your existing Parlay Guard folder.")
        return 1
    module = importlib.import_module("src.api")
    first = probe(module.app)
    print("Initial portfolio HTTP check:", first)
    if all(first.get(path) == 200 for path in URLS):
        print("Portfolio route and JavaScript files are available. Startup check PASS.")
        return 0

    if first["/portfolio"] != 404:
        print("Portfolio returned a non-404 failure. Check the static files and application error above.")
        return 1

    print("/portfolio returned HTTP 404. Repairing the router registration...")
    source = API.read_text(encoding="utf-8")
    if MARKER in source:
        print("The repair marker already exists but /portfolio still returns 404. No changes made.")
        return 1
    if not (ROOT / "src" / "portfolio_api.py").is_file():
        print("Missing src/portfolio_api.py. Re-extract the original portfolio ZIP before fixing startup.")
        return 1

    repair = (
        "\n\n" + MARKER + "\n"
        "# Append registration to the final app, after any earlier app/router setup.\n"
        "from .portfolio_api import router as _pg_portfolio_router_repair_v2\n"
        "app.include_router(_pg_portfolio_router_repair_v2)\n"
    )
    import ast
    new_source = source + repair
    ast.parse(new_source)

    backup_dir = ROOT / "backups" / (
        "portfolio_route_repair_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    )
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(API, backup_dir / "api.py")
    temp = API.with_name("api.py.portfolio_fix.tmp")
    try:
        temp.write_text(new_source, encoding="utf-8")
        temp.replace(API)
        # Reload to test the same final app the Uvicorn entrypoint will expose.
        module = importlib.reload(module)
        second = probe(module.app)
        print("After repair HTTP check:", second)
        if not all(second.get(path) == 200 for path in URLS):
            raise RuntimeError("Portfolio still did not return HTTP 200 on all three paths")
    except Exception as exc:
        shutil.copy2(backup_dir / "api.py", API)
        print("The repair did not pass checks; original src/api.py restored.")
        print(f"Details: {type(exc).__name__}: {exc}")
        return 1
    finally:
        temp.unlink(missing_ok=True)

    print("Portfolio startup check PASS. Original api.py backup:", backup_dir / "api.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
