# main.py
# Entry point. Run with: uvicorn main:app --reload
# Connects all routers and serves HTML pages.
# No business logic lives here -- everything is in api/, pipeline/, market/

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="X-Invest", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Routers are imported lazily so the app can still start
# even if an optional dependency is missing.
logger = logging.getLogger("x-invest")
logging.basicConfig(level=logging.INFO)

# Base directory for static assets (project root when running via uvicorn)
STATIC_IMG = Path(__file__).resolve().parent / "static" / "img"
HOME_BG_NAMES = ["home-bg.png", "home-bg.jpeg", "home-bg.jpg"]


def _home_bg_context() -> dict:
    """Pass cache-busting version and filename for home background image."""
    for name in HOME_BG_NAMES:
        path = STATIC_IMG / name
        if path.is_file():
            return {
                "home_bg_file": name,
                "home_bg_version": int(path.stat().st_mtime),
            }
    return {"home_bg_file": "home-bg.png", "home_bg_version": 0}

def _include_routers() -> None:
    for module_name in ("api.chat", "api.market", "api.signal"):
        try:
            mod = __import__(module_name, fromlist=["router"])
            router = getattr(mod, "router", None)
            if router is None:
                raise AttributeError(f"{module_name} has no 'router'")
            app.include_router(router)
            logger.info("Included router: %s", module_name)
        except Exception as e:
            logger.exception("Failed to include router %s: %s", module_name, e)

_include_routers()


@app.get("/")
async def home(request: Request):
    ctx = {"request": request, **_home_bg_context()}
    return templates.TemplateResponse("index.html", ctx)

@app.get("/chat")
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/market")
async def market_page(request: Request):
    return templates.TemplateResponse("market.html", {"request": request})

@app.get("/debug/routes")
async def debug_routes():
    """Debug helper: list registered HTTP route paths."""
    paths = []
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = sorted(getattr(r, "methods", []) or [])
        if path and methods:
            paths.append({"path": path, "methods": methods})
    return {"routes": paths}
