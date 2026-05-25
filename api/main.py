from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os


from api.chat import router as chat_router
from api.market import router as market_router
from api.signal import router as signal_router
from api.backtest_api import router as backtest_router

app = FastAPI(title="X-Invest API")


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


app.include_router(chat_router)
app.include_router(market_router)
app.include_router(signal_router)
app.include_router(backtest_router)


if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================================
# SERVE PAGES
# ==========================================
templates_path = Path(__file__).parent.parent / "templates"

@app.get("/")
async def root():
    index_file = templates_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "X-Invest API is running"}

@app.get("/chat")
async def chat_page():
    return FileResponse(templates_path / "chat.html")

@app.get("/market")
async def market_page():
    return FileResponse(templates_path / "market.html")

@app.get("/backtest")
async def backtest_page():
    return FileResponse(templates_path / "backtest.html")