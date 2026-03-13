# models/schemas.py
# Pydantic models for all API request/response validation.
#
# WHY THIS MATTERS:
# Without these, a missing field or wrong type in the request body
# causes a cryptic 500 error. With Pydantic, FastAPI returns a clean
# 422 Unprocessable Entity with the exact field that failed.
#
# HOW TO USE IN api/ FILES:
#   from models.schemas import ChatRequest, ChatResponse
#   @router.post("/api/chat")
#   async def chat_endpoint(body: ChatRequest) -> ChatResponse:
#       ...  # body.session_id and body.message are already validated

from pydantic import BaseModel, Field
from typing import Optional

# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="UUID identifying the conversation session")
    message:    str = Field(..., min_length=1, description="User message, cannot be empty")

class ChatResponse(BaseModel):
    response:   str
    session_id: str

class ClearRequest(BaseModel):
    session_id: str

class ClearResponse(BaseModel):
    status: str  # "cleared"

# ── Market ────────────────────────────────────────────────────────────────────

class CompanyMeta(BaseModel):
    ticker:  str
    name_en: str
    name_ar: str
    market:  str
    flag:    str
    sector:  str

class DashboardResponse(BaseModel):
    ticker:     str
    name_en:    str
    name_ar:    Optional[str] = None
    market:     Optional[str] = None
    flag:       Optional[str] = None
    sector:     Optional[str] = None
    price:      float
    currency:   str
    change:     Optional[float] = None
    change_pct: Optional[float] = None
    market_cap: Optional[int]   = None
    pe_ratio:   Optional[float] = None
    pb_ratio:   Optional[float] = None
    eps:        Optional[float] = None
    dividend:   Optional[float] = None
    week52_high:Optional[float] = None
    week52_low: Optional[float] = None
    volume:     Optional[int]   = None
    news:       list            = []

# ── Signal ────────────────────────────────────────────────────────────────────

class SignalResponse(BaseModel):
    ticker:     str
    signal:     str   # "bullish" | "neutral" | "bearish" | "unavailable"
    confidence: float = 0.0
    rsi:        Optional[float] = None
    sma_cross:  Optional[bool]  = None
    rf_signal:  Optional[str]   = None
    disclaimer: str  = ""
    error:      str  = ""
