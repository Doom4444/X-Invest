import sys
import os
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

# ==========================================
# 1. FIX IMPORT PATH
# ==========================================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

router = APIRouter()

# ==========================================
# 2. HELPER: CLEAN DATA FOR JSON
# ==========================================
def clean_data(obj):
    """تحويل أنواع pandas/numpy إلى أنواع JSON قياسية"""
    if isinstance(obj, (pd.Timestamp, pd.Period, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_data(item) for item in obj]
    return obj

# ==========================================
# 3. ENDPOINT
# ==========================================
@router.post("/api/backtest")
async def run_backtest_simulation(request: Request):
    try:
        body = await request.json()
        ticker = body.get("ticker", "AAPL").upper()
        capital = float(body.get("initial_capital", 10000.0))
        start = body.get("start", "2024-01-01")
        end = body.get("end", "2025-12-31")  # ✅ تم إضافة end date المطلوب

        print(f"🚀 Received request: {ticker} from {start} to {end}")

        # استيراد محرك الـ Backtest
        from prediction.backtest import BacktestEngine

        # 1️⃣ إنشاء الكائن
        eng = BacktestEngine(
            ticker=ticker,
            start=start,
            end=end,
            initial_capital=capital,
            use_model=True
        )

        # 2️⃣ تشغيل المحرك (يطبع النتائج في التيرمنال ويحفظ الشارت تلقائياً)
        metrics = eng.run()

        # 3️⃣ استخراج البيانات المطلوبة للواجهة الأمامية
        # daily_equity عبارة عن قائمة قواميس: [{"date":..., "equity":...}, ...]
        equity_data = [d["equity"] for d in eng.daily_equity]
        trades_data = eng.trades
        metrics_data = eng.metrics

        # 4️⃣ تنظيف البيانات من أنواع numpy/pandas غير المدعومة في JSON
        clean_equity = clean_data(equity_data)
        clean_trades = clean_data(trades_data)
        clean_metrics = clean_data(metrics_data)

        return JSONResponse({
            "success": True,
            "equity": clean_equity,
            "trades": clean_trades,
            "metrics": clean_metrics
        })

    except FileNotFoundError as e:
        print(f"❌ Model not found: {e}")
        return JSONResponse({
            "success": False,
            "error": "Model not found. Please run Train.py first."
        }, status_code=500)

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ Backtest Error: {str(e)}")
        print(error_detail)

        return JSONResponse({
            "success": False,
            "error": str(e),
            "details": error_detail
        }, status_code=500)