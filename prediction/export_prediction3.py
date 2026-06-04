#!/usr/bin/env python3
"""
export_prediction.py — X-INVEST (Dynamic Current Price & Full Universe)
✅ Current_Price يتغير يومياً بناءً على مسار التوقعات (Simulation Path)
✅ يحل مشكلة ثبات السعر ويوفر خط زمني متصل للداشبورد
✅ معالجة خاصة لسهم BRK-B (استخدام BRK.B عند الفشل)
✅ حذف الأعمدة غير المطلوبة (Generated_At, P50_Price)
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# ── إعداد المسارات لضمان الاستيراد الصحيح ─────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
PRED_DIR   = os.path.join(ROOT_DIR, "prediction")

for p in [ROOT_DIR, PRED_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from prediction.predict import predict_signal
except ImportError:
    print("❌ تأكد من وجود predict.py في مجلد prediction/")
    sys.exit(1)

# ── قائمة الأسهم المستهدفة ─────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "BRK-B", "V", "MA",
    "UNH", "JNJ", "MRK", "ABBV",
    "PG", "PEP", "KO",
    "HD", "CVX"
]

OUTPUT_FILE = "monthly_predictions_CLEAN.csv"

def process_ticker(ticker_symbol, display_name, all_rows):
    """معالجة سهم واحد وإضافة بياناته للقائمة مع السعر الديناميكي"""
    try:
        res = predict_signal(ticker_symbol)
        if res.get("error"):
            raise ValueError(res["error"])
            
        base_price = res["current_price"]
        fc_series = res.get("forecast_series", [])
        
        if not fc_series:
            raise ValueError("No forecast data")

        # المتغير الذي سيقوم بتحديث السعر الحالي لكل صف (Simulation Path)
        # يبدأ بالسعر الحقيقي، ثم يأخذ سعر الهدف من اليوم السابق
        running_current_price = base_price 

        for day in fc_series:
            date_str = day.get("date")
            if not date_str: continue
            
            # 1. الحصول على السعر المستهدف (P50) من المودل
            p50 = float(day.get("p50", day.get("price", base_price)))
            
            # 2. تعيين "السعر الحالي" لهذا الصف
            # في اليوم الأول هو السعر الحقيقي، في الأيام التالية هو سعر الهدف السابق
            current_price_for_row = running_current_price
            
            # 3. تحديث المتغير لليوم التالي (الهدف يصبح سعر الغد الحالي)
            running_current_price = p50 
            
            # 4. تحديد الاتجاه بناءً على السعر الحالي الجديد vs الهدف
            direction = "UP" if p50 > current_price_for_row else "DOWN"
            
            # 5. حساب الثقة والإشارة (نفس المنطق السابق)
            try:
                days_ahead = (pd.to_datetime(date_str) - pd.Timestamp.today()).days
            except:
                days_ahead = 0
            
            # اضمحلال الثقة 0.5% يومياً
            dyn_conf = max(0.40, res["confidence"] - (max(0, days_ahead) * 0.005))
            
            if dyn_conf >= 0.48:
                signal = "BUY" if direction == "UP" else "SELL"
            else:
                signal = "HOLD"

            # 6. إضافة الصف
            all_rows.append({
                "Ticker": display_name, # استخدام الاسم الأصلي (مثل BRK-B)
                "Forecast_Date": date_str,
                "Signal_Today": signal,
                "Direction": direction,
                "Confidence_%": round(dyn_conf * 100, 2),
                "Current_Price": round(current_price_for_row, 2), # ✅ السعر الديناميكي
                "Price_Target": round(p50, 2),
                "Expected_Return_%": round(res["expected_return"], 4),
                "P90_Upper": round(day.get("p90", day.get("upper", p50)), 2),
                "P10_Lower": round(day.get("p10", day.get("lower", p50)), 2)
            })
        return True
        
    except Exception as e:
        print(f"❌ {str(e)[:40]}")
        return False

def export_full_universe():
    print(f"🚀 جاري التصدير لـ {len(TICKERS)} سهم...")
    all_rows = []
    failed = []

    for ticker in TICKERS:
        print(f"📈 {ticker}...", end=" ")
        
        # محاولة المعالجة بالاسم الأصلي
        success = process_ticker(ticker, ticker, all_rows)
        
        # ❗ معالجة خاصة لسهم BRK-B إذا فشل الاسم الأصلي
        if not success and ticker == "BRK-B":
            print(f"\n  🔄 محاولة بديلة: BRK.B ...", end=" ")
            success = process_ticker("BRK.B", "BRK-B", all_rows) # نستخدم BRK-B كاسم للعرض
        
        if success:
            print("✅")
        else:
            print("❌")
            failed.append(ticker)

    if not all_rows:
        print("\n⚠️ لم يتم توليد أي بيانات.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["Ticker", "Forecast_Date"]).reset_index(drop=True)

    # ترتيب الأعمدة النهائية
    cols = [
        "Ticker", "Forecast_Date", "Signal_Today", "Direction", "Confidence_%",
        "Current_Price", "Price_Target", "Expected_Return_%",
        "P90_Upper", "P10_Lower"
    ]
    df = df[[c for c in cols if c in df.columns]]
    
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ تم التصدير بنجاح: {OUTPUT_FILE}")
    print(f"📊 الأسهم الناجحة: {len(df['Ticker'].unique())}/{len(TICKERS)}")
    
    if failed:
        print(f"\n⚠️ الأسهم التي فشلت: {failed}")

if __name__ == "__main__":
    export_full_universe()