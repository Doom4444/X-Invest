import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from prediction.backtest import run_backtest

st.set_page_config(page_title="X-Invest | Backtest Report", layout="wide")

st.title("📊 تقرير اختبار الأداء (Backtest Report)")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ إعدادات المحاكاة")
ticker = st.sidebar.text_input("🔍 رمز السهم (Ticker)", value="AAPL").upper().strip()
initial_capital = st.sidebar.number_input("💰 رأس المال الابتدائي ($)", value=10000.0, step=1000.0)
start_date = st.sidebar.date_input("📅 تاريخ البداية", value=datetime(2022, 1, 1))

run_btn = st.sidebar.button("🚀 تشغيل المحاكاة", type="primary", use_container_width=True)

# Main Area
if run_btn:
    if not ticker:
        st.error("⚠️ من فضلك أدخل رمز السهم!")
    else:
        with st.spinner(f"⏳ جاري تحليل أداء {ticker} ..."):
            try:
                results = run_backtest(
                    ticker=ticker, 
                    initial_capital=float(initial_capital), 
                    start=start_date.strftime("%Y-%m-%d")
                )
                
                equity_df = results["equity"]
                trades = results["trades"]
                metrics = results["metrics"]

                # Metrics
                total_return = ((equity_df.iloc[-1] - initial_capital) / initial_capital) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💵 رأس المال النهائي", f"${equity_df.iloc[-1]:,.2f}", delta=f"{total_return:.2f}%")
                col2.metric("🎯 نسبة النجاح (Win Rate)", f"{metrics['win_rate']*100:.1f}%")
                col3.metric("📉 أقصى تراجع (Max DD)", f"{metrics['max_dd']*100:.1f}%")
                col4.metric("⚖️ عامل الربح (PF)", f"{metrics['profit_factor']:.2f}")

                st.divider()

                # Chart
                st.subheader("📈 منحنى نمو رأس المال (Equity Curve)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df.index, 
                    y=equity_df.values, 
                    mode='lines',
                    name='Portfolio Value',
                    fill='tozeroy',
                    fillcolor='rgba(34, 197, 94, 0.2)',
                    line=dict(color='#22c55e', width=2)
                ))
                st.plotly_chart(fig, use_container_width=True)

                # Trades Table
                st.subheader("📋 سجل الصفقات")
                if trades:
                    trades_df = pd.DataFrame(trades)
                    st.dataframe(trades_df, use_container_width=True, height=300)
                else:
                    st.info("⚠️ لم يتم تنفيذ أي صفقات خلال هذه الفترة.")

            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء التشغيل: {str(e)}")