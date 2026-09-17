import os
import yfinance as yf
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
import pandas as pd
import pandas_ta as ta
import numpy as np
import time

# =============================================================================
# 1. PORTFOLIO CONFIGURATION & TICKERS
# =============================================================================
my_costs = {
    "SNDK": 1622.50, 
    "ORCL": 0, 
    "PBR": 21.14, 
    "NVDA": 228.08, 
    "AVGO": 0, 
    "GLW": 166.02, 
    "CCJ": 0, 
    "GOOG": 0, 
    "LRCX": 0, 
    "VRT": 0, 
    "GEV": 0, 
    "CEG": 0, 
    "DELL": 524.25, 
    "TSM": 421.41, 
    "ZS": 0, 
    "VST": 146.08    
}

tickers = list(my_costs.keys())

# ==============================================================================
# 2. MACRO TECH SECTOR REGIME (QQQ)
# ==============================================================================
print("Fetching Macro Tech Sector Regime Context (QQQ)...")
start_date = datetime.now(timezone.utc) - timedelta(days=100)
qqq_hist = yf.Ticker("QQQ").history(start=start_date, auto_adjust=True, actions=True)

ema20 = qqq_hist.ta.ema(length=20).iloc[-1]
ema50 = qqq_hist.ta.ema(length=50).iloc[-1]

adx_data = qqq_hist.ta.adx(length=14)
adx_col = [c for c in adx_data.columns if c.startswith('ADX')][0]
current_adx = adx_data[adx_col].iloc[-1]
qqq_latest_close = qqq_hist['Close'].iloc[-1]

if current_adx > 25:
    if qqq_latest_close > ema50:
        tech_market_regime = "BULLISH"
    else:
        tech_market_regime = "BEARISH"
else:
    tech_market_regime = "NEUTRAL"

print(f"QQQ Regime: {tech_market_regime} (ADX: {current_adx:.2f}, Close: {qqq_latest_close:.2f}, EMA50: {ema50:.2f})")

# ==============================================================================
# 3. BULK MARKET DATA DOWNLOAD
# ==============================================================================
print("Executing bulk historical data download via Yahoo Finance...")
try:
    all_hist = yf.download(tickers, period="1y", auto_adjust=True, group_by='ticker')
except Exception as e:
    print(f"Bulk download failed ({e}), falling back to per-ticker fetch.")
    all_hist = None

# ==============================================================================
# 4. QUANTITATIVE ANALYSIS ENGINE (Pure Python, Zero LLM)
# ==============================================================================
analysis_data = []
calculated_market_data = {}

for ticker in tickers:
    try:
        if all_hist is not None and not all_hist.empty and ticker in all_hist.columns.get_level_values(0):
            hist = all_hist[ticker].copy()
        else:
            hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
            
        if hist.empty or len(hist) < 200:
            continue
            
        hist = hist.dropna(subset=['Close'])
        latest_close = hist['Close'].iloc[-1]
        
        # --- ATR & VOLATILITY STOP LOSS (Proven 3.0x in Backtest) ---
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        atr_pct = (atr / latest_close) * 100
        atr_stop_loss = latest_close - (3.0 * atr)
        
        # --- OBV (14-Day & 5-Day Volume Momentum) ---
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        
        obv_ema14 = obv.ewm(span=14, adjust=False).mean()
        obv_trend = "Rising" if (not pd.isna(latest_obv) and latest_obv > obv_ema14.iloc[-1]) else "Falling"
        
        obv_ema5 = obv.ewm(span=5, adjust=False).mean()
        obv_5d_trend = "Rising" if (not pd.isna(latest_obv) and latest_obv > obv_ema5.iloc[-1]) else "Falling"
        
        # --- MACD (12, 26, 9) ---
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            macd_status = "Bullish Crossover" if macd_line.iloc[-2] <= signal_line.iloc[-2] else "Bullish Territory"
        else:
            macd_status = "Bearish Crossover" if macd_line.iloc[-2] >= signal_line.iloc[-2] else "Bearish Territory"
            
        # Cost validation
        cost_val = my_costs.get(ticker, 0.0)
        is_owned = bool(cost_val and float(cost_val) > 0)
        actual_cost_str = f"{float(cost_val):.2f}" if is_owned else "N/A"
        is_profitable = (latest_close >= float(cost_val)) if is_owned else False

        # --- BOLLINGER BANDS & SQUEEZE LOOKBACK ---
        bbands = hist.ta.bbands(length=20, std=2)
        lower_col = [c for c in bbands.columns if c.startswith('BBL')][0]
        upper_col = [c for c in bbands.columns if c.startswith('BBU')][0]
        width_col = [c for c in bbands.columns if c.startswith('BBB')][0]
        
        bb_lower = float(bbands[lower_col].iloc[-1])
        bb_upper = float(bbands[upper_col].iloc[-1])
        bb_bandwidth = float(bbands[width_col].iloc[-1])
        is_squeezed = bb_bandwidth < 5.0
        
        # Dynamic lookback window
        profile_window = 50 if is_squeezed else 21
        hist_window = hist.tail(profile_window).copy()
        
        # Volume Profile Point of Control (POC)
        min_price = hist_window['Low'].min()
        max_price = hist_window['High'].max()
        bins = np.linspace(min_price, max_price, 51)
        bin_volume = np.zeros(len(bins) - 1)
        
        for _, row in hist_window.iterrows():
            d_low, d_high, d_vol = row['Low'], row['High'], row['Volume']
            slices = np.linspace(d_low, d_high, 15) if d_high != d_low else np.array([d_low])
            vol_per_slice = d_vol / len(slices)
            for price_point in slices:
                idx = max(0, min(np.digitize(price_point, bins) - 1, len(bin_volume) - 1))
                bin_volume[idx] += vol_per_slice

        poc_idx = np.argmax(bin_volume)
        poc_midpoint = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)

        swing_low_dyn = float(hist['Low'].tail(profile_window).min())
        swing_high_dyn = float(hist['High'].tail(profile_window).max())
        ema200 = float(hist.ta.ema(length=200).iloc[-1])
        
        # Support / Resistance candidates
        support_candidates = [
            poc_midpoint if poc_midpoint < latest_close else np.nan,
            swing_low_dyn if swing_low_dyn < latest_close else np.nan,
            bb_lower if bb_lower < latest_close else np.nan,
            latest_close - (2.0 * atr),
            ema200 if ema200 < latest_close else np.nan
        ]
        valid_supports = [s for s in support_candidates if np.isfinite(s) and s < latest_close]
        support_level = max(valid_supports) if valid_supports else (latest_close - (2.0 * atr))

        resistance_candidates = [
            poc_midpoint if poc_midpoint > latest_close else np.nan,
            swing_high_dyn if swing_high_dyn > latest_close else np.nan,
            bb_upper if bb_upper > latest_close else np.nan,
            latest_close + (2.0 * atr),
            ema200 if ema200 > latest_close else np.nan
        ]
        valid_resistances = [r for r in resistance_candidates if np.isfinite(r) and r > latest_close]
        resistance_level = min(valid_resistances) if valid_resistances else (latest_close + (2.0 * atr))

        # Channel guardrail
        if (resistance_level - support_level) < (1.0 * atr):
            support_level = latest_close - (2.0 * atr)
            resistance_level = latest_close + (2.0 * atr)

        # --- 3.0:1 MATHEMATICAL PROFIT TARGET (Directly from Backtest Engine) ---
        entry_ref = float(cost_val) if is_owned else latest_close
        initial_risk = max(entry_ref - support_level, 1.5 * atr)
        target_3r = entry_ref + (3.0 * initial_risk)
        
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close
        support_buffer = support_level * 0.020
        
        # Bounce confirmation: price is at support AND closed green or has rising OBV
        yesterday_close = hist['Close'].iloc[-2]
        is_at_support = risk_distance <= support_buffer
        is_bounce_confirmed = is_at_support and (obv_trend == "Rising" or obv_5d_trend == "Rising" or latest_close >= yesterday_close)

        # Trend Determination
        if latest_close > ema200 and "Bullish" in macd_status:
            trend = "Bullish"
        elif latest_close < ema200 and "Bearish" in macd_status:
            trend = "Bearish"
        else:
            trend = "Sideways"

        # --- DETERMINISTIC ACTIONABLE DECISION TREE ---
        prefix = "[Squeeze Active] " if is_squeezed else ""
        
        # 1. Hard Volatility Stop or Technical Breakdown
        if latest_close < atr_stop_loss or latest_close < support_level:
            rec = "Sell (Cut Loss)" if is_owned else "Sell"
            note = f"{prefix}Breached stop {atr_stop_loss:.2f} / support {support_level:.2f}. Cut loss priority."
            trend = "Bearish"
            
        # 2. 3.0:1 Mathematical Target Achieved
        elif latest_close >= target_3r:
            rec = "Sell (Take-Profit)"
            note = f"{prefix}3:1 Target {target_3r:.2f} reached! Lock in asymmetric gains."
            trend = "Bullish"
            
        # 3. Bearish Divergence in a Profitable Position
        elif is_owned and is_profitable and obv_trend == "Falling" and "Bearish Crossover" in macd_status:
            rec = "Sell (Take-Profit)"
            note = f"{prefix}Bearish divergence (falling OBV & MACD cross). Take profit to lock in gains."
            trend = "Bearish"
            
        # 4. Testing Intermediate Resistance Ceiling
        elif latest_close >= (resistance_level * 0.985):
            if latest_close >= resistance_level and (obv_trend == "Rising" or obv_5d_trend == "Rising"):
                rec = "Buy" if not is_owned else "Hold (Accumulate)"
                note = f"{prefix}Clean breakout above {resistance_level:.2f}. Target expansion toward {target_3r:.2f}."
                trend = "Bullish"
            else:
                rec = "Hold"
                note = f"{prefix}Testing resistance {resistance_level:.2f}. Hold for 3:1 target {target_3r:.2f}; watch breakout above {resistance_level + 0.01:.2f}."
                trend = "Bullish"
                
        # 5. Testing Support Floor (Bounce Potential)
        elif is_at_support:
            if is_owned:
                if is_bounce_confirmed:
                    rec = "Hold (Accumulate)"
                    note = f"{prefix}Support bounce confirmed at {support_level:.2f}. Safe to add shares with stop {atr_stop_loss:.2f}."
                else:
                    rec = "Hold"
                    note = f"{prefix}Holding support floor at {support_level:.2f}. Awaiting green bounce candle before adding; stop {atr_stop_loss:.2f}."
            else:
                if is_bounce_confirmed:
                    rec = "Buy"
                    note = f"{prefix}Testing support floor at {support_level:.2f}. Optimal bounce entry with stop {atr_stop_loss:.2f}."
                else:
                    rec = "Hold"
                    note = f"{prefix}At support floor {support_level:.2f}. Await green bounce candle to enter; stop {atr_stop_loss:.2f}."
            trend = "Bullish" if is_bounce_confirmed else "Sideways"
            
        # 6. Favorable Forward R:R Expansion toward Resistance
        elif (reward_distance / max(risk_distance, 0.01)) >= 3.0 and obv_trend == "Rising":
            rec = "Hold (Accumulate)" if is_owned else "Buy"
            note = f"{prefix}Healthy 3:1 R:R expansion toward resistance {resistance_level:.2f} (Target: {target_3r:.2f})."
            trend = "Bullish"
            
        # 7. Standard Consolidation / Holding
        else:
            rec = "Hold"
            note = f"{prefix}Consolidating between support {support_level:.2f} and resistance {resistance_level:.2f}."

        calculated_market_data[ticker] = {
            "latest_price": f"{latest_close:.2f}",
            "support": f"{support_level:.2f}",
            "resistance": f"{resistance_level:.2f}",
            "target_3r": f"{target_3r:.2f}",
            "atr_stop": f"{atr_stop_loss:.2f}",
            "cost_display": actual_cost_str
        }

        analysis_data.append({
            "stock_name": ticker,
            "obv_status": obv_trend,
            "macd_status": macd_status,
            "trend": trend,
            "recommendation": rec,
            "important_note": note
        })

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

# ==============================================================================
# 5. COMPILE REPORT INTO CORPORATE PDF
# ==============================================================================
class CorporatePDF(FPDF):
    def __init__(self, adx, ema20, ema50, close):
        super().__init__()
        self.adx = adx
        self.ema20 = ema20
        self.ema50 = ema50
        self.close = close

    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 4, "F")
        self.ln(4)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, "Daily US Market Report (Quantitative Engine v2)", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(79, 70, 229)
        self.cell(0, 5, "WATCHLIST TECHNICAL SUMMARY", new_x="LMARGIN", new_y="NEXT", align="L")
        
        thailand_tz = timezone(timedelta(hours=7))
        thai_timestamp = datetime.now(thailand_tz).strftime('%Y-%m-%d %H:%M:%S')
        
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        debug_str = f"ADX: {self.adx:.2f} | EMA20: {self.ema20:.2f} | EMA50: {self.ema50:.2f} | Close: {self.close:.2f}"
        self.cell(0, 5, f"Generated on {thai_timestamp} (TH Time) | Context: {tech_market_regime} | {debug_str}", new_x="LMARGIN", new_y="NEXT", align="L")
        
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y() + 4, 200, self.get_y() + 4)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(241, 245, 249)
        self.line(10, self.get_y(), 200, self.get_y())
        
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(100, 10, f"Page {self.page_no()}", align="L")
        self.cell(90, 10, "Pure Quantitative Strategy Engine | Zero AI Latency | Cost: $0.00", align="R")

pdf = CorporatePDF(current_adx, ema20, ema50, qqq_latest_close)
pdf.add_page()

# Table formatting: Sums to 190 mm
column_widths = (12, 13, 13, 13, 13, 13, 12, 16, 13, 15, 57)

with pdf.table(col_widths=column_widths, text_align="LEFT", line_height=4.5, padding=2, outer_border_width=0.5) as table:
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support", "Resist", "ATR Stop", "OBV14", "MACD", "Trend21", "Rec.", "Important Note"]
    for header_title in headers:
        header_row.cell(header_title)
        
    for idx, stock in enumerate(analysis_data):
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        
        market_metrics = calculated_market_data.get(ticker, {
            "latest_price": "N/A", "support": "N/A", "resistance": "N/A", "atr_stop": "N/A", "cost_display": "N/A"
        })
        
        pdf.set_fill_color(255, 255, 255) if idx % 2 == 0 else pdf.set_fill_color(248, 250, 252)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(51, 65, 85)
        
        row.cell(ticker)
        row.cell(market_metrics["cost_display"])
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
        row.cell(market_metrics["atr_stop"])
        row.cell(str(stock.get("obv_status", "")))
        row.cell(str(stock.get("macd_status", "")))
        
        # Color coding for Trend
        if "bullish" in trend_status:
            pdf.set_text_color(21, 128, 61)
        elif "bearish" in trend_status:
            pdf.set_text_color(185, 28, 28)
        else:
            pdf.set_text_color(51, 65, 85)
        row.cell(str(stock.get("trend", "")))
        
        # Color coding for Recommendation
        if "buy" in rec_status:
            pdf.set_text_color(21, 128, 61)
        elif "sell" in rec_status:
            pdf.set_text_color(185, 28, 28)
        elif "accumulate" in rec_status:
            pdf.set_text_color(16, 185, 129)
        else:
            pdf.set_text_color(180, 83, 9)
        row.cell(str(stock.get("recommendation", "")))
        
        pdf.set_text_color(51, 65, 85)
        row.cell(str(stock.get("important_note", "")))

# --- PDF Output with Safe Fallback ---
output_filename = "us_market_analysis_v2.pdf"
try:
    pdf.output(output_filename)
    print(f"PDF output finalized successfully as '{output_filename}'.")
except PermissionError:
    timestamp_suffix = datetime.now().strftime('%H%M%S')
    fallback_filename = f"us_market_analysis_{timestamp_suffix}.pdf"
    pdf.output(fallback_filename)
    print(f"Notice: '{output_filename}' is currently open in your viewer. Saved output to '{fallback_filename}'.")
