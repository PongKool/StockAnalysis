import os
import json
import yfinance as yf
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from fpdf import FPDF
from fpdf.fonts import FontFace
from datetime import datetime, timezone, timedelta
import pandas as pd
import pandas_ta as ta
import numpy as np
import time


# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION
my_costs = {
    "SNDK": 1658.69, 
    "ORCL": 144.39, 
    "AMZN": 0, 
    "NVDA": 216.49, 
    "AVGO": 0, 
    "SHOP": 152.98, 
    "DIS": 108.31, 
    "GOOG": 0, 
    "LRCX": 331.22,
    "VRT": 0,
    "MSFT": 491.01, 
    "CEG": 281.74,
    "NFLX": 81.17,
    "TSM": 425.85,
    "UBER": 78.78, 
    "CRDO": 0    
}

tickers = list(my_costs.keys())

calculated_market_data = {}
client = genai.Client()

class StockAnalysisSchema(BaseModel):
    stock_name: str = Field(description="Ticker symbol.")
    obv_status: str = Field(description="Rising/Falling/Flat")
    macd_status: str = Field(description="MACD crossover/territory state.")
    trend: str = Field(description="Bullish/Bearish/Sideways")
    recommendation: str = Field(description="Buy/Hold/Hold (Accumulate)/Sell")
    important_note: str = Field(description="Technical reason mapping volatility or breakout levels.")

class StockAnalysisList(BaseModel):
    analyses: list[StockAnalysisSchema]

print("Fetching Macro Tech Sector Regime Context (QQQ)...")

# Fetch 100 days of data to account for indicator "warm-up". 
# ADX and EMA use recursive smoothing and require a long history to calculate accurate, stable values for the final row.

start_date = datetime.now(timezone.utc) - timedelta(days=100)
qqq_hist = yf.Ticker("QQQ").history(start=start_date, auto_adjust=True, actions=True)

# 2. Calculate Indicators
# EMA 20
ema20 = qqq_hist.ta.ema(length=20).iloc[-1]
# ADX 14
adx_data = qqq_hist.ta.adx(length=14)
adx_col = [c for c in adx_data.columns if c.startswith('ADX')][0]
current_adx = adx_data[adx_col].iloc[-1]

# 3. Logic: Trend strength (ADX) + Direction (Price vs EMA 20)
qqq_latest_close = qqq_hist['Close'].iloc[-1]

# --- ADD DEBUG PRINTS HERE ---
print(f"--- DEBUG DATA ---")
print(f"Calculated ADX: {current_adx}")
print(f"Calculated EMA20: {ema20}")
print(f"Latest Close: {qqq_latest_close}")
print(f"Condition (ADX > 25): {current_adx > 25}")
print(f"Condition (Close > EMA20): {qqq_latest_close > ema20}")
# -----------------------------

if current_adx > 25:
    if qqq_latest_close > ema20:
        tech_market_regime = "BULLISH"
    else:
        tech_market_regime = "BEARISH"
else:
    tech_market_regime = "NEUTRAL"

print("Executing bulk historical data download via Yahoo Finance...")
try:
    all_hist = yf.download(tickers, period="6mo", auto_adjust=True, group_by='ticker')
except Exception as e:
    print(f"Bulk download failed: {e}")
    all_hist = None

data_summary = f"REGIME: {tech_market_regime}\n"

# 2. DATA PROCESSING LOOP
for ticker in tickers:
    try:
        if all_hist is not None and not all_hist.empty and ticker in all_hist.columns.get_level_values(0):
            hist = all_hist[ticker].copy()
        else:
            hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
            
        if hist.empty or len(hist) < 26:
            continue
            
        hist = hist.dropna(subset=['Close'])
        latest_close = hist['Close'].iloc[-1]
        
        # --- ATR & TRAILING STOP LOSS ---
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        atr_pct = (atr / latest_close) * 100
        atr_stop_loss = latest_close - (2.5 * atr)
        
        # --- CALCULATE OBV ---
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
    
        # REVISED: Compare OBV against its Moving Average (mimicking TradingView)
        # 14-day macro trend
        obv_ema14 = obv.ewm(span=14, adjust=False).mean()
        if pd.isna(latest_obv) or pd.isna(obv_ema14.iloc[-1]):
            obv_trend = "Flat"
        else:
            obv_trend = "Rising" if latest_obv > obv_ema14.iloc[-1] else "Falling"
    
        # REVISED: 5-day short-term trend calculation
        obv_ema5 = obv.ewm(span=5, adjust=False).mean()
        if pd.isna(latest_obv) or pd.isna(obv_ema5.iloc[-1]):
            obv_5d_trend = "Flat"
        else:
            obv_5d_trend = "Rising" if latest_obv > obv_ema5.iloc[-1] else "Falling"
        
            
        # --- CALCULATE MACD ---
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            macd_status = "Bullish Crossover" if macd_line.iloc[-2] <= signal_line.iloc[-2] else "Bullish Territory"
        else:
            macd_status = "Bearish Crossover" if macd_line.iloc[-2] >= signal_line.iloc[-2] else "Bearish Territory"
            
        cost_val = my_costs.get(ticker, 0.0)

        # Fallback to latest price if cost is 0, None, or empty
        if not cost_val or cost_val == 0 or str(cost_val).strip() == "":
            cost_val = latest_close

        actual_cost = f"{cost_val:.2f}"
        is_profitable = "Yes" if latest_close >= cost_val else "No"
        
        # --- FAST INSTITUTIONAL SUPPORT & RESISTANCE (SMA 20 + 1-Month Volume Profile) ---
        # 1. Safely calculate fast 20-Day SMA for US Tech stocks
        if len(hist) >= 20:
            sma_trend = float(hist['Close'].rolling(window=20).mean().iloc[-1])
        else:
            sma_trend = float(hist['Close'].mean())
        
        # 2. Calculate Volume Profile over a tight 1-month window (21 days)
        hist_1m = hist.tail(21).copy()
        
        # Safely group by price bins without triggering Pandas version errors
        price_bins = pd.cut(hist_1m['Close'], bins=10)
        volume_by_bin = hist_1m.groupby(price_bins)['Volume'].sum()
        
        poc_bin = volume_by_bin.idxmax()
        poc_midpoint = float(poc_bin.mid)
        
        # Define near-term swing levels (10 days) first
        swing_low_10d = float(hist['Low'].tail(10).min())
        swing_high_10d = float(hist['High'].tail(10).max())
    
        # Calculate Average True Range (ATR) for volatility context
        tr = np.maximum(
            hist['High'] - hist['Low'],
            np.maximum(
                abs(hist['High'] - hist['Close'].shift(1)),
                abs(hist['Low'] - hist['Close'].shift(1))
            )
        )
        atr_14 = float(tr.rolling(14).mean().iloc[-1])
    
        # Hybrid Support & Resistance with ATR and Volume Profile
        at_least_support = latest_close - (1.5 * atr_14)
        structural_support = max(swing_low_10d, poc_midpoint if poc_midpoint < latest_close else 0)
        support_level = float(max(structural_support, at_least_support))
        if support_level >= latest_close:
            support_level = float(swing_low_10d)
    
        at_least_resistance = latest_close + (1.5 * atr_14)
        structural_resistance = min(swing_high_10d, poc_midpoint if poc_midpoint > latest_close else float('inf'))
        resistance_level = float(min(structural_resistance, at_least_resistance))
        if resistance_level <= latest_close:
            resistance_level = float(swing_high_10d)
        
        # HIGH-BETA MILESTONE OPTIMIZATION (14-day history windowed into key nodes)
        # Moved up here so closes_14d is defined before it's used in the conditional block below
        closes_14d = hist_1m['Close'].tail(14).tolist()
        optimized_trend = [
            closes_14d[0],   # T-14 (Macro baseline)
            closes_14d[4],   # T-10
            closes_14d[9],   # T-5  (One week ago)
            closes_14d[11],  # T-3  (Recent momentum switch)
            closes_14d[-2],  # Yesterday
            closes_14d[-1]   # Today
        ]
        trend_string = ", ".join([f"{val:.1f}" for val in optimized_trend])

        # --- RISK/REWARD RATIO WITH BREAKOUT/BOUNCE DETECTION ---
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close

        # Define a tight buffer zone (e.g., within 1.5% of the support level)
        support_buffer = support_level * 0.015

        if latest_close < support_level:
            rr_ratio_str = "Breakdown"
        elif latest_close > resistance_level:
            rr_ratio_str = "Breakout"
        # NEW: Detect if price is hugging support but starting to bounce/hold
        elif risk_distance <= support_buffer and (obv_trend == "Rising" or latest_close >= closes_14d[-2]):
            rr_ratio_str = "Testing Support (Bounce Potential)"
        elif reward_distance <= 0:
            rr_ratio_str = "Poor"
        elif risk_distance == 0:
            rr_ratio_str = "Excellent"
        else:
            rr_ratio_str = f"1:{(reward_distance / risk_distance):.1f}"

        atr_to_target = (reward_distance / atr) if reward_distance > 0 else 0.0

        # --- SAVE & SUMMARIZE DATA ---
        calculated_market_data[ticker] = {
            "latest_price": f"{latest_close:.2f}",
            "support": f"{support_level:.2f}",
            "resistance": f"{resistance_level:.2f}",
            "atr_stop": f"{atr_stop_loss:.2f}"
        }
        
        data_summary += (
            f"T: {ticker} |C: {actual_cost} |L: {latest_close:.2f} |P: {is_profitable} |"
            f"S: {support_level:.2f} |R: {resistance_level:.2f} |ATR: {atr:.1f} ({atr_pct:.1f}%)|"
            f"Stop: {atr_stop_loss:.2f} |RR: {rr_ratio_str} |Days: {atr_to_target:.1f} |"
            f"OBV: {obv_trend} |OBV5D: {obv_5d_trend} |MACD: {macd_status} |Closes:[{trend_string}]\n"
        )
        
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")
        

# 3. REQUEST STRUCTURED ANALYSIS FROM GEMINI
prompt = f"""

You are an expert institutional technical analyst managing a high-beta technology and semiconductor portfolio. You are given the 'GLOBAL TECH SECTOR REGIME' context derived from the Nasdaq-100 (QQQ). Factor this heavily into your systemic risk decisions.

CRITICAL PORTFOLIO RISK & EXIT RULES:
1. **Bearish Divergence Rule:** Pay deep attention to instances where price action is stable or rising, but the OBV Trend is "Falling". This indicates institutional distribution/selling behind the scenes. If a position is profitable and showing an OBV divergence, flag it immediately as a Take-Profit exit.
2. **Volatility Stop Filter:** If the asset's current price breaks below its calculated 'Volatility Stop Loss' (Stop:), you must immediately flag an exit priority. Override lagging indicators and force a Cautious/Sell recommendation to protect trading principal from volatility contraction.
3. **Trailing Take-Profit Exits:** If a position is profitable ("Yes"), prioritize capital protection:
   - Downgrade recommendation to **Sell** immediately if the "MACD Status" is a "Bearish Crossover" OR the OBV trend is "Falling" (signals institutional distribution).
   - EXCEPTION: If the immediate price trend and OBV trend are both confidently **"Rising"**, you may issue a **"Hold"** or **"Hold (Accumulate)"** to capture extended upside, even if the absolute MACD line is temporarily lagging or sitting in Bearish Territory.
   - If the GLOBAL TECH SECTOR REGIME is BEARISH, tighten requirements; exit if momentum begins to flatten even if a full crossover hasn't completed.
4. **Position Sizing & Probability Filtering:**
   - Issue a **"Buy"** or a **"Hold (Accumulate)"** recommendation if the stock demonstrates strong potential to continue upward. Strong potential is defined as having a **"Rising" OBV trend**, an overall **"Bullish" trend**, AND a healthy MACD profile.
   - **PROBABILITY & RISK FILTER:** Compare the total percentage distance to target resistance against the stock's 'Daily ATR Volatility (%)'.
       * If the percentage distance to resistance requires more than 5x its normal Daily ATR Volatility (Days: > 5.0), the target is statistically unrealistic for a short-term horizon.
       * **REVERSAL CORRECTION:** If this 5x threshold is exceeded AND the stock's macro trend is already "Bearish" WITHOUT showing an immediate recovery signature (i.e., its OBV is "Falling" or its MACD is in "Bearish Territory"), do NOT issue a "Hold". Instead, force a **"Sell"** or **"Sell (Cut Loss)"** recommendation due to high opportunity cost.
       * Only issue a neutral **"Hold"** if the target is within the 5x threshold but you are awaiting a definitive breakout on an otherwise stable/sideways asset (provided its OBV is not actively "Falling" and MACD is not in "Bearish Territory").   
   - **CRITICAL RISK FILTER:** Except during an active breakout scenario, DO NOT recommend a "Buy" or "Hold (Accumulate)" if the Risk/Reward status (RR:) is labeled as "Poor" or "Breakdown". 
       * **BREAKDOWN RULE:** If RR: is labeled as "Breakdown", you must force a "Sell" or "Sell (Cut Loss)" to preserve capital. 
       * **BREAKOUT RULE:** If the asset's price has broken out above resistance (resulting in an RR: of 'Breakout'), you may override general boundaries and issue a "Buy" or "Hold (Accumulate)" if OBV, MACD, and immediate trend milestones confirm strong upward velocity.
       * **BUY THE BOUNCE RULE:** If RR: is labeled as "Testing Support (Bounce Potential)" AND the MACD Status is NOT a "Bearish Crossover", you are authorized to issue a "Buy" or "Hold (Accumulate)". This represents an institutional entry floor with an optimal risk-to-reward profile.
   - **GLOBAL REGIME OVERRIDE:** If the GLOBAL TECH SECTOR REGIME is BULLISH, you are authorized to issue a neutral "Hold" (instead of an automatic "Sell") for any asset trading safely above its Volatility Stop Loss. Furthermore, if the raw data explicitly flags the short-term volume trend as **OBV5D: Rising** (confirming recent institutional accumulation at support), you are fully authorized to upgrade a "Sell" to a neutral **"Hold"** or **"Hold (Accumulate)"**, even if the absolute daily macro indicator (OBV:) is still flagged as Falling.   

OUTPUT INSTRUCTION FOR THE 'IMPORTANT_NOTE' FIELD:
You MUST explicitly mention how technical profiles or volatility metrics justified your decision.
- If the Latest Close (L:) is within 1.5% of the Resistance level (R:), calculate the breakout target (Resistance + 0.01) and explicitly state it in the note (e.g., "Watch for a clean breakout above $XXXX.XX").
- If the recommendation is "Sell", check the profitability flag (P:). If P is "Yes", explicitly label your reason as a "Take-Profit" action. If P is "No" (or cost is N/A), you MUST explicitly label your reason as a "Cut-Loss" action and forbid any mention of "Take-Profit".
- If the stock was downgraded due to demanding too many 'ATRs to Target' (Days: > 5.0), explicitly note that the upside target requires too many days of average volatility.
- If the stock has successfully broken above its resistance floor, note that old resistance has turned into support. 

CRITICAL FORMATTING:
- Keep the 'important_note' detailed yet dense (strictly under 45 words) to ensure deep technical justification fits within the table structure.
- Write in plain text only. Do NOT use markdown syntax (like **, *, or `) or raw quotation marks inside the text fields.

Stocks to analyze: {', '.join(tickers)}
Data Input: {data_summary}
"""

max_retries = 5
retry_delay = 5  # Initial delay in seconds
response = None

for attempt in range(max_retries):
    try:
        print(f"Generating structured technical analysis via Gemini API (Attempt {attempt + 1}/{max_retries})...")
        response = client.models.generate_content(
            model='gemini-3.6-flash',  # Ensure your model ID is up to date
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StockAnalysisList,
                temperature=0.15
            )
        )
        analysis_data = json.loads(response.text.strip())["analyses"]
        break
    except Exception as e:
        print(f"API Error on attempt {attempt + 1}: {e}")
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff multiplier
        else:
            print("Max retries reached. Utilizing fallback strategy.")
            analysis_data = [{
                "stock_name": t,
                "cost": f"{my_costs.get(t, 0.0):.2f}" if my_costs.get(t, 0.0) > 0 else "N/A",
                "obv_status": "Error",
                "macd_status": "Error",
                "trend": "Error",
                "recommendation": "Error",
                "important_note": "System extraction failure."
            } for t in tickers]

# --- 3. Calculate LLM Token Costs Safely ---
if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
else:
    input_tokens = 0
    output_tokens = 0

cost_usd = ((input_tokens * 0.75) / 1000000) + ((output_tokens * 3.75) / 1000000)

# --- Fetch Real-time Exchange Rate ---
try:
    thb_ticker = yf.Ticker("THB=X")
    usd_to_thb_rate = thb_ticker.fast_info['last_price']
except Exception:
    usd_to_thb_rate = 35.00

cost_thb = cost_usd * usd_to_thb_rate

# Format string to display in the PDF footer
token_cost_display = f"Tokens: In {input_tokens:,} / Out {output_tokens:,} | Cost: ${cost_usd:.6f} (~{cost_thb:.2f} THB)"


# 4. COMPILE REPORT INTO PDF TABLE LAYOUT
class CorporatePDF(FPDF):
    def __init__(self, adx, ema, close, token_cost_str):
        super().__init__()
        self.adx = adx
        self.ema = ema
        self.close = close
        self.token_cost_str = token_cost_str

    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 4, "F")
        self.ln(4)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, "Daily US Market Report", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(79, 70, 229)
        self.cell(0, 5, "WATCHLIST TECHNICAL SUMMARY", new_x="LMARGIN", new_y="NEXT", align="L")
        
        thailand_tz = timezone(timedelta(hours=7))
        thai_timestamp = datetime.now(thailand_tz).strftime('%Y-%m-%d %H:%M:%S')
        
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        debug_str = f"ADX: {self.adx:.2f} | EMA20: {self.ema:.2f} | Close: {self.close:.2f}"
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
        
        # Display Page Number on the left and Token Cost info on the right
        self.cell(100, 10, f"Page {self.page_no()}", align="L")
        self.cell(90, 10, self.token_cost_str, align="R")

pdf = CorporatePDF(current_adx, ema20, qqq_latest_close, token_cost_display)
pdf.add_page()


# Sums up to exactly 190 mm (Fits perfectly inside standard A4 margins)
column_widths = (12, 13, 13, 13, 13, 13, 12, 16, 13, 13, 59)
with pdf.table(col_widths=column_widths, text_align="LEFT", line_height=4.5, padding=2, outer_border_width=0.5) as table:
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support21", "Resist21", "ATR Stop14", "OBV14", "MACD", "Trend21", "Rec.", "Important Note"]
    for header_title in headers:
        header_row.cell(header_title)
        
    for idx, stock in enumerate(analysis_data):
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        
        market_metrics = calculated_market_data.get(ticker, {"latest_price": "N/A", "support": "N/A", "resistance": "N/A", "atr_stop": "N/A"})
        
        pdf.set_fill_color(255, 255, 255) if idx % 2 == 0 else pdf.set_fill_color(248, 250, 252)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(51, 65, 85)
        
        row.cell(ticker)
        current_cost = my_costs.get(ticker, 0.0)
        if not current_cost or current_cost == 0 or str(current_cost).strip() == "":
            cost_display = market_metrics.get("latest_price", "N/A")
        else:
            cost_display = f"{current_cost:.2f}"
        row.cell(cost_display)
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
        row.cell(market_metrics["atr_stop"])
        row.cell(str(stock.get("obv_status", "")))
        row.cell(str(stock.get("macd_status", "")))
        
        if "bullish" in trend_status:
            pdf.set_text_color(21, 128, 61)
        elif "bearish" in trend_status:
            pdf.set_text_color(185, 28, 28)
        else:
            pdf.set_text_color(51, 65, 85)
        row.cell(str(stock.get("trend", "")))
        
        if "buy" in rec_status:
            pdf.set_text_color(21, 128, 61)
        elif "sell" in rec_status:
            pdf.set_text_color(185, 28, 28)
        else:
            pdf.set_text_color(180, 83, 9)
        row.cell(str(stock.get("recommendation", "")))
        
        # Determine row background color to prevent bleeding state artifacts
        bg_color = (255, 255, 255) if idx % 2 == 0 else (248, 250, 252)

        # Pass a clean FontFace instance directly to the cell style parameter.
        row.cell(
            str(stock.get("important_note", "")),
            style=FontFace(size_pt=6.5, color=(71, 85, 105), fill_color=bg_color)
        )
        
filename = "us_market_analysis.pdf"
pdf.output(filename)
print(f"PDF output finalized successfully as {filename}.")
