import os
import json
import yfinance as yf
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
import pandas as pd

# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION
tickers = ["MU", "NVDA", "ORCL", "SNDK", "MSFT", "TSM", "VST", "LRCX", "NOW", "PLTR", "WMT", "AVGO", "ANET", "TPR", "IONQ"]
my_costs = {
    "MU": 999.89, "NVDA": 217.03, "ORCL": 0, "SNDK": 2113.20, 
    "MSFT": 459.63, "TSM": 424.30, "VST": 153.65, "LRCX": 319.36, 
    "NOW": 107.68, "PLTR": 134.49, "WMT": 120.34, "AVGO": 451.22, 
    "ANET": 169.94, "TPR": 150.20, "IONQ": 64.31
}
calculated_market_data = {}
client = genai.Client()

class StockAnalysisSchema(BaseModel):
    stock_name: str = Field(description="The ticker symbol of the stock.")
    cost: str = Field(description="The exact entry cost value provided in data input.")
    obv_status: str = Field(description="OBV trend direction, e.g., Rising / Falling / Flat")
    macd_status: str = Field(description="Current MACD state, e.g., Bullish Territory / Bearish Crossover")
    trend: str = Field(description="Overall direction: Bullish, Bearish, or Sideways")
    recommendation: str = Field(description="Actionable decision: Buy, Hold, Hold (Accumulate), or Sell")
    important_note: str = Field(description="Technical explanation factoring in momentum depletion, breakout confirmation, or if targets are mathematically too far away based on ATR days.")

print("Fetching Macro Tech Sector Regime Context (QQQ)...")
try:
    qqq_hist = yf.Ticker("QQQ").history(period="1mo", auto_adjust=True)
    qqq_ema20 = qqq_hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    qqq_latest = qqq_hist['Close'].iloc[-1]
    tech_market_regime = "BULLISH (QQQ above 20EMA)" if qqq_latest > qqq_ema20 else "BEARISH/CAUTIOUS (QQQ below 20EMA)"
except Exception:
    tech_market_regime = "NEUTRAL (Data Unavailable)"

print(f"Current Tech Sector Context: {tech_market_regime}")

print("Executing bulk historical data download via Yahoo Finance...")
try:
    all_hist = yf.download(tickers, period="6mo", auto_adjust=True, group_by='ticker')
except Exception as e:
    print(f"Bulk download failed: {e}")
    all_hist = None

data_summary = f"GLOBAL TECH SECTOR REGIME: {tech_market_regime}\n"

# 2. DATA PROCESSING LOOP
for ticker in tickers:
    try:
        if all_hist is not None and ticker in all_hist.columns.levels[0]:
            hist = all_hist[ticker].copy()
        else:
            hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
            
        if hist.empty or len(hist) < 26:
            continue
            
        hist = hist.dropna(subset=['Close'])
        latest_close = hist['Close'].iloc[-1]
        
        # --- TECH PORT OPTIMIZATION: CALCULATE 14-DAY ATR (VOLATILITY SCALE) ---
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        atr_pct = (atr / latest_close) * 100
        
        # --- NO. 1 IMPLEMENTATION: VOLATILITY-BASED TRAILING STOP LOSS ---
        atr_multiplier = 2.5
        atr_stop_loss = latest_close - (atr_multiplier * atr)
        
        # --- CALCULATE OBV ---
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        obv_mean_change = obv.diff().tail(14).mean()
        if pd.isna(obv_mean_change) or obv_mean_change == 0:
            obv_trend = "Flat"
        else:
            obv_trend = "Rising" if obv_mean_change > 0 else "Falling"
            
        # --- CALCULATE MACD ---
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        
        if latest_macd > latest_signal:
            macd_status = "Bullish Crossover" if prev_macd <= prev_signal else "Bullish Territory"
        else:
            macd_status = "Bearish Crossover" if prev_macd >= prev_signal else "Bearish Territory"
            
        cost_val = my_costs.get(ticker, 0.0)
        actual_cost = f"{cost_val:.2f}" if cost_val > 0 else "N/A"
        is_profitable = "Yes" if (cost_val > 0 and latest_close > cost_val) else "No"
        
        # --- LOCKED SUPPORT/RESISTANCE TO 21-DAY WINDOW ---
        hist_1m = hist.tail(21)
        support_level = hist_1m['Low'].min()
        resistance_level = hist_1m['High'].max()
        
        # --- CALCULATE RISK/REWARD RATIO ---
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close
        if reward_distance <= 0:
            rr_ratio_str = "Poor (At Resistance)"
        elif risk_distance <= 0:
            rr_ratio_str = "Excellent (At Support)"
        else:
            rr_ratio_str = f"1:{(reward_distance / risk_distance):.2f}"
            
        # --- NO. 3 IMPLEMENTATION: ATR UNITS TO TARGET (PROBABILITY FILTER) ---
        if reward_distance > 0:
            atr_to_target = reward_distance / atr
        else:
            atr_to_target = 0.0

        recent_closes = hist_1m.tail(14)
        trend_string = ", ".join([f"{row['Close']:.2f}" for _, row in recent_closes.iterrows()])
        
        # Track data metrics including our new ATR Stop for PDF reference
        calculated_market_data[ticker] = {
            "latest_price": f"{latest_close:.2f}",
            "support": f"{support_level:.2f}",
            "resistance": f"{resistance_level:.2f}",
            "atr_stop": f"{atr_stop_loss:.2f}"
        }
        
        # Append updated volatility intelligence fields to the text summary block
        data_summary += (
            f"Ticker: {ticker} | Entry Cost: {actual_cost} | Latest Close: {latest_close:.2f} | "
            f"Is Position Profitable?: {is_profitable} | 1Mo Support: {support_level:.2f} | 1Mo Resistance: {resistance_level:.2f} | "
            f"Daily ATR Volatility: {atr:.2f} ({atr_pct:.1f}%) | Volatility Stop Loss (2.5x ATR): {atr_stop_loss:.2f} | "
            f"Risk/Reward: {rr_ratio_str} | ATRs Required to Hit Target Resistance: {atr_to_target:.1f} days | "
            f"OBV: {latest_obv:.0f} ({obv_trend}) | MACD: {latest_macd:.2f} (Signal: {latest_signal:.2f}, {macd_status}) | "
            f"Recent Close Trend: [{trend_string}]\n"
        )
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")

# 3. REQUEST STRUCTURED ANALYSIS FROM GEMINI
prompt = f"""
You are an expert institutional technical analyst managing a high-beta technology and semiconductor portfolio. You are given the 'GLOBAL TECH SECTOR REGIME' context derived from the Nasdaq-100 (QQQ). Factor this heavily into your systemic risk decisions.

CRITICAL PORTFOLIO RISK & EXIT RULES:
1. **Bearish Divergence Rule:** Pay deep attention to instances where price action is stable or rising, but the OBV Trend is "Falling". This indicates institutional distribution/selling behind the scenes. If a position is profitable and showing an OBV divergence, flag it immediately as a Take-Profit exit.
2. **Volatility Stop Filter (NEW):** If the asset's current price breaks below its calculated 'Volatility Stop Loss (2.5x ATR)', you must immediately flag an exit priority. Override lagging indicators and force a Cautious/Sell recommendation to protect trading principal from volatility contraction.
3. **Trailing Take-Profit Exits:** If a position is profitable ("Yes"), prioritize capital protection:
   - Downgrade recommendation to **Sell** immediately if the "MACD Status" is a "Bearish Crossover" OR the OBV trend is "Falling" (signals institutional distribution).
   - If the price trend and OBV trend are both confidently **"Rising"**, you may issue a **"Hold"** or **"Hold (Accumulate)"** to capture extended upside, even if the MACD line is temporarily lagging or sitting in Bearish Territory.
   - If the GLOBAL TECH SECTOR REGIME is BEARISH, tighten requirements; exit if momentum begins to flatten even if a full crossover hasn't completed.
4. **Increasing Positions / Probability Filtering (NEW):**
   - Issue a **"Buy"** or a **"Hold (Accumulate)"** recommendation if the stock demonstrates strong potential to continue upward.
   - Strong potential is defined as having a **"Rising" OBV trend**, an overall **"Bullish" trend**, AND a healthy MACD profile.
   - **PROBABILITY FILTER (ATR Distance):** Check 'ATRs Required to Hit Target Resistance'. If a stock requires more than 4.5 days of average ATR movement to reach its resistance ceiling, the target is statistically unrealistic for short-term trades. Downgrade its recommendation to a neutral **"Hold"** because the target takes too much time/effort to achieve.
   - **CRITICAL RISK FILTER:** Even if parameters are strong, DO NOT recommend a "Buy" or "Hold (Accumulate)" if the Risk/Reward status is labeled as **"Poor (At Resistance)"**. Downgrade to **"Hold"** to await a definitive breakout.

OUTPUT INSTRUCTION FOR THE 'IMPORTANT_NOTE' FIELD:
You MUST explicitly mention how technical profiles or volatility metrics justified your decision. 
- If a stock is close to resistance (like SNDK), calculate the breakout target ($1-month Resistance + 0.01$) and explicitly state it in the note (e.g., "Watch for a clean breakout above $XXXX.XX").
- If a stock was downgraded due to demanding too many 'ATRs to Target', note that the upside target requires too many days of average volatility. 
- If the stock has successfully broken above its resistance floor, note that old resistance has turned into support. 
Keep it concise enough to fit the table cell.

Stocks to analyze: {', '.join(tickers)}
Data Input: {data_summary}
"""

print("Generating structured technical analysis via Gemini API...")
try:
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[StockAnalysisSchema],
            temperature=0.15
        )
    )
    analysis_data = json.loads(response.text.strip())
except Exception as e:
    print(f"API Error or Schema Parsing exception: {e}. Utilizing fallback strategy.")
    analysis_data = [
        {
            "stock_name": t, 
            "cost": f"{my_costs.get(t, 0.0):.2f}" if my_costs.get(t, 0.0) > 0 else "N/A", 
            "obv_status": "Error", "macd_status": "Error", "trend": "Error", 
            "recommendation": "Error", "important_note": "System tracking failure. Could not structure data payload safely."
        } for t in tickers
    ]

# 4. COMPILE REPORT INTO PDF TABLE LAYOUT
class CorporatePDF(FPDF):
    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 4, "F")
        self.ln(4)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, "Daily Market Report", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(79, 70, 229)
        self.cell(0, 5, "WATCHLIST TECHNICAL SUMMARY", new_x="LMARGIN", new_y="NEXT", align="L")
        
        thailand_tz = timezone(timedelta(hours=7))
        thai_timestamp = datetime.now(thailand_tz).strftime('%Y-%m-%d %H:%M:%S')
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f"Generated automatically on {thai_timestamp} (Thailand Time) | Regime: {tech_market_regime}", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y() + 4, 200, self.get_y() + 4)
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_draw_color(241, 245, 249)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

pdf = CorporatePDF()
pdf.add_page()

# Shifted column layout spacing to create a dedicated visual column for the Volatility Stop
column_widths = (13, 11, 11, 11, 11, 14, 11, 21, 15, 11, 61)

with pdf.table(col_widths=column_widths, text_align="LEFT", line_height=6, padding=1.5, outer_border_width=0.5) as table:
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support", "Resist.", "ATR Stop", "OBV", "MACD", "Trend", "Rec.", "Important Note"]
    for header_title in headers:
        header_row.cell(header_title)
        
    for idx, stock in enumerate(analysis_data):
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        
        market_metrics = calculated_market_data.get(ticker, {"latest_price": "N/A", "support": "N/A", "resistance": "N/A", "atr_stop": "N/A"})
        
        if idx % 2 == 0:
            pdf.set_fill_color(255, 255, 255)
        else:
            pdf.set_fill_color(248, 250, 252)
            
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(51, 65, 85)
        
        row.cell(ticker)
        row.cell(str(stock.get("cost", "")))
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
        
        # Display the custom dynamic Volatility Trailing Stop Loss level
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
        
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(71, 85, 105)
        row.cell(str(stock.get("important_note", "")))

filename = "morning_market_analysis.pdf"
pdf.output(filename)
print(f"PDF output finalized successfully as {filename}.")
