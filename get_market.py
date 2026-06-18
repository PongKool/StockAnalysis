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

# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION
tickers = ["MU", "NVDA", "SNDK", "NOW", "MSFT", "TSM", "VST", "LRCX", "PLTR", "WMT", "AVGO", "ANET", "TPR", "IONQ"]
my_costs = {
    "MU": 1020.16, "NVDA": 217.03, "SNDK": 2094.39, "NOW": 107.68, "MSFT": 459.63, 
    "TSM": 424.30, "VST": 153.65, "LRCX": 319.36, "PLTR": 134.49, 
    "WMT": 120.47, "AVGO": 451.22, "ANET": 169.94, "TPR": 150.20, "IONQ": 64.31
}
calculated_market_data = {}
client = genai.Client()

class StockAnalysisSchema(BaseModel):
    stock_name: str = Field(description="Ticker symbol.")
    cost: str = Field(description="Entry cost or N/A.")
    obv_status: str = Field(description="Rising/Falling/Flat")
    macd_status: str = Field(description="MACD crossover/territory state.")
    trend: str = Field(description="Bullish/Bearish/Sideways")
    recommendation: str = Field(description="Buy/Hold/Hold (Accumulate)/Sell")
    important_note: str = Field(description="Technical reason mapping volatility or breakout levels.")

class StockAnalysisList(BaseModel):
    analyses: list[StockAnalysisSchema]

print("Fetching Macro Tech Sector Regime Context (QQQ)...")
try:
    qqq_hist = yf.Ticker("QQQ").history(period="1mo", auto_adjust=True)
    qqq_ema20 = qqq_hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    qqq_latest = qqq_hist['Close'].iloc[-1]
    tech_market_regime = "BULLISH" if qqq_latest > qqq_ema20 else "BEARISH"
except Exception:
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
        obv_mean_change = obv.diff().tail(14).mean()
        obv_trend = "Flat" if pd.isna(obv_mean_change) or obv_mean_change == 0 else ("Rising" if obv_mean_change > 0 else "Falling")
            
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
        actual_cost = f"{cost_val:.2f}" if cost_val > 0 else "N/A"
        is_profitable = "Yes" if (cost_val > 0 and latest_close > cost_val) else "No"
        
        # --- SUPPORT/RESISTANCE TO 21-DAY WINDOW ---
        hist_1m = hist.tail(21)
        support_level = hist_1m['Low'].min()
        resistance_level = hist_1m['High'].max()

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
            f"OBV: {obv_trend} |MACD: {macd_status} |Closes:[{trend_string}]\n"
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
     * Only issue a neutral **"Hold"** if the target is within the 5x threshold but you are awaiting a definitive breakout on an otherwise stable/sideways asset.  

- **CRITICAL RISK FILTER:** Except during an active breakout scenario, DO NOT recommend a "Buy" or "Hold (Accumulate)" if the Risk/Reward status (RR:) is labeled as "Poor" or "Breakdown". 
  * **BREAKDOWN RULE:** If RR: is labeled as "Breakdown", you must force a "Sell" or "Sell (Cut Loss)" to preserve capital. 
  * **BREAKOUT RULE:** If the asset's price has broken out above resistance (resulting in an RR: of 'Breakout'), you may override general boundaries and issue a "Buy" or "Hold (Accumulate)" if OBV, MACD, and immediate trend milestones confirm strong upward velocity.
  * **BUY THE BOUNCE RULE:** If RR: is labeled as "Testing Support (Bounce Potential)" AND the MACD Status is NOT a "Bearish Crossover", you are authorized to issue a "Buy" or "Hold (Accumulate)". This represents an institutional entry floor with an optimal risk-to-reward profile.   

OUTPUT INSTRUCTION FOR THE 'IMPORTANT_NOTE' FIELD:
You MUST explicitly mention how technical profiles or volatility metrics justified your decision.
- If the Latest Close (L:) is within 1.5% of the Resistance level (R:), calculate the breakout target (Resistance + 0.01) and explicitly state it in the note (e.g., "Watch for a clean breakout above $XXXX.XX").
- If the recommendation is "Sell", check the profitability flag (P:). If P is "Yes", explicitly label your reason as a "Take-Profit" action. If P is "No" (or cost is N/A), explicitly label your reason as a "Cut-Loss" or risk mitigation action.
- If the stock was downgraded due to demanding too many 'ATRs to Target' (Days: > 5.0), explicitly note that the upside target requires too many days of average volatility.
- If the stock has successfully broken above its resistance floor, note that old resistance has turned into support. Keep it concise enough to fit a small table cell.

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
            response_schema=StockAnalysisList,
            temperature=0.15
        )
    )
    analysis_data = json.loads(response.text.strip())["analyses"]
except Exception as e:
    print(f"API Error: {e}. Utilizing fallback strategy.")
    analysis_data = [{
        "stock_name": t,
        "cost": f"{my_costs.get(t, 0.0):.2f}" if my_costs.get(t, 0.0) > 0 else "N/A",
        "obv_status": "Error",
        "macd_status": "Error",
        "trend": "Error",
        "recommendation": "Error",
        "important_note": "System extraction failure."
    } for t in tickers]

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
        self.cell(0, 5, f"Generated on {thai_timestamp} (TH Time) | Sector Context: {tech_market_regime}", new_x="LMARGIN", new_y="NEXT", align="L")
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

# Sums up to exactly 190 mm (Fits perfectly inside standard A4 margins)
column_widths = (12, 12, 12, 12, 12, 14, 11, 15, 12, 11, 67)
with pdf.table(col_widths=column_widths, text_align="LEFT", line_height=4.5, padding=2, outer_border_width=0.5) as table:
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
        
        pdf.set_fill_color(255, 255, 255) if idx % 2 == 0 else pdf.set_fill_color(248, 250, 252)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(51, 65, 85)
        
        row.cell(ticker)
        row.cell(str(stock.get("cost", "")))
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
        
filename = "morning_market_analysis.pdf"
pdf.output(filename)
print(f"PDF output finalized successfully as {filename}.")
