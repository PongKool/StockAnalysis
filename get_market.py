import os
import json
import yfinance as yf
from google import genai
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
import pandas as pd

# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION FIRST
# Define your target tech and growth watchlist tickers (VST completely removed)
tickers = ["MU", "NVDA", "ORCL", "SNDK", "MSFT", "TSM", "LLY", "LRCX", "NOW", "AMD", "CACI", "AVGO", "ANET"]

# YOUR ACTUAL COST BASIS DICTIONARY (VST completely removed)
my_costs = {
    "MU": 424.62, "NVDA": 220.80, "ORCL": 183.72, "SNDK": 1418.17, 
    "MSFT": 455.37, "TSM": 424.30, "LLY": 971.12, "LRCX": 305.41, 
    "NOW": 107.68, "AMD": 448.37, "CACI": 524.53, "AVGO": 446.13, 
    "ANET": 171.11
}

# Initialize Gemini Client (Uses the API key stored in your environment)
client = genai.Client()

print("Fetching technical data from Yahoo Finance...")
data_summary = ""

# 2. DATA GATHERING LOOP (Safe now that 'tickers' is defined above)
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        # Fetching 3 months of price context to accurately calculate 26-period EMA for MACD
        hist = stock.history(period="3mo")
        
        if hist.empty or len(hist) < 26:
            continue
            
        # --- THE FIX: Clean out any incomplete live/placeholder rows containing NaN ---
        hist = hist.dropna(subset=['Close'])
        
        # --- CALCULATE OBV ---
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        obv_trend = "Rising" if obv.tail(5).diff().mean() > 0 else "Falling"
        
        # --- CALCULATE MACD (State + Trajectory Logic) ---
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        
        # Check previous day to determine if a fresh crossover occurred
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        
        if latest_macd > latest_signal:
            if prev_macd <= prev_signal:
                macd_status = "Bullish Crossover"
            else:
                macd_status = "Bullish Territory"
        else:
            if prev_macd >= prev_signal:
                macd_status = "Bearish Crossover"
            else:
                macd_status = "Bearish Territory"
                
        # Format actual cost to 2 decimal places to save tokens
        cost_val = my_costs.get(ticker, 0.0)
        actual_cost = f"{cost_val:.2f}" if cost_val > 0 else "N/A"
        
        # Calculate true Support and Resistance using the trailing 1 month (approx last 21 trading days)
        hist_1m = hist.tail(21)
        support_level = hist_1m['Low'].min()
        resistance_level = hist_1m['High'].max()
        latest_close = hist_1m['Close'].iloc[-1]
        
        # --- CALCULATE RISK/REWARD RATIO ---
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close
        
        # Handle edge case where price is exactly at or above resistance to prevent division by zero
        if reward_distance <= 0:
            rr_ratio_str = "Poor (At Resistance)"
        elif risk_distance <= 0:
            rr_ratio_str = "Excellent (At Support)"
        else:
            calculated_ratio = reward_distance / risk_distance
            rr_ratio_str = f"1:{calculated_ratio:.2f}"
            
        # Compress the recent 14 days of closing prices into a minimal string layout
        recent_closes = hist_1m.tail(14)
        trend_string = ", ".join([f"{row['Close']:.2f}" for _, row in recent_closes.iterrows()])
        
        # Token-optimized data line representing the stock status
        data_summary += (
            f"Ticker: {ticker} | Entry Cost: {actual_cost} | Latest Close: {latest_close:.2f} | "
            f"1Mo Support: {support_level:.2f} | 1Mo Resistance: {resistance_level:.2f} | "
            f"Risk/Reward: {rr_ratio_str} | "
            f"OBV: {latest_obv:.0f} ({obv_trend}) | MACD: {latest_macd:.2f} (Signal: {latest_signal:.2f}, {macd_status}) | "
            f"Recent Close Trend: [{trend_string}]\n"
        )
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")

# 3. REQUEST STRUCTURED ANALYSIS FROM GEMINI
prompt = f"""
You are an expert institutional technical analyst. Based on the market data summary, volume metrics (OBV), momentum indicators (MACD), calculated Risk/Reward profiles, and the provided "Entry Cost" below, analyze each individual stock.

CRITICAL ANALYSIS REQUIREMENT:
- For "cost", map back the EXACT "Entry Cost" value provided to you in the data input. Do not alter it.
- Factor the **Risk/Reward** ratio heavily into your decisions. If a stock is trading immediately underneath its 1-Month Resistance ceiling (a poor ratio), protect capital and avoid issuing a "Buy" regardless of how bullish the MACD looks.
- Factor the **OBV Trend** (Volume validation) and **MACD Status** (Momentum environment/extension/crossover) explicitly into your trend determination.
- For "recommendation" (Buy/Hold/Sell) and "important_note", evaluate the market technicals (Price vs Support/Resistance, Volume, and Momentum) in relation to that Entry Cost.

Stocks to analyze: {', '.join(tickers)}
Data Input:
{data_summary}

CRITICAL INSTRUCTION: You must reply ONLY with a valid, clean JSON array of objects. Do not wrap it in ```json blocks, and do not include any extra text. Each object in the JSON array must follow this exact schema:
{{
  "stock_name": "TICKER",
  "cost": "The exact entry cost provided to you",
  "latest_price": "Latest close price value",
  "support": "Support level value",
  "resistance": "Resistance level value",
  "obv_status": "e.g., Rising / Falling",
  "macd_status": "e.g., Bullish Territory / Bullish Crossover / Bearish Territory",
  "trend": "Bullish/Bearish/Sideways",
  "recommendation": "Buy/Hold/Sell",
  "important_note": "Technical commentary taking their entry cost, OBV validation, MACD momentum state, and structural Risk/Reward ratio into consideration"
}}
"""

print("Generating structured technical analysis via Gemini API...")
response = client.models.generate_content(
    model='gemini-3.5-flash',
    contents=prompt,
)

# Clean up raw markdown block wrappers if generated by the model
raw_json = response.text.strip()
if raw_json.startswith("```"):
    lines = raw_json.splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines[-1].startswith("```"):
        lines = lines[:-1]
    raw_json = "\n".join(lines).strip()

try:
    analysis_data = json.loads(raw_json)
except Exception as e:
    print("Failed to parse JSON. Falling back to an empty template table structure.")
    analysis_data = [
        {
            "stock_name": t,
            "cost": f"{my_costs.get(t, 0.0):.2f}" if my_costs.get(t, 0.0) > 0 else "N/A",
            "latest_price": "Error",
            "support": "Error",
            "resistance": "Error",
            "obv_status": "Error",
            "macd_status": "Error",
            "trend": "Error",
            "recommendation": "Error",
            "important_note": "Failed to parse data."
        } for t in tickers
    ]

# 4. COMPILE REPORT INTO PDF TABLE LAYOUT
class CorporatePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 41, 59) # Cool Slate 800
        self.cell(0, 10, "Daily Market Report - Watchlist Technical Summary", new_x="LMARGIN", new_y="NEXT", align="L")
        
        # Explicitly calculate current Thailand Time (UTC+7)
        thailand_tz = timezone(timedelta(hours=7))
        now_thailand = datetime.now(thailand_tz)
        thai_timestamp = now_thailand.strftime('%Y-%m-%d %H:%M:%S')
        
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139) # Slate 500
        self.cell(0, 5, f"Generated automatically on {thai_timestamp} (Thailand Time)", new_x="LMARGIN", new_y="NEXT", align="L")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184) # Slate 400
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

pdf = CorporatePDF()
pdf.add_page()

with pdf.table(col_widths=(15, 15, 15, 15, 15, 15, 15, 20, 20, 45), text_align="LEFT") as table:
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(15, 23, 42) # Slate 900
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support", "Resist.", "OBV", "MACD", "Trend", "Rec.", "Important Note"]
    for header_title in headers:
        header_row.cell(header_title)

    pdf.set_font("Helvetica", "", 8)
    for stock in analysis_data:
        row = table.row()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        
        pdf.set_text_color(51, 65, 85)
        row.cell(str(stock.get("stock_name", "")))
        row.cell(str(stock.get("cost", "")))
        row.cell(str(stock.get("latest_price", "")))
        row.cell(str(stock.get("support", "")))
        row.cell(str(stock.get("resistance", "")))
        row.cell(str(stock.get("obv_status", "")))
        row.cell(str(stock.get("macd_status", "")))
        
        if "bullish" in trend_status:
            pdf.set_text_color(34, 197, 94)   # Vibrant Green
        elif "bearish" in trend_status:
            pdf.set_text_color(239, 68, 68)   # Vibrant Red
        else:
            pdf.set_text_color(51, 65, 85)    # Default Slate 700
        row.cell(str(stock.get("trend", "")))
        
        if "buy" in rec_status:
            pdf.set_text_color(34, 197, 94)   # Vibrant Green
        elif "sell" in rec_status:
            pdf.set_text_color(239, 68, 68)   # Vibrant Red
        else:
            pdf.set_text_color(234, 179, 8)   # Vibrant Amber for Hold
        row.cell(str(stock.get("recommendation", "")))
        
        pdf.set_text_color(51, 65, 85)
        row.cell(str(stock.get("important_note", "")))

filename = "morning_market_analysis.pdf"
pdf.output(filename)
print(f"PDF output finalized successfully as {filename}.")
