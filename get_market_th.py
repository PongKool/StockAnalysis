import os
import json
import yfinance as yf
from google import genai
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
import pandas as pd

# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION FIRST (THAI SET WATCHLIST)
tickers = ["ADVANC.BK", "AOT.BK", "BBL.BK", "GULF.BK", "PRM.BK", "KTB.BK", "PTT.BK", "SCB.BK", "WHA.BK"]

my_costs = {
    "ADVANC.BK": 370.62,       # Custom entries configured in THB
    "AOT.BK": 55.00,
    "BBL.BK": 168.45,
    "GULF.BK": 57.92,
    "PRM.BK": 8.70,
    "KTB.BK": 32.53,
    "PTT.BK": 32.12,
    "SCB.BK": 143.38,
    "WHA.BK": 4.87
}

# Dictionary to hold the exact calculated numbers for the PDF table mapping
calculated_market_data = {}

# Initialize Gemini Client
client = genai.Client()

print("Fetching technical data from Yahoo Finance for Thai Equities...")
data_summary = ""

# 2. DATA GATHERING LOOP
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        # Pull unadjusted historical matrix data
        hist = stock.history(period="3mo", auto_adjust=False)
        
        if hist.empty or len(hist) < 26:
            continue
            
        # Clean out any incomplete live/placeholder rows containing NaN
        hist = hist.dropna(subset=['Close'])
        
        # Grab the absolute real quote closing price from summary info metadata
        # This completely bypasses the data frame adjustments to lock onto the precise market closing bell price
        try:
            info = stock.info
            latest_close = info.get('regularMarketPrice') or info.get('currentPrice') or hist['Close'].iloc[-1]
        except Exception:
            latest_close = hist['Close'].iloc[-1]
        
        # --- CALCULATE OBV ---
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        obv_trend = "Rising" if obv.tail(5).diff().mean() > 0 else "Falling"
        
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
            if prev_macd <= prev_signal:
                macd_status = "Bullish Crossover"
            else:
                macd_status = "Bullish Territory"
        else:
            if prev_macd >= prev_signal:
                macd_status = "Bearish Crossover"
            else:
                macd_status = "Bearish Territory"
                
        cost_val = my_costs.get(ticker, 0.0)
        actual_cost = f"{cost_val:.2f}" if cost_val > 0 else "N/A"
        
        # Support and Resistance levels
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
            calculated_ratio = reward_distance / risk_distance
            rr_ratio_str = f"1:{calculated_ratio:.2f}"
            
        recent_closes = hist_1m.tail(14)
        trend_string = ", ".join([f"{row['Close']:.2f}" for _, row in recent_closes.iterrows()])
        
        # Save exact programmatic calculations to map directly into the PDF table row builder later
        calculated_market_data[ticker] = {
            "latest_price": f"{latest_close:.2f}",
            "support": f"{support_level:.2f}",
            "resistance": f"{resistance_level:.2f}"
        }
        
        data_summary += (
            f"Ticker: {ticker} | Entry Cost (THB): {actual_cost} | Latest Close (THB): {latest_close:.2f} | "
            f"1Mo Support: {support_level:.2f} | 1Mo Resistance: {resistance_level:.2f} | "
            f"Risk/Reward: {rr_ratio_str} | "
            f"OBV: {latest_obv:.0f} ({obv_trend}) | MACD: {latest_macd:.2f} (Signal: {latest_signal:.2f}, {macd_status}) | "
            f"Recent Close Trend: [{trend_string}]\n"
        )
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")

# 3. REQUEST STRUCTURED ANALYSIS FROM GEMINI
prompt = f"""
You are an expert institutional technical analyst evaluating equities on the Stock Exchange of Thailand (SET). All currency denominations are in Thai Baht (THB). Based on the market data summary, volume metrics (OBV), momentum indicators (MACD), calculated Risk/Reward profiles, and the provided "Entry Cost" below, analyze each individual stock.

CRITICAL ANALYSIS REQUIREMENT:
- For "cost", map back the EXACT "Entry Cost" value provided to you in the data input. Do not alter it.
- Factor the **Risk/Reward** ratio heavily into your decisions. If a stock is trading immediately underneath its 1-Month Resistance ceiling (a poor ratio), protect capital and avoid issuing a "Buy" regardless of how bullish the MACD looks.
- Factor the **OBV Trend** (Volume validation) and **MACD Status** (Momentum environment/extension/crossover) explicitly into your trend determination.
- For "recommendation" (Buy/Hold/Sell) and "important_note", evaluate the market technicals (Price vs Support/Resistance, Volume, and Momentum) in relation to that Entry Cost.

We only require the LLM to output recommendation, trend status, obv_status, macd_status and structural text insights.
Stocks to analyze: {', '.join(tickers)}
Data Input:
{data_summary}

CRITICAL INSTRUCTION: You must reply ONLY with a valid, clean JSON array of objects. Do not wrap it in ```json blocks, and do not include any extra text. Each object in the JSON array must follow this exact schema:
{{
  "stock_name": "TICKER",
  "cost": "The exact entry cost provided to you",
  "obv_status": "e.g., Rising / Falling",
  "macd_status": "e.g., Bullish Territory / Bullish Crossover / Bearish Territory",
  "trend": "Bullish/Bearish/Sideways",
  "recommendation": "Buy/Hold/Sell",
  "important_note": "Technical commentary taking their entry cost in THB, OBV validation, MACD momentum state, and structural Risk/Reward ratio into consideration"
}}
"""

print("Generating structured technical analysis via Gemini API...")
response = client.models.generate_content(
    model='gemini-3.5-flash',
    contents=prompt,
)

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
    print("Failed to parse JSON. Falling back to template table structure.")
    analysis_data = [
        {
            "stock_name": t,
            "cost": f"{my_costs.get(t, 0.0):.2f}" if my_costs.get(t, 0.0) > 0 else "N/A",
            "obv_status": "Error",
            "macd_status": "Error",
            "trend": "Error",
            "recommendation": "Error",
            "important_note": "Failed to parse data payload safely."
        } for t in tickers
    ]

# 4. COMPILE REPORT INTO PDF TABLE LAYOUT
class CorporatePDF(FPDF):
    def header(self):
        # Top decorative primary accent bar
        self.set_fill_color(30, 41, 59) # Deep Slate Blue
        self.rect(0, 0, 210, 4, "F")
        
        self.ln(4)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42) # Dark Charcoal
        self.cell(0, 10, "Daily Thai Market Report", new_x="LMARGIN", new_y="NEXT", align="L")
        
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(79, 70, 229) # Indigo Accent
        self.cell(0, 5, "WATCHLIST TECHNICAL SUMMARY (SET)", new_x="LMARGIN", new_y="NEXT", align="L")
        
        thailand_tz = timezone(timedelta(hours=7))
        now_thailand = datetime.now(thailand_tz)
        thai_timestamp = now_thailand.strftime('%Y-%m-%d %H:%M:%S')
        
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139) # Muted Slate
        self.cell(0, 5, f"Generated automatically on {thai_timestamp} (Thailand Time)", new_x="LMARGIN", new_y="NEXT", align="L")
        
        # Subtle divider line under header
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y() + 4, 200, self.get_y() + 4)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        # Subtle top divider line for footer
        self.set_draw_color(241, 245, 249)
        self.line(10, self.get_y(), 200, self.get_y())
        
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

pdf = CorporatePDF()
pdf.add_page()

# Setup Table Styles (Professional spacing and explicit column configuration)
pdf.set_font("Helvetica", "", 8)
# Total width = 190mm (Fits perfectly on standard A4 with 10mm side margins)
column_widths = (20, 13, 13, 14, 14, 12, 24, 16, 11, 53) 

with pdf.table(
    col_widths=column_widths, 
    text_align="LEFT",
    line_height=5,
    padding=2,
    outer_border_width=0.5,
    headers_line_height=6
) as table:
    
    # --- HEADER ROW ---
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(255, 255, 255) # White text for header
    pdf.set_fill_color(30, 41, 59)     # Deep Slate Background
    
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support", "Resist.", "OBV", "MACD", "Trend", "Rec.", "Important Note (THB Context)"]
    for header_title in headers:
        header_row.cell(header_title)
    
    # --- DATA ROWS ---
    for idx, stock in enumerate(analysis_data):
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        
        market_metrics = calculated_market_data.get(ticker, {"latest_price": "N/A", "support": "N/A", "resistance": "N/A"})
        
        # Zebra striping background configuration
        if idx % 2 == 0:
            pdf.set_fill_color(255, 255, 255) # Pure White
        else:
            pdf.set_fill_color(248, 250, 252) # Off-White / Light Grey
            
        # Standard structural text color
        pdf.set_text_color(51, 65, 85) 
        
        # Base cells
        row.cell(ticker)
        row.cell(str(stock.get("cost", "")))
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
        row.cell(str(stock.get("obv_status", "")))
        row.cell(str(stock.get("macd_status", "")))
        
        # Conditional Formatting for Trend (Muted, premium color variants)
        if "bullish" in trend_status:
            pdf.set_text_color(21, 128, 61)   # Emerald Dark Green
        elif "bearish" in trend_status:
            pdf.set_text_color(185, 28, 28)   # Crimson Dark Red
        else:
            pdf.set_text_color(51, 65, 85)
        row.cell(str(stock.get("trend", "")))
        
        # Conditional Formatting for Recommendation
        if "buy" in rec_status:
            pdf.set_text_color(21, 128, 61)   # Emerald Dark Green
        elif "sell" in rec_status:
            pdf.set_text_color(185, 28, 28)   # Crimson Dark Red
        else:
            pdf.set_text_color(180, 83, 9)    # Amber Dark Yellow
        row.cell(str(stock.get("recommendation", "")))
        
        # Reset color to soft slate for the descriptive note block
        pdf.set_text_color(71, 85, 105)
        row.cell(str(stock.get("important_note", "")))

filename = "thai_market_analysis.pdf"
pdf.output(filename)
print(f"Thai portfolio analysis finalized successfully as {filename}.")
