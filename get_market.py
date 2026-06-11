import os
import json
import yfinance as yf
from google import genai
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
import pandas as pd

# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION FIRST
tickers = ["MU", "NVDA", "ORCL", "SNDK", "MSFT", "TSM", "LLY", "LRCX", "NOW", "AMD", "CACI", "AVGO", "ANET"]
my_costs = {
    "MU": 424.62, "NVDA": 220.80, "ORCL": 183.72, "SNDK": 1418.17, 
    "MSFT": 455.37, "TSM": 424.30, "LLY": 971.12, "LRCX": 305.41, 
    "NOW": 107.68, "AMD": 448.37, "CACI": 524.53, "AVGO": 446.13, "ANET": 171.11
}

# Dictionary to hold the exact calculated numbers for the PDF table mapping
calculated_market_data = {}

# Initialize Gemini Client
client = genai.Client()

print("Fetching technical data from Yahoo Finance...")
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
        
        # Determine if the current position is profitable
        is_profitable = "Yes" if (cost_val > 0 and latest_close > cost_val) else "No"

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

        # Pack structured indicators and profit-states into the text stream context
        data_summary += (
            f"Ticker: {ticker} | Entry Cost: {actual_cost} | Latest Close: {latest_close:.2f} | "
            f"Is Position Profitable?: {is_profitable} | "
            f"1Mo Support: {support_level:.2f} | 1Mo Resistance: {resistance_level:.2f} | "
            f"Risk/Reward: {rr_ratio_str} | "
            f"OBV: {latest_obv:.0f} ({obv_trend}) | MACD: {latest_macd:.2f} (Signal: {latest_signal:.2f}, {macd_status}) | "
            f"Recent Close Trend: [{trend_string}]\n"
        )

    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")

# 3. REQUEST STRUCTURED ANALYSIS FROM GEMINI
prompt = f"""
You are an expert institutional technical analyst and risk manager. Your primary objective is to evaluate current positions and protect open capital using dynamic technical momentum indicators (MACD and OBV) as trailing Take Profit (exit) criteria.

CRITICAL TAKE-PROFIT EXIT ANALYSIS RULES:
1. First, check the "Is Position Profitable?" metric for the stock.
2. If the position is profitable ("Yes"), prioritize locking in gains over blindly holding:
   - **Take Profit / Sell Trigger:** If "MACD Status" reflects a "Bearish Crossover" OR the OBV trend is "Falling", underlying momentum/volume is exhausted. You MUST set the "recommendation" to "Sell" to take profits.
   - **Hold Trend:** If the position is profitable, but the MACD is in "Bullish Territory" and OBV is "Rising", allow profits to run and set the recommendation to "Hold".
3. If "Is Position Profitable?" is "No", look to "Hold" if a technical support recovery is forming, or "Sell" defensively if support structural levels break.
4. For the "cost" field in output, map back the EXACT "Entry Cost" value provided to you in the data input.

Stocks to analyze: {', '.join(tickers)}
Data Input: {data_summary}

CRITICAL INSTRUCTION: You must reply ONLY with a valid, clean JSON array of objects. Do not wrap it in ```json blocks, and do not include any extra text.
Each object in the JSON array must follow this exact schema:
{{
  "stock_name": "TICKER",
  "cost": "The exact entry cost provided to you",
  "obv_status": "e.g., Rising / Falling",
  "macd_status": "e.g., Bullish Territory / Bearish Crossover / Bearish Territory",
  "trend": "Bullish/Bearish/Sideways",
  "recommendation": "Buy/Hold/Sell",
  "important_note": "Provide rigorous justification stating why the combination of profit state, MACD crossover, or OBV structural trend dictates a Hold or a Take-Profit Sell."
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
        self.set_fill_color(30, 41, 59) # Deep Slate Navy
        self.rect(0, 0, 210, 4, "F")
        self.ln(4)
        
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42) # Dark Charcoal
        self.cell(0, 10, "Daily Market Report", new_x="LMARGIN", new_y="NEXT", align="L")
        
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(79, 70, 229) # Indigo Sub-header Accent
        self.cell(0, 5, "WATCHLIST TECHNICAL SUMMARY", new_x="LMARGIN", new_y="NEXT", align="L")
        
        thailand_tz = timezone(timedelta(hours=7))
        now_thailand = datetime.now(thailand_tz)
        thai_timestamp = now_thailand.strftime('%Y-%m-%d %H:%M:%S')
        
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139) # Muted Slate Grey
        self.cell(0, 5, f"Generated automatically on {thai_timestamp} (Thailand Time)", new_x="LMARGIN", new_y="NEXT", align="L")
        
        # Elegant header divider line
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y() + 4, 200, self.get_y() + 4)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        # Soft divider line for footer
        self.set_draw_color(241, 245, 249)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

pdf = CorporatePDF()
pdf.add_page()

# Table Structural Configuration
column_widths = (14, 13, 13, 13, 13, 12, 23, 16, 12, 61)

with pdf.table(col_widths=column_widths, text_align="LEFT", line_height=6, padding=1.5, outer_border_width=0.5) as table:
    # --- HEADER ROW ---
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(255, 255, 255) # White text for headers
    pdf.set_fill_color(30, 41, 59)     # Deep Slate Blue Background
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support", "Resist.", "OBV", "MACD", "Trend", "Rec.", "Important Note"]
    for header_title in headers:
        header_row.cell(header_title)

    # --- DATA ROWS ---
    for idx, stock in enumerate(analysis_data):
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        market_metrics = calculated_market_data.get(ticker, {"latest_price": "N/A", "support": "N/A", "resistance": "N/A"})

        # Zebra Striping Background Configuration
        if idx % 2 == 0:
            pdf.set_fill_color(255, 255, 255) # Pure White Row
        else:
            pdf.set_fill_color(248, 250, 252) # Light Grey Alternating Row

        # Base font properties for row structure
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85) # Slate Charcoal Text

        # Standard Data Nodes
        row.cell(ticker)
        row.cell(str(stock.get("cost", "")))
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
        row.cell(str(stock.get("obv_status", "")))
        row.cell(str(stock.get("macd_status", "")))

        # Premium/Muted Conditional Color Design for Trend
        if "bullish" in trend_status:
            pdf.set_text_color(21, 128, 61)   # Emerald Green
        elif "bearish" in trend_status:
            pdf.set_text_color(185, 28, 28)   # Deep Crimson
        else:
            pdf.set_text_color(51, 65, 85)
        row.cell(str(stock.get("trend", "")))

        # Premium/Muted Conditional Color Design for Recommendation
        if "buy" in rec_status:
            pdf.set_text_color(21, 128, 61)   # Emerald Green
        elif "sell" in rec_status:
            pdf.set_text_color(185, 28, 28)   # Deep Crimson
        else:
            pdf.set_text_color(180, 83, 9)    # Muted Dark Amber
        row.cell(str(stock.get("recommendation", "")))

        # Clean paragraph configuration for multi-line LLM commentary blocks
        pdf.set_font("Helvetica", "", 7)  # Slightly smaller font to prevent vertical bloat
        pdf.set_text_color(71, 85, 105)   # Softer grey body font
        row.cell(str(stock.get("important_note", "")))

filename = "morning_market_analysis.pdf"
pdf.output(filename)
print(f"PDF output finalized successfully as {filename}.")
