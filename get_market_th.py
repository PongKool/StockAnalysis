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

# --- FETCH THAI BLUE-CHIP MACRO REGIME ---
print("Evaluating Thai SET50 Macro Economic Regime...")
macro_regime = "Bullish" # Default fallback
try:
    # TDEX tracks the SET50 Index (Thailand's top 50 blue chips)
    macro_stock = yf.Ticker("TDEX.BK")
    macro_hist = macro_stock.history(period="3mo", auto_adjust=False)
    macro_hist = macro_hist.dropna(subset=['Close'])
    
    # Calculate 20-Day EMA to determine overall market posture
    macro_ema20 = macro_hist['Close'].ewm(span=20, adjust=False).mean()
    latest_macro_close = macro_hist['Close'].iloc[-1]
    latest_macro_ema = macro_ema20.iloc[-1]
    
    if latest_macro_close < latest_macro_ema:
        macro_regime = "Bearish/Cautious"
except Exception as e:
    print(f"Warning: Could not calculate Thai macro regime: {e}")

print(f"Current Thai Market Regime: {macro_regime}")
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
        
        # Determine if the current Thai position is profitable
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
        
        # Pack structured indicators and profit-states into the text stream context (THB Focus)
        data_summary += (
            f"Ticker: {ticker} | Entry Cost (THB): {actual_cost} | Latest Close (THB): {latest_close:.2f} | "
            f"Is Position Profitable?: {is_profitable} | "
            f"1Mo Support: {support_level:.2f} | 1Mo Resistance: {resistance_level:.2f} | "
            f"Risk/Reward: {rr_ratio_str} | "
            f"OBV: {latest_obv:.0f} ({obv_trend}) | MACD: {latest_macd:.2f} (Signal: {latest_signal:.2f}, {macd_status}) | "
            f"Recent Close Trend: [{trend_string}]\n"
        )
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")

# 3. REQUEST STRUCTURED ANALYSIS FROM GEMINI (VALUE & CAP-PRESERVATION LOGIC)
prompt = f"""
You are an institutional conservative asset manager evaluating premier defensive and value equities on the Stock Exchange of Thailand (SET). All currency denominations are in Thai Baht (THB).
You are given the 'GLOBAL THAI MARKET REGIME' context derived from the SET50 Index ETF (TDEX): **{macro_regime}**. Use this to gauge systemic domestic liquidity and risk.

CRITICAL VALUE-PORTFOLIO RISK & DEFENSE RULES:
1. **Capital Preservation & Risk Management (Exits):**
   - **Take-Profit Exit:** If a position is profitable ("Yes") but shows structural breakdown ("MACD Status" is a "Bearish Crossover" OR OBV trend is "Falling"), downgrade to **Take-Profit Exit** to lock in profits.
   - **Support-Aware Stop-Loss:** If a position is losing money ("No") AND the Latest Close price has broken structurally below the calculated 1Mo Support floor, downgrade to **Support-Aware Stop-Loss** immediately to cut losses. If it is losing money but still holding above or at the support floor, maintain a **Hold** to see if the demand area triggers a rebound.
2. **Increasing Positions / Accumulation ("Buy" or "Hold"):**
   - Issue a **"Buy"** or an explicit **"Hold (Accumulate)"** recommendation if a stock demonstrates clear potential to go up.
   - Strong potential is defined as having a **"Rising" OBV trend** (volume accumulation) combined with a healthy MACD profile (**"Bullish Territory"** or a fresh **"Bullish Crossover"**).
   - Even if the stock is not at its absolute 1-month support floor, prioritize this rising volume + MACD momentum combination as a signal to scale into or increase the position safely.

OUTPUT INSTRUCTION FOR THE 'IMPORTANT_NOTE' FIELD:
1. You MUST explicitly mention how the combination of the **Rising OBV volume trend** and the **MACD status** justified your decision to buy or increase positions.
2. You must explicitly integrate the calculated **Risk/Reward (R/R)** ratio provided in the data input string into this summary text.
3. Keep it tightly concise enough to fit the table cell row layout without overflowing.

Stocks to analyze: {', '.join(tickers)}
Data Input: {data_summary}

CRITICAL INSTRUCTION: You must reply ONLY with a valid, clean JSON array of objects. Do not wrap it in ```json blocks, and do not include any extra text.
Each object in the JSON array must follow this exact schema:
{{
  "stock_name": "TICKER",
  "cost": "The exact entry cost provided to you",
  "obv_status": "e.g., Rising / Falling",
  "macd_status": "e.g., Bullish Territory / Bullish Crossover / Bearish Territory",
  "trend": "Bullish/Bearish/Sideways",
  "recommendation": "Buy/Hold/Hold (Accumulate)/Take-Profit Exit/Support-Aware Stop-Loss",
  "important_note": "Rigorous technical commentary explaining how the explicit Risk/Reward ratio, Thai profit state, and the core OBV + MACD trend combination drive your decision."
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

# Setup Table Styles (Exactly 10 Columns adding up to 190mm printable width)
pdf.set_font("Helvetica", "", 8)
column_widths = (18, 14, 14, 14, 14, 15, 18, 15, 13, 55)

with pdf.table(col_widths=column_widths, text_align="LEFT", line_height=6, padding=2, outer_border_width=0.5) as table:
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
        
        # Conditional Formatting for Trend (Muted, premium variants)
        if "bullish" in trend_status:
            pdf.set_text_color(21, 128, 61)   # Emerald Dark Green
        elif "bearish" in trend_status:
            pdf.set_text_color(185, 28, 28)   # Crimson Dark Red
        else:
            pdf.set_text_color(51, 65, 85)
        row.cell(str(stock.get("trend", "")))
        
        # Conditional Formatting for Recommendation (Updated vocabulary matching prompt)
        if "buy" in rec_status or "accumulate" in rec_status:
            pdf.set_text_color(21, 128, 61)   # Emerald Dark Green
        elif "exit" in rec_status or "loss" in rec_status or "sell" in rec_status:
            pdf.set_text_color(185, 28, 28)   # Crimson Dark Red
        else:
            pdf.set_text_color(180, 83, 9)    # Amber Dark Yellow
        row.cell(str(stock.get("recommendation", "")))
        
        # Reset color to soft slate for the descriptive note block
        pdf.set_text_color(71, 85, 105)
        
        # --- FIX UNICODE ENCODING CRASH ---
        important_note_clean = str(stock.get("important_note", "")).replace("–", "-")
        important_note_clean = important_note_clean.encode('latin-1', 'replace').decode('latin-1')
        row.cell(important_note_clean)

filename = "thai_market_analysis.pdf"
pdf.output(filename)
print(f"Thai portfolio analysis finalized successfully as {filename}.")
