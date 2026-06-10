import os
import json
import yfinance as yf
from google import genai
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
import pandas as pd

# 1. Initialize Gemini Client
client = genai.Client()

tickers = ["MU", "NVDA", "ORCL", "SNDK", "MSFT", "TSM", "LLY", "LRCX", "NOW", "AMD", "CACI", "AVGO", "ANET"]

my_costs = {
    "MU": 424.62, "NVDA": 220.80, "ORCL": 183.72, "SNDK": 1418.17, "MSFT": 455.37,
    "TSM": 424.30, "LLY": 971.12, "LRCX": 305.41, "NOW": 107.68, "AMD": 448.37,
    "CACI": 524.53, "AVGO": 446.13, "ANET": 171.11
}

print("Fetching technical data from Yahoo Finance...")
data_summary = ""
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty or len(hist) < 26:
            continue
        
        # OBV Calculation
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        obv_trend = "Rising" if obv.tail(5).diff().mean() > 0 else "Falling"
        
        # MACD Core Formulas
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
        
        hist_1m = hist.tail(21)
        support_level = hist_1m['Low'].min()
        resistance_level = hist_1m['High'].max()
        latest_close = hist_1m['Close'].iloc[-1]
        
        # Explicit Mathematical Risk / Reward Bounds Tracking
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close
        
        if reward_distance <= 0:
            rr_ratio_str = "Poor (At Res)"
        elif risk_distance <= 0:
            rr_ratio_str = "Exc (At Supp)"
        else:
            calculated_ratio = reward_distance / risk_distance
            rr_ratio_str = f"1:{calculated_ratio:.2f}"
            
        recent_closes = hist_1m.tail(14)
        trend_string = ", ".join([f"{row['Close']:.2f}" for _, row in recent_closes.iterrows()])
        
        data_summary += (
            f"Ticker: {ticker} | Entry Cost: {actual_cost} | Latest Close: {latest_close:.2f} | "
            f"1Mo Support: {support_level:.2f} | 1Mo Resistance: {resistance_level:.2f} | "
            f"Risk/Reward: {rr_ratio_str} | "
            f"OBV: {latest_obv:.0f} ({obv_trend}) | MACD: {latest_macd:.2f} (Signal: {latest_signal:.2f}, {macd_status}) | "
            f"Recent Close Trend: [{trend_string}]\n"
        )
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")

# 3. Request structured JSON format from Gemini enforcing strict Quantitative Filters
prompt = f"""
You are an expert institutional technical analyst. Evaluate each stock asset strictly filtering standard momentum indicators through the provided quantitative Risk/Reward calculations.

CRITICAL RISK-MANAGEMENT RULES:
1. Strict Risk/Reward Filter: If the Risk/Reward status says "Poor (At Res)" or the calculated mathematical upside ratio is less than 1:1.50, you are strictly FORBIDDEN from issuing a "BUY" recommendation, even if the MACD indicator is strongly bullish. Overextended assets must be categorized as HOLD or SELL to preserve capital.
2. If an asset is flagged with an asymmetric profile like "Exc (At Supp)" and volume metrics (OBV) are Rising, favor a BUY execution.
3. Your text commentary inside "important_note" must explicitly justify how the decision respects these quantitative boundaries relative to the Entry Cost.

CRITICAL JSON STRUCTURAL REQUIREMENT:
You must return a raw JSON array of objects. Each object MUST contain exactly these keys:
- "stock_name": The ticker string (e.g., "NVDA")
- "cost": The exact entry cost string provided in the data input.
- "latest_price": The numeric latest close price string.
- "support": The 1Mo Support string.
- "resistance": The 1Mo Resistance string.
- "obv_status": The OBV trend status (e.g., "Rising" or "Falling").
- "macd_status": The current MACD status string.
- "trend": Short trend text (e.g., "Bullish", "Bearish", "Sideways").
- "recommendation": A clear target string ("BUY", "HOLD", or "SELL") conforming strictly to the risk logic filters above.
- "important_note": A highly professional 1-2 sentence technical commentary summary focusing on risk dynamics.

Stocks to analyze: {', '.join(tickers)}
Data Input: {data_summary}
"""

print("Generating structured technical analysis via Gemini API...")
response = client.models.generate_content(
    model='gemini-3.5-flash', 
    contents=prompt,
    config={'response_mime_type': 'application/json'}
)

raw_json = response.text.strip()

try:
    analysis_data = json.loads(raw_json)
except Exception as e:
    print(f"Failed to parse JSON directly ({e}). Executing emergency matrix fallback...")
    analysis_data = [
        {
            "stock_name": t, 
            "cost": f"{my_costs.get(t, 0.0):.2f}" if my_costs.get(t, 0.0) > 0 else "N/A", 
            "latest_price": "N/A", "support": "N/A", "resistance": "N/A", 
            "obv_status": "N/A", "macd_status": "N/A", "trend": "N/A", 
            "recommendation": "HOLD", "important_note": "Failed to parse generation stream cleanly."
        } for t in tickers
    ]

# 4. Premium Portrait Design Compilation Engine
class CorporatePortraitPDF(FPDF):
    def header(self):
        # Slate Accent top border bar
        self.set_fill_color(30, 41, 59) # Slate 800
        self.rect(0, 0, 210, 4, "F")
        
        self.ln(6)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(15, 23, 42) # Slate 900
        self.cell(0, 7, "EXECUTIVE MARKET REPORT", new_x="LMARGIN", new_y="NEXT", align="L")
        
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(71, 85, 105) # Slate 600
        self.cell(0, 5, "Watchlist Portfolio Technical & Momentum Summary", new_x="LMARGIN", new_y="NEXT", align="L")
        
        # Explicit Thailand Time Zone Header Sync
        thailand_tz = timezone(timedelta(hours=7))
        thai_timestamp = datetime.now(thailand_tz).strftime('%Y-%m-%d %H:%M:%S')
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(148, 163, 184) # Slate 400
        self.cell(0, -11, f"Generated: {thai_timestamp} (Thailand Time)", new_x="LMARGIN", new_y="NEXT", align="R")
        
        self.ln(14)
        self.set_draw_color(226, 232, 240) # Slate 200
        self.line(12, self.get_y(), 198, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(100, 10, "Internal Strategy Document - Confidential", align="L")
        self.set_x(-40)
        self.cell(28, 10, f"Page {self.page_no()}", align="R")

pdf = CorporatePortraitPDF(orientation='P', unit='mm', format='A4')
pdf.set_margins(12, 12, 12)
pdf.add_page()

# Absolute horizontal mapping dimensions matching sequence fields below exactly
col_widths = (13, 15, 15, 15, 15, 14, 25, 14, 13, 47)
headers = ["Ticker", "Cost", "Price", "Support", "Resist.", "OBV", "MACD Status", "Trend", "Rec.", "Technical Commentary"]

# Render Header Row
pdf.set_font("Helvetica", "B", 7.5)
pdf.set_fill_color(241, 245, 249) # Light Slate 100
pdf.set_text_color(51, 65, 85)     # Slate 700
pdf.set_draw_color(203, 213, 225)  # Slate 300

for i, h_title in enumerate(headers):
    pdf.cell(col_widths[i], 7, h_title, border=1, align="C" if i < 9 else "L", fill=True)
pdf.ln()

# Render Corporate Rows
pdf.set_font("Helvetica", "", 7.5)
row_count = 0

for stock in analysis_data:
    fill_row = row_count % 2 == 1
    pdf.set_fill_color(248, 250, 252) # Alternating slate rows
    pdf.set_text_color(51, 65, 85)
    pdf.set_draw_color(226, 232, 240)
    
    # Pre-calculate cell wrapping demand metrics
    text_note = str(stock.get("important_note", ""))
    lines_needed = pdf.multi_cell(col_widths[9], 4.5, text_note, split_only=True)
    row_h = max(len(lines_needed) * 4.2, 7.5)
    
    # Page Break Intercept handling
    if pdf.get_y() + row_h > 275:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_fill_color(241, 245, 249)
        for i, h_title in enumerate(headers):
            pdf.cell(col_widths[i], 7, h_title, border=1, align="C" if i < 9 else "L", fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 7.5)

    # Sequential row generation under exact width index limits
    pdf.cell(col_widths[0], row_h, str(stock.get("stock_name", "")), border=1, align="C", fill=fill_row)
    pdf.cell(col_widths[1], row_h, f"${stock.get('cost', 'N/A')}", border=1, align="C", fill=fill_row)
    pdf.cell(col_widths[2], row_h, f"${stock.get('latest_price', '0.00')}", border=1, align="C", fill=fill_row)
    pdf.cell(col_widths[3], row_h, f"${stock.get('support', '0.00')}", border=1, align="C", fill=fill_row)
    pdf.cell(col_widths[4], row_h, f"${stock.get('resistance', '0.00')}", border=1, align="C", fill=fill_row)
    pdf.cell(col_widths[5], row_h, str(stock.get("obv_status", "")), border=1, align="C", fill=fill_row)
    pdf.cell(col_widths[6], row_h, str(stock.get("macd_status", "")), border=1, align="L", fill=fill_row)
    
    # Dynamic Trend Accent Texts
    trend_status = str(stock.get("trend", "")).strip().lower()
    if "bullish" in trend_status:
        pdf.set_text_color(22, 163, 74) # Institutional Green
    elif "bearish" in trend_status:
        pdf.set_text_color(220, 38, 38) # Institutional Red
    pdf.cell(col_widths[7], row_h, str(stock.get("trend", "")), border=1, align="C", fill=fill_row)
    pdf.set_text_color(51, 65, 85) # Reset text accent
    
    # Soft Color Execution Badges
    rec_status = str(stock.get("recommendation", "")).strip().lower()
    if "buy" in rec_status:
        pdf.set_fill_color(220, 252, 231) # Pastel Green Fill
        pdf.set_text_color(21, 128, 61)   # Dark Green Text
    elif "sell" in rec_status:
        pdf.set_fill_color(254, 226, 226) # Pastel Red Fill
        pdf.set_text_color(185, 28, 28)   # Dark Red Text
    else:
        pdf.set_fill_color(254, 249, 195) # Pastel Amber Fill
        pdf.set_text_color(161, 98, 7)    # Dark Amber Text
        
    pdf.cell(col_widths[8], row_h, str(stock.get("recommendation", "")), border=1, align="C", fill=True)
    
    # Flawless Dynamic Multi-cell Block Injection
    pdf.set_text_color(71, 85, 105)
    current_y = pdf.get_y()
    
    pdf.multi_cell(col_widths[9], row_h / len(lines_needed), text_note, border=1, align="L", fill=fill_row)
    pdf.set_xy(12, current_y + row_h)
    row_count += 1

filename = "morning_market_analysis.pdf"
pdf.output(filename)
print(f"PDF output finalized successfully as {filename}.")
