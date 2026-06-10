import os
import json
import yfinance as yf
from google import genai
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
import pandas as pd

# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION (THAI SET WATCHLIST)
tickers = ["ADVANC.BK", "AOT.BK", "BBL.BK", "GULF.BK", "PRM.BK", "KTB.BK", "PTT.BK", "SCB.BK", "WHA.BK"]

my_costs = {
    "ADVANC.BK": 370.62,
    "AOT.BK": 55.00,
    "BBL.BK": 168.45,
    "GULF.BK": 57.92,
    "PRM.BK": 8.70,
    "KTB.BK": 32.53,
    "PTT.BK": 32.12,
    "SCB.BK": 143.38,
    "WHA.BK": 4.87
}

calculated_market_data = {}
client = genai.Client()

print("Fetching technical data from Yahoo Finance for Thai Equities...")
data_summary = ""

# 2. DATA GATHERING LOOP
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo", auto_adjust=False)
        
        if hist.empty or len(hist) < 26:
            continue
            
        hist = hist.dropna(subset=['Close'])
        
        try:
            info = stock.info
            latest_close = info.get('regularMarketPrice') or info.get('currentPrice') or hist['Close'].iloc[-1]
        except Exception:
            latest_close = hist['Close'].iloc[-1]
        
        # CALCULATE OBV
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        obv_trend = "Rising" if obv.tail(5).diff().mean() > 0 else "Falling"
        
        # CALCULATE MACD
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
        
        # CALCULATE RISK/REWARD RATIO
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
You are an expert institutional technical analyst evaluating equities on the Stock Exchange of Thailand (SET). All currency denominations are in Thai Baht (THB). Based on the market data summary provided below, analyze each stock comprehensively.

CRITICAL ANALYSIS REQUIREMENT:
- For "cost", map back the EXACT "Entry Cost" value provided to you in the data input. Do not alter it.
- Factor the **Risk/Reward** ratio heavily into your decisions. If a stock is trading immediately underneath its 1-Month Resistance ceiling (a poor ratio), protect capital and avoid issuing a "Buy" recommendation.
- Factor the **OBV Trend** (Volume validation) and **MACD Status** (Momentum environment/extension/crossover) explicitly into your trend determination.
- For "recommendation" (Buy/Hold/Sell) and "important_note", evaluate the market technicals (Price vs Support/Resistance, Volume, and Momentum) in relation to that Entry Cost.

We only require the LLM to output: recommendation, trend status, obv_status, macd_status and structural text insights.
Stocks to analyze: {', '.join(tickers)}

Data Input:
{data_summary}

CRITICAL INSTRUCTION: You must reply ONLY with a valid, clean JSON array of objects. Do not wrap it in ```json blocks, and do not include any extra text. Each object in the JSON array must follow this exact structure:
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

# 4. COMPILE REPORT INTO PROFESSIONAL PDF LAYOUT
class ProfessionalPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        # Define professional color scheme
        self.DARK_BLUE = (31, 41, 55)
        self.LIGHT_BLUE = (59, 130, 246)
        self.ACCENT_GRAY = (107, 114, 128)
        self.GREEN = (34, 197, 94)
        self.RED = (239, 68, 68)
        self.YELLOW = (234, 179, 8)
        self.LIGHT_GRAY = (243, 244, 246)
        self.WHITE = (255, 255, 255)
        self.DARK_TEXT = (15, 23, 42)

    def header(self):
        # Top decorative bar
        self.set_fill_color(*self.DARK_BLUE)
        self.rect(0, 0, self.WIDTH, 8, 'F')
        
        # Logo/Title section with accent
        self.set_xy(10, 8)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*self.LIGHT_BLUE)
        self.cell(0, 8, "THAI MARKET ANALYSIS", new_x="LMARGIN", new_y="NEXT")
        
        self.set_xy(10, 16)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*self.DARK_BLUE)
        self.cell(0, 6, "Daily Technical Analysis Report - Stock Exchange of Thailand (SET)", new_x="LMARGIN", new_y="NEXT")
        
        # Timestamp section
        thailand_tz = timezone(timedelta(hours=7))
        now_thailand = datetime.now(thailand_tz)
        thai_timestamp = now_thailand.strftime('%d %B %Y | %H:%M:%S')
        
        self.set_xy(10, 22)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*self.ACCENT_GRAY)
        self.cell(0, 5, f"Report Generated: {thai_timestamp} (Bangkok Time)", new_x="LMARGIN", new_y="NEXT")
        
        # Decorative line
        self.set_xy(10, 28)
        self.set_draw_color(*self.LIGHT_BLUE)
        self.set_line_width(0.5)
        self.line(10, 28, 200, 28)
        
        self.ln(8)

    def footer(self):
        self.set_y(-20)
        
        # Decorative line
        self.set_draw_color(*self.LIGHT_BLUE)
        self.set_line_width(0.5)
        self.line(10, self.y, 200, self.y)
        
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.ACCENT_GRAY)
        
        # Left: Document info
        self.set_xy(10, self.y)
        self.cell(100, 5, "Thai Market Technical Analysis", new_x="LMARGIN", new_y="NEXT")
        
        # Right: Page number
        self.set_xy(130, self.y - 5)
        self.set_text_color(*self.ACCENT_GRAY)
        self.cell(0, 5, f"Page {self.page_no()}", align="R")

    def add_section_title(self, title):
        """Add a professional section title"""
        self.set_y(self.get_y() + 3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.LIGHT_BLUE)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT", border="B")
        self.set_draw_color(*self.LIGHT_BLUE)
        self.line(10, self.get_y() - 1, 200, self.get_y() - 1)
        self.ln(3)

    def add_legend(self):
        """Add legend for recommendations and trends"""
        self.add_section_title("Legend & Key Indicators")
        
        legend_items = [
            ("Buy", self.GREEN, "Strong bullish signal with favorable risk/reward ratio"),
            ("Hold", self.YELLOW, "Neutral position; wait for clearer directional signal"),
            ("Sell", self.RED, "Bearish momentum with unfavorable technicals"),
            ("Bullish", self.GREEN, "Uptrend indicated by technical indicators"),
            ("Bearish", self.RED, "Downtrend indicated by technical indicators"),
            ("Rising OBV", self.GREEN, "Bullish volume confirmation"),
            ("Falling OBV", self.RED, "Bearish volume confirmation"),
        ]
        
        self.set_font("Helvetica", "", 9)
        for label, color, description in legend_items:
            # Colored indicator box
            self.set_fill_color(*color)
            self.set_xy(15, self.get_y())
            self.rect(15, self.get_y(), 3, 3, 'F')
            self.set_xy(20, self.get_y())
            self.set_text_color(*self.DARK_BLUE)
            self.cell(25, 4, label + ":", new_x="RIGHT")
            self.set_text_color(*self.ACCENT_GRAY)
            self.cell(0, 4, description, new_x="LMARGIN", new_y="NEXT")
        
        self.ln(2)

pdf = ProfessionalPDF()
pdf.add_page()

# Add legend section
pdf.add_legend()

# Add analysis section
pdf.add_section_title("Watchlist Technical Analysis - Currency: Thai Baht (THB)")

# Professional data table with improved spacing and styling
with pdf.table(
    col_widths=(16, 14, 14, 14, 14, 12, 14, 12, 12, 32),
    text_align="CENTER",
    line_height=6.5,
    padding=(2, 2)
) as table:
    # Header row styling
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(*pdf.DARK_BLUE)
    
    header_row = table.row()
    headers = [
        "Ticker",
        "Cost (THB)",
        "Price (THB)",
        "Support",
        "Resistance",
        "OBV",
        "MACD",
        "Trend",
        "Action",
        "Technical Notes"
    ]
    
    # Add header cells without 'fill' parameter
    for header_title in headers:
        header_row.cell(header_title)
    
    # Data rows with alternating background colors
    pdf.set_font("Helvetica", "", 8)
    row_index = 0
    
    for stock in analysis_data:
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        obv_status = str(stock.get("obv_status", "")).strip()
        macd_status = str(stock.get("macd_status", "")).strip()
        
        # Alternating row background for readability
        row_fill = row_index % 2 == 0
        if row_fill:
            pdf.set_fill_color(*pdf.LIGHT_GRAY)
        else:
            pdf.set_fill_color(*pdf.WHITE)
        
        # Pull precise programmatic data points
        market_metrics = calculated_market_data.get(ticker, {"latest_price": "N/A", "support": "N/A", "resistance": "N/A"})
        
        # Ticker cell
        pdf.set_text_color(*pdf.DARK_BLUE)
        row.cell(ticker)
        
        # Cost cell
        row.cell(str(stock.get("cost", "")))
        
        # Price cells (numeric data)
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
        
        # OBV Status (colored text)
        if "rising" in obv_status.lower():
            pdf.set_text_color(*pdf.GREEN)
        else:
            pdf.set_text_color(*pdf.RED)
        row.cell(obv_status)
        
        # MACD Status
        pdf.set_text_color(*pdf.DARK_BLUE)
        row.cell(macd_status)
        
        # Trend (colored text)
        if "bullish" in trend_status:
            pdf.set_text_color(*pdf.GREEN)
        elif "bearish" in trend_status:
            pdf.set_text_color(*pdf.RED)
        else:
            pdf.set_text_color(*pdf.ACCENT_GRAY)
        row.cell(str(stock.get("trend", "")))
        
        # Recommendation (colored background)
        pdf.set_text_color(255, 255, 255)
        if "buy" in rec_status:
            pdf.set_fill_color(*pdf.GREEN)
        elif "sell" in rec_status:
            pdf.set_fill_color(*pdf.RED)
        else:
            pdf.set_fill_color(*pdf.YELLOW)
            pdf.set_text_color(*pdf.DARK_BLUE)
        
        row.cell(str(stock.get("recommendation", "")))
        
        # Important Note
        pdf.set_text_color(*pdf.DARK_BLUE)
        if row_fill:
            pdf.set_fill_color(*pdf.LIGHT_GRAY)
        else:
            pdf.set_fill_color(*pdf.WHITE)
        
        note_text = str(stock.get("important_note", ""))
        # Truncate note for table readability
        if len(note_text) > 50:
            note_text = note_text[:47] + "..."
        row.cell(note_text)
        
        row_index += 1

pdf.ln(5)

# Add professional footer section with key insights
pdf.add_section_title("Analysis Summary")

pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(*pdf.ACCENT_GRAY)

summary_text = (
    "This report provides a technical analysis of selected Thai equities listed on the Stock Exchange of Thailand (SET). "
    "The analysis incorporates multiple technical indicators including price support/resistance levels, On-Balance Volume (OBV), "
    "and MACD momentum signals. Recommendations are risk-adjusted and consider both entry costs and reward potential. "
    "All prices and values are denominated in Thai Baht (THB). Investors should combine this technical analysis with "
    "fundamental research and personal risk tolerance before making investment decisions."
)
pdf.multi_cell(0, 5, summary_text, align="J")

# Save the PDF
filename = "thai_market_analysis.pdf"
pdf.output(filename)
print(f"Professional Thai portfolio analysis finalized successfully as {filename}.")
