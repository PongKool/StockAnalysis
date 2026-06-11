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
tickers = ["MU", "NVDA", "ORCL", "SNDK", "MSFT", "TSM", "LLY", "LRCX", "NOW", "AMD", "CACI", "AVGO", "ANET"]
my_costs = {
    "MU": 424.62, "NVDA": 220.80, "ORCL": 183.72, "SNDK": 1418.17, 
    "MSFT": 455.37, "TSM": 424.30, "LLY": 971.12, "LRCX": 305.41, 
    "NOW": 107.68, "AMD": 448.37, "CACI": 524.53, "AVGO": 446.13, "ANET": 171.11
}

calculated_market_data = {}
client = genai.Client()

# Strict schema to guarantee flawless, clean JSON mapping from the Gemini API
class StockAnalysisSchema(BaseModel):
    stock_name: str = Field(description="The ticker symbol of the stock.")
    cost: str = Field(description="The exact entry cost value provided in data input.")
    obv_status: str = Field(description="OBV trend direction, e.g., Rising / Falling")
    macd_status: str = Field(description="Current MACD state, e.g., Bullish Territory / Bearish Crossover")
    trend: str = Field(description="Overall direction: Bullish, Bearish, or Sideways")
    recommendation: str = Field(description="Actionable decision: Buy, Hold, or Sell")
    important_note: str = Field(description="Technical reason proving why momentum changes justify a Hold or Profit-Take Sell.")

print("Fetching technical data from Yahoo Finance...")
data_summary = ""

# 2. DATA GATHERING LOOP
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        # auto_adjust=True fixes dividend/split gaps and includes the live trading day price bar
        hist = stock.history(period="3mo", auto_adjust=True)
        if hist.empty or len(hist) < 26:
            continue

        hist = hist.dropna(subset=['Close'])
        latest_close = hist['Close'].iloc[-1]  # Safely represents the latest available market price

        # --- CALCULATE OBV ---
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        # Shifted diff() to run before tail() to ensure NO NaN corruption occurs
        obv_trend = "Rising" if obv.diff().tail(5).mean() > 0 else "Falling"

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
            rr_ratio_str = f"1:{(reward_distance / risk_distance):.2f}"

        recent_closes = hist_1m.tail(14)
        trend_string = ", ".join([f"{row['Close']:.2f}" for _, row in recent_closes.iterrows()])

        calculated_market_data[ticker] = {
            "latest_price": f"{latest_close:.2f}",
            "support": f"{support_level:.2f}",
            "resistance": f"{resistance_level:.2f}"
        }

        data_summary += (
            f"Ticker: {ticker} | Entry Cost: {actual_cost} | Latest Close: {latest_close:.2f} | "
            f"Is Position Profitable?: {is_profitable} | 1Mo Support: {support_level:.2f} | 1Mo Resistance: {resistance_level:.2f} | "
            f"Risk/Reward: {rr_ratio_str} | OBV: {latest_obv:.0f} ({obv_trend}) | "
            f"MACD: {latest_macd:.2f} (Signal: {latest_signal:.2f}, {macd_status}) | Recent Close Trend: [{trend_string}]\n"
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
            "obv_status": "Error",
            "macd_status": "Error",
            "trend": "Error",
            "recommendation": "Error",
            "important_note": "System tracking failure. Could not structure data payload safely."
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
        self.cell(0, 5, f"Generated automatically on {thai_timestamp} (Thailand Time)", new_x="LMARGIN", new_y="NEXT", align="L")
        
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

column_widths = (14, 13, 13, 13, 13, 12, 23, 16, 12, 61)

with pdf.table(col_widths=column_widths, text_align="LEFT", line_height=6, padding=1.5, outer_border_width=0.5) as table:
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support", "Resist.", "OBV", "MACD", "Trend", "Rec.", "Important Note"]
    for header_title in headers:
        header_row.cell(header_title)

    for idx, stock in enumerate(analysis_data):
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        market_metrics = calculated_market_data.get(ticker, {"latest_price": "N/A", "support": "N/A", "resistance": "N/A"})

        if idx % 2 == 0:
            pdf.set_fill_color(255, 255, 255)
        else:
            pdf.set_fill_color(248, 250, 252)

        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(51, 65, 85)

        row.cell(ticker)
        row.cell(str(stock.get("cost", "")))
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
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

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(71, 85, 105)
        row.cell(str(stock.get("important_note", "")))

filename = "morning_market_analysis.pdf"
pdf.output(filename)
print(f"PDF output finalized successfully as {filename}.")
