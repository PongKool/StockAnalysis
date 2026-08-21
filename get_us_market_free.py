import json
import os
from datetime import datetime, timedelta, timezone
from fpdf import FPDF
from fpdf.fonts import FontFace
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import yfinance as yf

# 1. PORTFOLIO & CONFIGURATION
my_costs = {
    "SNDK": 1681.42,
    "ORCL": 143.83,
    "AMZN": 0,
    "NVDA": 209.96,
    "AVGO": 0,
    "SHOP": 0,
    "DIS": 107.38,
    "GOOG": 0,
    "LRCX": 332.64,
    "VRT": 0,
    "MSFT": 0,
    "ISRG": 391.81,
    "NFLX": 80.51,
    "TSM": 425.95,
    "UBER": 78.90,
    "GLW": 160.15,
}
tickers = list(my_costs.keys())
client = genai.Client()


# 2. OUTPUT SCHEMA DEFINITION
class StockAnalysisSchema(BaseModel):
    stock_name: str = Field(description="Ticker symbol.")
    latest_price: str = Field(description="Latest stock close price formatted as 0.00")
    support_level: str = Field(description="Identified key technical support level")
    resistance_level: str = Field(description="Identified key technical resistance level")
    volatility_stop: str = Field(description="Calculated ATR / Volatility trailing stop loss")
    primary_indicators: str = Field(description="Key technical signals observed (e.g. MACD Bullish Crossover, OBV Accumulation, RSI Oversold)")
    trend: str = Field(description="Bullish / Bearish / Sideways")
    news_catalyst: str = Field(description="Top recent news event, earnings driver, or catalyst found via web search (under 12 words)")
    recommendation: str = Field(description="Buy / Hold / Hold (Accumulate) / Sell / Cut Loss / Take Profit")
    important_note: str = Field(description="Synthesized technical and fundamental rationale (strictly under 35 words)")


class StockAnalysisList(BaseModel):
    market_regime: str = Field(description="Macro tech sector trend status (BULLISH, BEARISH, or NEUTRAL)")
    analyses: list[StockAnalysisSchema]


# 3. COLLECT RAW MARKET DATA (NO MANUAL MATH REQUIRED)
print("Gathering raw market price history from Yahoo Finance...")
raw_market_payload = {}

for ticker in ["QQQ"] + tickers:
    try:
        t_obj = yf.Ticker(ticker)
        hist = t_obj.history(period="2mo", auto_adjust=True)
        if hist.empty:
            continue
        
        # Keep clean recent OHLCV data for Gemini to analyze
        recent = hist.tail(25)[["Open", "High", "Low", "Close", "Volume"]].round(2)
        recent.index = recent.index.strftime('%Y-%m-%d')

        # Extract latest real-time news headlines from Yahoo Finance
        raw_news = t_obj.news or []
        news_titles = [
            n.get("content", {}).get("title", n.get("title", ""))
            for n in raw_news[:3]
            if n.get("content", {}).get("title") or n.get("title")
        ]

        raw_market_payload[ticker] = {
            "my_cost": my_costs.get(ticker, 0.0),
            "recent_ohlcv": recent.to_dict(orient="index"),
            "latest_news": news_titles,
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# 4. PROMPT GEMINI TO ANALYZE INDICATORS, SEARCH NEWS, & EVALUATE
prompt = f"""
You are an institutional portfolio manager and senior technical analyst. 
You are provided raw OHLCV price and volume history for the Nasdaq-100 (QQQ) and a watchlist of tech stocks.

YOUR RESPONSIBILITIES:
1. **Macro Regime**: Analyze QQQ's price structure, trend momentum, and determine if the Tech Market Regime is BULLISH, BEARISH, or NEUTRAL.
2. **Technical Indicator Calculation**: Ingest the raw OHLCV data for each stock and determine all key technical indicators (Support, Resistance, ATR/Volatility Stops, RSI, MACD state, and Volume/OBV divergence).
3. **News & Catalyst Assessment**: Synthesize the provided Yahoo Finance news headlines (`latest_news`) into a concise summary of the primary catalyst or sentiment (strictly under 12 words).
4. **Actionable Decision**:
   - Compare current price to the user's cost basis (if cost > 0).
   - If profitable and showing bearish divergence or momentum breakdown, prioritize 'Take Profit' or 'Sell'.
   - If breaking below calculated ATR stop, force 'Cut Loss'.
   - Issue 'Buy' / 'Hold (Accumulate)' on clean breakouts or verified support bounces with positive news catalysts.

Stocks to evaluate: {', '.join(tickers)}
Raw Market Data:
{json.dumps(raw_market_payload, default=str)}
"""

print("Sending data to Gemini with real-time Google Search grounding...")
try:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StockAnalysisList,
            temperature=0.15,
        ),
    )
    result = json.loads(response.text.strip())
    tech_market_regime = result.get("market_regime", "NEUTRAL")
    analysis_data = result.get("analyses", [])
except Exception as e:
    print(f"Analysis generation failed: {e}")
    analysis_data = []
    tech_market_regime = "UNKNOWN"


# 5. GENERATE CLEAN CORPORATE PDF REPORT
class CorporatePDF(FPDF):
    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 297, 4, "F")  # A4 Landscape width
        self.ln(4)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(15, 23, 42)
        self.cell(0, 8, "AI-Driven Market Analysis & Watchlist Intelligence", new_x="LMARGIN", new_y="NEXT")
        
        thailand_tz = timezone(timedelta(hours=7))
        timestamp = datetime.now(thailand_tz).strftime("%Y-%m-%d %H:%M:%S")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f"Generated: {timestamp} (TH Time) | Macro Tech Regime: {tech_market_regime}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y() + 2, 287, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-12)
        self.set_draw_color(241, 245, 249)
        self.line(10, self.get_y(), 287, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


pdf = CorporatePDF(orientation="L", unit="mm", format="A4")
pdf.add_page()

# Total width: 277 mm (A4 landscape fitting)
column_widths = (14, 15, 15, 16, 16, 16, 30, 16, 18, 48, 73)
headers = ["Ticker", "Cost", "Price", "Support", "Resist", "ATR Stop", "Key Indicators", "Trend", "Rec.", "Latest News / Catalyst", "Rationale"]

header_style = FontFace(emphasis="B", color=(255, 255, 255), fill_color=(30, 41, 59), size_pt=7)

with pdf.table(
    col_widths=column_widths, 
    text_align="LEFT", 
    line_height=4.5, 
    padding=2, 
    outer_border_width=0.5,
    headings_style=header_style
) as table:
    header_row = table.row()
    for h in headers:
        header_row.cell(h)

    for idx, item in enumerate(analysis_data):
        row = table.row()
        ticker = item.get("stock_name", "")
        cost_val = my_costs.get(ticker, 0.0)
        cost_str = f"{cost_val:.2f}" if cost_val > 0 else "N/A"
        rec = str(item.get("recommendation", "")).lower()
        trend = str(item.get("trend", "")).lower()
        bg_color = (255, 255, 255) if idx % 2 == 0 else (248, 250, 252)

        # Style for standard data cells with alternating light background
        cell_style = FontFace(size_pt=7, color=(51, 65, 85), fill_color=bg_color)

        row.cell(ticker, style=cell_style)
        row.cell(cost_str, style=cell_style)
        row.cell(str(item.get("latest_price", "")), style=cell_style)
        row.cell(str(item.get("support_level", "")), style=cell_style)
        row.cell(str(item.get("resistance_level", "")), style=cell_style)
        row.cell(str(item.get("volatility_stop", "")), style=cell_style)
        row.cell(str(item.get("primary_indicators", "")), style=cell_style)

        # Dynamic Trend Color Cell
        trend_color = (21, 128, 61) if "bullish" in trend else ((185, 28, 28) if "bearish" in trend else (51, 65, 85))
        row.cell(str(item.get("trend", "")), style=FontFace(size_pt=7, color=trend_color, fill_color=bg_color))

        # Dynamic Recommendation Color Cell
        if "buy" in rec or "accumulate" in rec:
            rec_color = (21, 128, 61)
        elif "sell" in rec or "cut" in rec:
            rec_color = (185, 28, 28)
        else:
            rec_color = (180, 83, 9)
        row.cell(str(item.get("recommendation", "")), style=FontFace(size_pt=7, color=rec_color, fill_color=bg_color))

        # News and Rationale Cells
        row.cell(str(item.get("news_catalyst", "")), style=FontFace(size_pt=6.5, color=(30, 41, 59), fill_color=bg_color))
        row.cell(str(item.get("important_note", "")), style=FontFace(size_pt=6.5, color=(71, 85, 105), fill_color=bg_color))

output_pdf = "us_market_ai_free.pdf"
pdf.output(output_pdf)
print(f"Report finalized and exported to {output_pdf}")
