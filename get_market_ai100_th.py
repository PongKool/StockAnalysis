import os
import json
import yfinance as yf
from google import genai
from google.genai import types
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
import pandas as pd

# Set to True when testing layout/code for $0.00; False for live runs
DRY_RUN = False 

# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION FIRST (THAI SET WATCHLIST)
my_costs = {
    "BBL.BK": 193.32,
    "GULF.BK": 67.41,
    "SCC.BK": 265.11,
    "TU.BK": 13.31,
    "AOT.BK": 0,
    "BCP.BK": 0,
    "BDMS.BK": 20.01,
    "KTB.BK": 44.54,
    "PTT.BK": 41.62,
    "TRUE.BK": 14.09,
    "WHA.BK": 4.87,
    "BH.BK": 0,
    "ADVANC.BK": 355.31,
    "GPSC.BK": 51.59,
    "OR.BK": 12.65,
    "IVL.BK": 25.32,
    "KBANK.BK": 0
}
tickers = list(my_costs.keys())
calculated_market_data = {}

# --- FETCH THAI BLUE-CHIP MACRO REGIME ---
print("Evaluating Thai SET50 Macro Economic Regime...")
macro_regime = "Neutral/Consolidating"
try:
    macro_stock = yf.Ticker("TDEX.BK")
    macro_hist = macro_stock.history(period="6mo", auto_adjust=False)
    macro_hist = macro_hist.dropna(subset=["Close"])
    if len(macro_hist) >= 100:
        macro_sma_100 = macro_hist["Close"].rolling(window=100).mean()
        latest_macro_close = float(macro_hist["Close"].iloc[-1])
        latest_macro_sma = float(macro_sma_100.iloc[-1])
        if latest_macro_close > latest_macro_sma * 1.005:
            macro_regime = "Bullish"
        elif latest_macro_close < latest_macro_sma * 0.995:
            macro_regime = "Bearish/Cautious"
except Exception as e:
    print(f"Warning: Could not calculate Thai macro regime: {e}. Using Neutral/Consolidating.")

print(f"Current Thai Market Regime: {macro_regime}")
print("Fetching technical data in batch from Yahoo Finance for Thai Equities...")

data_summary = ""
try:
    batch_df = yf.download(tickers=tickers, period="6mo", auto_adjust=False, group_by="column", progress=False)
except Exception as e:
    print(f"Error during batch download: {e}")
    batch_df = pd.DataFrame()

# --- DATA PROCESSING LOOP ---
for ticker in tickers:
    support_level = 0.0
    resistance_level = 0.0
    rr_ratio_str = "N/A"
    try:
        if batch_df.empty or ticker not in batch_df['Close']:
            continue
        hist = pd.DataFrame({
            'Open': batch_df['Open'][ticker],
            'High': batch_df['High'][ticker],
            'Low': batch_df['Low'][ticker],
            'Close': batch_df['Close'][ticker],
            'Volume': batch_df['Volume'][ticker]
        }).dropna(subset=['Close'])
        
        if hist.empty or len(hist) < 26:
            continue
            
        latest_close = hist['Close'].iloc[-1]
        
        # Calculate OBV
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        obv_ema14 = obv.ewm(span=14, adjust=False).mean()
        obv_trend = "Rising" if not pd.isna(latest_obv) and not pd.isna(obv_ema14.iloc[-1]) and latest_obv > obv_ema14.iloc[-1] else "Flat/Falling"
        
        # Calculate MACD
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        
        if latest_macd > latest_signal:
            macd_status = "Bullish Crossover" if macd_line.iloc[-2] <= signal_line.iloc[-2] else "Bullish Territory"
        else:
            macd_status = "Bearish Crossover" if macd_line.iloc[-2] >= signal_line.iloc[-2] else "Bearish Territory"
            
        raw_cost = my_costs.get(ticker, 0.0)
        actual_cost = "0.00" if not raw_cost else f"{float(raw_cost):.2f}"
        
        # Volume Profile Support & Resistance
        current_price_val = float(latest_close)
        sma_50 = float(hist['Close'].rolling(window=50).mean().iloc[-1]) if len(hist) >= 50 else float(hist['Close'].mean())
        hist_macro = hist.tail(63).copy()
        price_bins = pd.cut(hist_macro['Close'], bins=10)
        volume_by_bin = hist_macro.groupby(price_bins)['Volume'].sum()
        poc_bin = volume_by_bin.idxmax()
        poc_midpoint = float(poc_bin.mid)
        
        if current_price_val > poc_midpoint:
            support_level = poc_midpoint
            resistance_level = float(sma_50 if sma_50 > current_price_val else hist_macro['High'].max())
        else:
            resistance_level = poc_midpoint
            support_level = float(sma_50 if sma_50 < current_price_val else hist_macro['Low'].min())
            
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close
        if latest_close < support_level:
            rr_ratio_str = "Breakdown"
        elif latest_close > resistance_level:
            rr_ratio_str = "Breakout"
        elif risk_distance <= 0:
            rr_ratio_str = "At Support"
        elif reward_distance <= 0:
            rr_ratio_str = "At Resistance"
        else:
            rr_ratio_str = f"1:{reward_distance / risk_distance:.2f}"
            
        recent_closes = hist.tail(14)
        trend_string = ", ".join([f"{row['Close']:.2f}" for _, row in recent_closes.iterrows()])
        
        calculated_market_data[ticker] = {
            "latest_price": f"{latest_close:.2f}",
            "support": f"{support_level:.2f}",
            "resistance": f"{resistance_level:.2f}"
        }
        
        data_summary += (
            f"Ticker: {ticker} | Cost: {actual_cost} | Close: {latest_close:.2f} | "
            f"Support: {support_level:.2f} | Resistance: {resistance_level:.2f} | R/R: {rr_ratio_str} | "
            f"OBV: {obv_trend} | MACD: {macd_status} | Trend Data: [{trend_string}]\n"
        )
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")

# --- 3. STREAMLINED PURE AI JUDGMENT PROMPT ---
prompt = f"""
You are an expert institutional portfolio manager evaluating Thai SET equities under the macro regime: **{macro_regime}**. 
Review the pre-calculated metrics, volume profiles, technical indicators, and price data below. Use your independent financial judgment to determine optimal recommendations (e.g., Buy, Hold, Hold (Accumulate), Take-Profit Exit, or Stop-Loss).

OUTPUT INSTRUCTIONS:
- Return a valid JSON array of objects matching the exact schema below. Do not wrap in markdown code blocks.
- For `important_note`: Provide a dense, professional technical justification under 45 words in plain text (no markdown, no quotes).

Data Input:
{data_summary}

JSON Schema Required:
[
  {{
    "stock_name": "TICKER",
    "cost": "Exact entry cost provided",
    "obv_status": "Rising/Falling/Flat",
    "macd_status": "Bullish Territory/Bearish Crossover/etc",
    "trend": "Bullish/Bearish/Sideways",
    "recommendation": "Buy/Hold/Hold (Accumulate)/Take-Profit Exit/Stop-Loss",
    "important_note": "Concise technical rationale based on data."
  }}
]
"""

if DRY_RUN:
    print("--- DRY RUN MODE ACTIVE ---")
    analysis_data = [{
        "stock_name": t, 
        "cost": f"{my_costs.get(t, 0.0):.2f}", 
        "obv_status": "Rising", 
        "macd_status": "Bullish Territory", 
        "trend": "Bullish", 
        "recommendation": "Hold", 
        "important_note": "Dry run mock summary data."
    } for t in tickers]
    token_cost_display = "Tokens: In 0 / Out 0 | Cost: $0.000000 (0.00 THB) [DRY RUN]"
else:
    print("Generating structured technical analysis via Gemini API...")
    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=prompt, 
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    raw_json = response.text.strip()
    try:
        analysis_data = json.loads(raw_json)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        analysis_data = [{
            "stock_name": t, "cost": "N/A", "obv_status": "Error", "macd_status": "Error", 
            "trend": "Error", "recommendation": "Error", "important_note": "Failed to parse."
        } for t in tickers]
        
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    cost_usd = ((input_tokens * 0.75) / 1000000) + ((output_tokens * 3.75) / 1000000)
    try:
        thb_ticker = yf.Ticker("THB=X")
        usd_to_thb_rate = thb_ticker.fast_info['last_price']
    except Exception:
        usd_to_thb_rate = 35.00
    cost_thb = cost_usd * usd_to_thb_rate
    token_cost_display = f"Tokens: In {input_tokens:,} / Out {output_tokens:,} | Cost: ${cost_usd:.6f} (~{cost_thb:.2f} THB)"

# --- 4. COMPILE REPORT INTO PDF TABLE LAYOUT ---
class CorporatePDF(FPDF):
    def __init__(self, macro_regime="Bullish", token_cost_str="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macro_regime = macro_regime
        self.token_cost_str = token_cost_str

    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 4, "F")
        self.ln(4)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, "Daily Thai Market Report", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(79, 70, 229)
        self.cell(0, 5, "WATCHLIST TECHNICAL SUMMARY (SET)", new_x="LMARGIN", new_y="NEXT", align="L")
        thailand_tz = timezone(timedelta(hours=7))
        thai_timestamp = datetime.now(thailand_tz).strftime('%Y-%m-%d %H:%M:%S')
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f"Generated automatically on {thai_timestamp} (Thailand Time) | Macro Regime: {self.macro_regime}", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y() + 4, 200, self.get_y() + 4)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(241, 245, 249)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(100, 10, f"Page {self.page_no()}", align="L")
        self.cell(90, 10, self.token_cost_str, align="R")

pdf = CorporatePDF(macro_regime=macro_regime, token_cost_str=token_cost_display)
pdf.add_page()
pdf.set_font("Helvetica", "", 6.5)

col_widths = (11, 7, 7, 8, 8, 8, 8, 8, 10, 25)
with pdf.table(col_widths=col_widths, borders_layout="HORIZONTAL_LINES", line_height=4) as table:
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support", "Resist.", "OBV", "MACD", "Trend", "Rec.", "Important Note (THB Context)"]
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
            
        pdf.set_text_color(51, 65, 85)
        cost_num = my_costs.get(ticker, 0.0)
        display_cost = "-" if not cost_num or float(cost_num) == 0.0 else f"{float(cost_num):.2f}"
        display_ticker = ticker.replace(".BK", "")
        
        row.cell(display_ticker)
        row.cell(display_cost)
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
        
        if "buy" in rec_status or "accumulate" in rec_status:
            pdf.set_text_color(21, 128, 61)
        elif "exit" in rec_status or "loss" in rec_status or "sell" in rec_status:
            pdf.set_text_color(185, 28, 28)
        else:
            pdf.set_text_color(180, 83, 9)
        row.cell(str(stock.get("recommendation", "")))
        
        pdf.set_text_color(71, 85, 105)
        important_note_clean = str(stock.get("important_note", "")).replace("–", "-")
        important_note_clean = important_note_clean.encode('latin-1', 'replace').decode('latin-1')
        row.cell(important_note_clean, align="L")

filename = "thai_market_analysis.pdf"
pdf.output(filename)
print(f"Thai portfolio analysis finalized successfully as {filename}.")
