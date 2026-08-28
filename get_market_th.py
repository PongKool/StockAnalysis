import os
import json
import yfinance as yf
from google import genai
from google.genai import types
from fpdf import FPDF
from datetime import datetime, timezone, timedelta
import pandas as pd

DRY_RUN = False  # Set to True when testing layout/code for $0.00; False for live runs

# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION FIRST (THAI SET WATCHLIST)
my_costs = {
    "BH.BK": 190.69,       # Custom entries configured in THB
    "GULF.BK": 67.25,
    "SCC.BK": 260.44,
    "TU.BK": 13.28,
    "AOT.BK": 0,
    "BCP.BK": 0,
    "BDMS.BK": 20.01,
    "KTB.BK": 44.54,
    "PTT.BK": 0,
    "TRUE.BK": 14.09,
    "WHA.BK": 4.89,
    
    "GPSC.BK": 0,
    "ADVANC.BK": 0,
    "PTTEP.BK": 0,
    "OR.BK": 12.65,
    "IVL.BK": 22.34,
    "KBANK.BK": 0
        
}

tickers = list(my_costs.keys())

# Dictionary to hold the exact calculated numbers for the PDF table mapping
calculated_market_data = {}

# Initialize Gemini Client
client = genai.Client()

# --- FETCH THAI BLUE-CHIP MACRO REGIME ---
print("Evaluating Thai SET50 Macro Economic Regime...")
macro_regime = "Bullish" # Keep this default fallback!
try:
    # TDEX tracks the SET50 Index (Thailand's top 50 blue chips)
    macro_stock = yf.Ticker("TDEX.BK")
    macro_hist = macro_stock.history(period="3mo", auto_adjust=False)
    macro_hist = macro_hist.dropna(subset=['Close'])

    # Calculate 20-Day EMA to determine overall market posture
    macro_ema20 = macro_hist['Close'].ewm(span=20, adjust=False).mean()
    latest_macro_close = macro_hist['Close'].iloc[-1]
    latest_macro_ema = macro_ema20.iloc[-1]

    # Define a tight 0.5% buffer zone around the EMA
    upper_buffer = latest_macro_ema * 1.005
    lower_buffer = latest_macro_ema * 0.995

    # Evaluate the 3 states
    if latest_macro_close > upper_buffer:
        macro_regime = "Bullish"
    elif latest_macro_close < lower_buffer:
        macro_regime = "Bearish/Cautious"
    else:
        macro_regime = "Neutral/Consolidating"

except Exception as e:
    print(f"Warning: Could not calculate Thai macro regime: {e}")
    
print(f"Current Thai Market Regime: {macro_regime}")

print("Fetching technical data in batch from Yahoo Finance for Thai Equities...")
data_summary = ""

# 1. Batch download all tickers in a single network call
try:
    batch_df = yf.download(
        tickers=tickers,
        period="6mo",
        auto_adjust=False,
        group_by="column",
        progress=False
    )
except Exception as e:
    print(f"Error during batch download: {e}")
    batch_df = pd.DataFrame()

# 2. DATA PROCESSING LOOP
for ticker in tickers:
    try:
        # Extract single-ticker slice if available
        if batch_df.empty or ticker not in batch_df['Close']:
            print(f"Warning: No batch data returned for {ticker}")
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

        # --- CALCULATE OBV ---
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        
        # Compare OBV against its 14-day EMA to catch early volume in slow markets
        obv_ema14 = obv.ewm(span=14, adjust=False).mean()
        if pd.isna(latest_obv) or pd.isna(obv_ema14.iloc[-1]):
            obv_trend = "Flat"
        else:
            obv_trend = "Rising" if latest_obv > obv_ema14.iloc[-1] else "Falling"

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

        # Get cost from dictionary; default to 0.0 if not found
        raw_cost = my_costs.get(ticker, 0.0)
        if not raw_cost or raw_cost == 0:
            actual_cost = "0.00"
            position_status = "Watchlist (No Position)"
        else:
            actual_cost = f"{float(raw_cost):.2f}"
            position_status = "Profitable" if latest_close >= float(raw_cost) else "Unprofitable"

        # --- INSTITUTIONAL SUPPORT & RESISTANCE (SMA 50 + Volume Profile) ---
        # 1. Explicitly grab the current close price to prevent NameError crashes
        current_price_val = float(hist['Close'].iloc[-1])
        
        # 2. Safely calculate 50-Day SMA
        if len(hist) >= 50:
            sma_50 = float(hist['Close'].rolling(window=50).mean().iloc[-1])
        else:
            sma_50 = float(hist['Close'].mean())
        
        # 3. Calculate Volume Profile over a 3-month window
        hist_macro = hist.tail(63).copy()
        
        # Safely group by price bins without triggering Pandas version errors
        price_bins = pd.cut(hist_macro['Close'], bins=10)
        volume_by_bin = hist_macro.groupby(price_bins)['Volume'].sum()
        
        poc_bin = volume_by_bin.idxmax()
        poc_midpoint = float(poc_bin.mid)
        
        # 4. Assign Support and Resistance 
        if current_price_val > poc_midpoint:
            support_level = poc_midpoint
            resistance_level = float(sma_50 if sma_50 > current_price_val else hist_macro['High'].max())
        else:
            resistance_level = poc_midpoint
            support_level = float(sma_50 if sma_50 < current_price_val else hist_macro['Low'].min())
            
        # --- DYNAMIC BUFFER BASED ON MACRO REGIME ---
        if macro_regime == "Bullish":
            support_buffer = support_level * 0.025  # Wider buffer (buy earlier)
        elif macro_regime == "Bearish/Cautious":
            support_buffer = support_level * 0.005  # Tighter buffer (demand strict defense)
        else:
            support_buffer = support_level * 0.015  # Standard 1.5% buffer

        # --- CALCULATE RISK/REWARD RATIO ---
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close

        if latest_close < support_level:
            rr_ratio_str = "Breakdown"
        elif latest_close > resistance_level:
            rr_ratio_str = "Breakout (Above Resistance)" if macro_regime != "Bearish/Cautious" else "Fakeout Risk"
        elif risk_distance <= support_buffer:
            if macro_regime == "Bullish":
                rr_ratio_str = "High-Conviction Bounce"
            elif macro_regime == "Bearish/Cautious":
                rr_ratio_str = "High Risk (Catching Knives)"
            else:
                rr_ratio_str = "Testing Support (Bounce Potential)"
        elif reward_distance <= 0:
            rr_ratio_str = "Poor (At Resistance)"
        elif risk_distance <= 0:
            rr_ratio_str = "At Support"
        else:
            calculated_ratio = reward_distance / risk_distance
            rr_ratio_str = f"1:{calculated_ratio:.2f}"

        recent_closes = hist.tail(14)
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
            f"Position Status: {position_status} | "
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

1. **CONFIRMED BREAKOUT (PRICE ABOVE RESISTANCE):**
   - Condition: Latest Close is strictly breaking out ABOVE 1Mo Resistance (`Latest Close > 1Mo Resistance`), accompanied by a "Rising" OBV trend and a "Bullish" MACD profile:
     - For **Watchlist Stocks** (`Entry Cost: 0.00`): Recommend **Buy (Breakout)** to enter expanding volume momentum.
     - For **Held Positions** (`Entry Cost > 0`): Recommend **Hold (Accumulate)** to aggressively scale into the confirmed trend. Override standard "Poor R/R" exit rules.

2. **TESTING RESISTANCE / EXTENDED MOVE (PRICE AT OR BELOW RESISTANCE):**
   - Condition: Price is near or testing resistance (`Latest Close <= 1Mo Resistance`) with an unfavorable Risk/Reward ratio (< 1:1.0):
     - For **Held Positions (`Entry Cost > 0`)**:
       - **If Profitable (`Latest Close >= Entry Cost`)**:
         - If showing exhaustion (MACD bearish cross or falling OBV), recommend **Take-Profit Exit**.
         - If momentum remains intact (Rising OBV + Bullish MACD), recommend **Hold** (ride gains, do not add new risk at ceiling).
       - **If Unprofitable (`Latest Close < Entry Cost`)**:
         - Recommend **Hold** (maintain position for recovery/breakout without adding new capital near resistance, unless structural support is broken).
     - For **Watchlist Stocks (`Entry Cost: 0.00`)**: Recommend **Hold** (or **Watch**) to wait for either a confirmed breakout above resistance or a pullback to support.

3. **WATCHLIST / NEW POSITION ENTRY (`Entry Cost: 0.00`):**
   - **Buy (Initiate Position):** Recommend **Buy** if OBV is "Rising" with a healthy MACD profile ("Bullish Territory" or fresh "Bullish Crossover") AND Risk/Reward is favorable (near support or upside ratio > 1:1.5).
   - **Neutral / Wait:** Recommend **Hold** (or **Watch**) if conditions are unmet (e.g., Bearish MACD, Falling OBV). Never issue exit/stop-loss recommendations for unentered assets.

4. **HELD POSITIONS: DEFENSE & SUPPORT ACCUMULATION (`Entry Cost > 0`):**
   - **Hold (Accumulate):** If position is healthy with "Rising" OBV + "Bullish" MACD while resting near **1Mo Support** (favorable R/R), recommend **Hold (Accumulate)**.
   - **Support-Aware Stop-Loss:** If position is unprofitable AND Latest Close breaks structurally below 1Mo Support, recommend **Support-Aware Stop-Loss**. If holding support with bounce potential, maintain **Hold**.

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

if DRY_RUN:
    print("--- DRY RUN MODE ACTIVE: Skipping paid Gemini API call ---")
    analysis_data = [
        {
            "stock_name": t,
            "cost": f"{my_costs.get(t, 0.0):.2f}" if my_costs.get(t, 0.0) > 0 else "0.00",
            "obv_status": "Rising",
            "macd_status": "Bullish Territory",
            "trend": "Bullish",
            "recommendation": "Buy" if my_costs.get(t, 0.0) == 0 else "Hold (Accumulate)",
            "important_note": "Mock validation test: Checking table layout and cost alignment."
        }
        for t in tickers
    ]
    token_cost_display = "Tokens: In 0 / Out 0 | Cost: $0.000000 (0.00 THB) [DRY RUN]"
else:
    print("Generating structured technical analysis via Gemini API...")
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
    )

    raw_json = response.text.strip()
    if raw_json.startswith("```"):
        lines = raw_json.splitlines()
        if lines[0].startswith("```"): lines = lines[1:]
        if lines[-1].startswith("```"): lines = lines[:-1]
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
            }
            for t in tickers
        ]

    # --- 3. Calculate LLM Token Costs ---
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

# 4. COMPILE REPORT INTO PDF TABLE LAYOUT
class CorporatePDF(FPDF):
    def __init__(self, macro_regime="Bullish", token_cost_str="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macro_regime = macro_regime
        self.token_cost_str = token_cost_str

    def header(self):
        # Top decorative primary accent bar
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
        self.set_text_color(100, 116, 139) # Muted Slate
        self.cell(0, 5, f"Generated automatically on {thai_timestamp} (Thailand Time) | Macro Regime: {self.macro_regime}", new_x="LMARGIN", new_y="NEXT", align="L")
        
        # Subtle divider line under header
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y() + 4, 200, self.get_y() + 4)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(241, 245, 249)
        self.line(10, self.get_y(), 200, self.get_y())
        
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        
        # Display Page Number on the left and Token Cost info on the right
        self.cell(100, 10, f"Page {self.page_no()}", align="L")
        self.cell(90, 10, self.token_cost_str, align="R")

pdf = CorporatePDF(macro_regime=macro_regime, token_cost_str=token_cost_display)
pdf.add_page()

# Setup Table Styles (Exactly 10 Columns adding up to 190mm printable width)
pdf.set_font("Helvetica", "", 6.5)
col_widths = (11, 7, 7, 8, 8, 8, 8, 8, 10, 25)

# Removed invalid 'cell_alignment' kwarg to resolve fpdf.py structural initializer crash
with pdf.table(col_widths=col_widths, borders_layout="HORIZONTAL_LINES", line_height=4) as table:
    
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
        cost_num = my_costs.get(ticker, 0.0)
        
        if not cost_num or float(cost_num) == 0.0:
            display_cost = "-"
        else:
            display_cost = f"{float(cost_num):.2f}"

        display_ticker = ticker.replace(".BK", "")
        row.cell(display_ticker)
        row.cell(display_cost)
        
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
        row.cell(important_note_clean, align="L")

filename = "thai_market_analysis.pdf"
pdf.output(filename)
print(f"Thai portfolio analysis finalized successfully as {filename}.")
