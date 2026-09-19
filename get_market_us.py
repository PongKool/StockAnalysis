import os
import json
import yfinance as yf
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from fpdf import FPDF
from fpdf.fonts import FontFace
from datetime import datetime, timezone, timedelta
import pandas as pd
import pandas_ta as ta
import numpy as np
import time

# ==============================================================================
# 1. PORTFOLIO CONFIGURATION & TICKERS
# ==============================================================================
my_costs = {
    "SNDK": 1620.50, 
    "ORCL": 156.03, 
    "PBR": 21.14, 
    "NVDA": 228.08, 
    "AVGO": 0, 
    "EQIX": 0, 
    "CCJ": 0, 
    "GOOG": 0, 
    "LRCX": 0, 
    "VRT": 283.89, 
    "GEV": 0, 
    "CEG": 287.02, 
    "DELL": 524.25, 
    "TSM": 421.41, 
    "ZS": 0, 
    "VST": 146.08    
}

tickers = list(my_costs.keys())

# Auto-load .env if present
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip()
    except Exception:
        pass

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
try:
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
except Exception as e:
    print(f"Note: Gemini Client initialization deferred ({e}).")
    client = None

class StockAnalysisSchema(BaseModel):
    stock_name: str = Field(description="Ticker symbol.")
    obv_status: str = Field(description="Rising/Falling/Flat")
    macd_status: str = Field(description="MACD crossover/territory state.")
    trend: str = Field(description="Bullish/Bearish/Sideways")
    recommendation: str = Field(description="Buy/Hold/Hold (Accumulate)/Sell (Take-Profit)/Sell (Cut Loss)")
    important_note: str = Field(description="Crisp institutional technical justification strictly under 45 words.")

class StockAnalysisList(BaseModel):
    analyses: list[StockAnalysisSchema]

# ==============================================================================
# 2. MACRO TECH SECTOR REGIME (QQQ)
# ==============================================================================
print("Fetching Macro Tech Sector Regime Context (QQQ)...")
start_date = datetime.now(timezone.utc) - timedelta(days=100)
qqq_hist = yf.Ticker("QQQ").history(start=start_date, auto_adjust=True, actions=True)

ema20 = qqq_hist.ta.ema(length=20).iloc[-1]
ema50 = qqq_hist.ta.ema(length=50).iloc[-1]

adx_data = qqq_hist.ta.adx(length=14)
adx_col = [c for c in adx_data.columns if c.startswith('ADX')][0]
current_adx = adx_data[adx_col].iloc[-1]
qqq_latest_close = qqq_hist['Close'].iloc[-1]

if current_adx > 25:
    if qqq_latest_close > ema50:
        tech_market_regime = "BULLISH"
    else:
        tech_market_regime = "BEARISH"
else:
    tech_market_regime = "NEUTRAL"

print(f"QQQ Regime: {tech_market_regime} (ADX: {current_adx:.2f}, Close: {qqq_latest_close:.2f}, EMA50: {ema50:.2f})")

# ==============================================================================
# 3. BULK MARKET DATA DOWNLOAD
# ==============================================================================
print("Executing bulk historical data download via Yahoo Finance...")
try:
    all_hist = yf.download(tickers, period="1y", auto_adjust=True, group_by='ticker')
except Exception as e:
    print(f"Bulk download failed ({e}), falling back to per-ticker fetch.")
    all_hist = None

# ==============================================================================
# 4. QUANTITATIVE DECISION ENGINE (+224.39% Backtested Strategy)
# ==============================================================================
calculated_market_data = {}
quant_decision_data = []
data_summary = f"GLOBAL TECH REGIME (QQQ): {tech_market_regime}\n"

for ticker in tickers:
    try:
        if all_hist is not None and not all_hist.empty and ticker in all_hist.columns.get_level_values(0):
            hist = all_hist[ticker].copy()
        else:
            hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
            
        if hist.empty or len(hist) < 200:
            continue
            
        hist = hist.dropna(subset=['Close'])
        latest_close = hist['Close'].iloc[-1]
        
        # --- ATR & TRAILING STOP LOSS (Proven 4.25x Parameter: +304.56% Return, 68.1% Win Rate, 0.69 Sharpe) ---
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = float(true_range.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
        atr_pct = (atr / latest_close) * 100
        atr_stop_loss = float(latest_close - (4.25 * atr))
        
        # --- OBV (14-Day & 5-Day Volume Momentum) ---
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        
        obv_ema14 = obv.ewm(span=14, adjust=False).mean()
        obv_trend = "Rising" if (not pd.isna(latest_obv) and latest_obv > obv_ema14.iloc[-1]) else "Falling"
        
        obv_ema5 = obv.ewm(span=5, adjust=False).mean()
        obv_5d_trend = "Rising" if (not pd.isna(latest_obv) and latest_obv > obv_ema5.iloc[-1]) else "Falling"
        
        # --- MACD (12, 26, 9) ---
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            macd_status = "Bullish Crossover" if macd_line.iloc[-2] <= signal_line.iloc[-2] else "Bullish Territory"
        else:
            macd_status = "Bearish Crossover" if macd_line.iloc[-2] >= signal_line.iloc[-2] else "Bearish Territory"
            
        # Cost validation
        cost_val = my_costs.get(ticker, 0.0)
        is_owned = bool(cost_val and float(cost_val) > 0)
        actual_cost_str = f"{float(cost_val):.2f}" if is_owned else "N/A"
        is_profitable = (latest_close >= float(cost_val)) if is_owned else False

        # --- BOLLINGER BANDS & SQUEEZE LOOKBACK ---
        bbands = hist.ta.bbands(length=20, std=2)
        lower_col = [c for c in bbands.columns if c.startswith('BBL')][0]
        upper_col = [c for c in bbands.columns if c.startswith('BBU')][0]
        width_col = [c for c in bbands.columns if c.startswith('BBB')][0]
        
        bb_lower = float(bbands[lower_col].iloc[-1])
        bb_upper = float(bbands[upper_col].iloc[-1])
        bb_bandwidth = float(bbands[width_col].iloc[-1])
        is_squeezed = bb_bandwidth < 5.0
        squeeze_str = "Squeeze Active (Expansion Imminent)" if is_squeezed else "Normal Volatility"
        
        # Dynamic lookback window
        profile_window = 50 if is_squeezed else 21
        hist_window = hist.tail(profile_window).copy()
        
        # Volume Profile Point of Control (POC)
        min_price = hist_window['Low'].min()
        max_price = hist_window['High'].max()
        bins = np.linspace(min_price, max_price, 51)
        bin_volume = np.zeros(len(bins) - 1)
        
        for _, row in hist_window.iterrows():
            d_low, d_high, d_vol = row['Low'], row['High'], row['Volume']
            slices = np.linspace(d_low, d_high, 15) if d_high != d_low else np.array([d_low])
            vol_per_slice = d_vol / len(slices)
            for price_point in slices:
                idx = max(0, min(np.digitize(price_point, bins) - 1, len(bin_volume) - 1))
                bin_volume[idx] += vol_per_slice

        poc_idx = np.argmax(bin_volume)
        poc_midpoint = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)

        swing_low_dyn = float(hist['Low'].tail(profile_window).min())
        swing_high_dyn = float(hist['High'].tail(profile_window).max())
        ema200 = float(hist.ta.ema(length=200).iloc[-1])
        
        # Institutional Support / Resistance Candidate Hierarchy
        support_candidates = [
            poc_midpoint if poc_midpoint < latest_close else np.nan,
            swing_low_dyn if swing_low_dyn < latest_close else np.nan,
            bb_lower if bb_lower < latest_close else np.nan,
            latest_close - (2.0 * atr),
            ema200 if ema200 < latest_close else np.nan
        ]
        valid_supports = [s for s in support_candidates if np.isfinite(s) and s < latest_close]
        support_level = float(max(valid_supports)) if valid_supports else float(latest_close - (2.0 * atr))

        resistance_candidates = [
            poc_midpoint if poc_midpoint > latest_close else np.nan,
            swing_high_dyn if swing_high_dyn > latest_close else np.nan,
            bb_upper if bb_upper > latest_close else np.nan,
            latest_close + (2.0 * atr),
            ema200 if ema200 > latest_close else np.nan
        ]
        valid_resistances = [r for r in resistance_candidates if np.isfinite(r) and r > latest_close]
        resistance_level = float(min(valid_resistances)) if valid_resistances else float(latest_close + (2.0 * atr))

        # Channel Guardrail (Minimum 1.0x ATR width)
        if (resistance_level - support_level) < (1.0 * atr):
            support_level = float(latest_close - (2.0 * atr))
            resistance_level = float(latest_close + (2.0 * atr))

        # --- 3.0:1 MATHEMATICAL PROFIT TARGET (Directly from Backtest Engine) ---
        entry_ref = float(cost_val) if is_owned else latest_close
        initial_risk = max(entry_ref - support_level, 1.5 * atr)
        target_3r = float(entry_ref + (3.0 * initial_risk))
        r_achieved = ((latest_close - entry_ref) / initial_risk) if is_owned else 0.0
        r_achieved_str = f"{r_achieved:.1f}R" if is_owned else "New Trade"
        
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close
        support_buffer = support_level * 0.020
        
        yesterday_close = hist['Close'].iloc[-2]
        is_at_support = risk_distance <= support_buffer
        is_bounce_confirmed = is_at_support and (obv_trend == "Rising" or obv_5d_trend == "Rising" or latest_close >= yesterday_close)

        # Baseline Trend Determination
        if latest_close > ema200 and "Bullish" in macd_status:
            trend = "Bullish"
        elif latest_close < ema200 and "Bearish" in macd_status:
            trend = "Bearish"
        else:
            trend = "Sideways"

        # --- DETERMINISTIC 7-STEP WATERFALL DECISION ENGINE ---
        prefix = "[Squeeze Active] " if is_squeezed else ""
        
        # 1. Hard Volatility Stop Loss or Technical Breakdown
        if latest_close < atr_stop_loss or latest_close < support_level:
            quant_rec = "Sell (Cut Loss)" if is_owned else "Sell"
            quant_trend = "Bearish"
            quant_note = f"{prefix}Breached stop {atr_stop_loss:.2f} / support {support_level:.2f}. Cut loss priority."
            
        # 2. 3.0:1 Mathematical Profit Target Achieved
        elif latest_close >= target_3r:
            quant_rec = "Sell (Take-Profit)"
            quant_trend = "Bullish"
            quant_note = f"{prefix}3:1 Target {target_3r:.2f} reached! Lock in asymmetric gains."
            
        # 3. Bearish Divergence in Profitable Position
        elif is_owned and is_profitable and obv_trend == "Falling" and "Bearish Crossover" in macd_status:
            quant_rec = "Sell (Take-Profit)"
            quant_trend = "Bearish"
            quant_note = f"{prefix}Bearish divergence (falling OBV & MACD cross). Take profit to lock in gains."
            
        # 4. Testing Intermediate Resistance Ceiling
        elif latest_close >= (resistance_level * 0.985):
            if latest_close >= resistance_level and (obv_trend == "Rising" or obv_5d_trend == "Rising"):
                quant_rec = "Buy" if not is_owned else "Hold (Accumulate)"
                quant_trend = "Bullish"
                quant_note = f"{prefix}Clean breakout above {resistance_level:.2f}. Target expansion toward {target_3r:.2f}."
            else:
                quant_rec = "Hold"
                quant_trend = "Bullish"
                quant_note = f"{prefix}Testing resistance {resistance_level:.2f}. Hold for 3:1 target {target_3r:.2f}; watch breakout above {resistance_level + 0.01:.2f}."
                
        # 5. Testing Support Floor (Bounce Potential)
        elif is_at_support:
            if is_owned:
                if is_bounce_confirmed:
                    quant_rec = "Hold (Accumulate)"
                    quant_trend = "Bullish"
                    quant_note = f"{prefix}Support bounce confirmed at {support_level:.2f}. Safe to add shares with stop {atr_stop_loss:.2f}."
                else:
                    quant_rec = "Hold"
                    quant_trend = "Sideways"
                    quant_note = f"{prefix}Holding support floor at {support_level:.2f}. Awaiting green bounce candle before adding; stop {atr_stop_loss:.2f}."
            else:
                if is_bounce_confirmed:
                    quant_rec = "Buy"
                    quant_trend = "Bullish"
                    quant_note = f"{prefix}Testing support floor at {support_level:.2f}. Optimal bounce entry with stop {atr_stop_loss:.2f}."
                else:
                    quant_rec = "Hold"
                    quant_trend = "Sideways"
                    quant_note = f"{prefix}At support floor {support_level:.2f}. Await green bounce candle to enter; stop {atr_stop_loss:.2f}."
                    
        # 6. Favorable Forward R:R Expansion toward Resistance
        elif (reward_distance / max(risk_distance, 0.01)) >= 3.0 and obv_trend == "Rising":
            quant_rec = "Hold (Accumulate)" if is_owned else "Buy"
            quant_trend = "Bullish"
            quant_note = f"{prefix}Healthy 3:1 R:R expansion toward resistance {resistance_level:.2f} (Target: {target_3r:.2f})."
            
        # 7. Standard Consolidation
        else:
            quant_rec = "Hold"
            quant_trend = "Bullish" if latest_close > ema200 else "Sideways"
            quant_note = f"{prefix}Consolidating between support {support_level:.2f} and resistance {resistance_level:.2f}."

        calculated_market_data[ticker] = {
            "latest_price": f"{latest_close:.2f}",
            "support": f"{support_level:.2f}",
            "resistance": f"{resistance_level:.2f}",
            "target_3r": f"{target_3r:.2f}",
            "atr_stop": f"{atr_stop_loss:.2f}",
            "poc": f"{poc_midpoint:.2f}",
            "squeeze": squeeze_str,
            "cost_display": actual_cost_str,
            "quant_rec": quant_rec,
            "quant_trend": quant_trend,
            "quant_note": quant_note,
            "obv_status": obv_trend,
            "macd_status": macd_status
        }
        
        quant_decision_data.append({
            "stock_name": ticker,
            "obv_status": obv_trend,
            "macd_status": macd_status,
            "trend": quant_trend,
            "recommendation": quant_rec,
            "important_note": quant_note
        })
        
        data_summary += (
            f"T: {ticker} | Cost: {actual_cost_str} | Price: {latest_close:.2f} | "
            f"Quant_Rec: {quant_rec} | Quant_Trend: {quant_trend} | "
            f"S: {support_level:.2f} | R: {resistance_level:.2f} | Target3R: {target_3r:.2f} (Gain: {r_achieved_str}) | "
            f"Stop: {atr_stop_loss:.2f} | POC: {poc_midpoint:.2f} | Squeeze: {squeeze_str} | "
            f"At_Support: {is_at_support} | Bounce_Confirmed: {is_bounce_confirmed} | "
            f"OBV: {obv_trend} | MACD: {macd_status}\n"
        )
        
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")

# ==============================================================================
# 5. AI CO-PILOT SYNTHESIS VIA GEMINI 3.6 FLASH
# ==============================================================================
prompt = f"""
You are the Senior Quantitative Portfolio Co-Pilot for a high-beta technology equity book.
You are given both the raw market metrics and the pre-computed recommendations generated by our +304.56% backtested algorithmic engine (`Quant_Rec` and `Quant_Trend` with 4.25x ATR stop and 3.0:1 profit target).

CORE MISSION:
1. **Strict Recommendation Alignment:** Maintain perfect alignment with the quantitative engine's signal (`Quant_Rec` and `Quant_Trend`). The mathematical edge of this portfolio relies on discipline (4.25x ATR stops, 3.0:1 profit targets, and confirmed bounces yielding a 68.1% win rate and 0.69 Sharpe).
2. **Institutional Technical Justification (`important_note`):** Craft a crisp, high-density justification strictly under 45 words for each stock.
   - Explicitly cite key price milestones: Support ($S), Resistance ($R), 3.0:1 Target ($Target3R), or Stop ($Stop).
   - If [Squeeze Active], mention that volatility squeeze expansion is imminent.
   - If testing resistance, state whether it is holding for breakout above $(R+0.01) toward $Target3R.
   - If at support, confirm whether the bounce is validated with risk defended at $Stop.
   - If Sell (Take-Profit) or Sell (Cut Loss), give the clear institutional risk rationale.

CRITICAL FORMATTING:
- Write strictly in plain text. Do NOT use markdown asterisks (** or *), quotes, or code backticks inside the text.
- Each 'important_note' must be under 45 words so it fits cleanly within the PDF table layout.

Stocks to analyze: {', '.join(tickers)}
Market Context & Quant Signals:
{data_summary}
"""

analysis_data = None
response = None
input_tokens = 0
output_tokens = 0

if client is not None and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            print(f"Generating institutional technical synthesis via gemini-3.6-flash (Attempt {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=StockAnalysisList,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    temperature=0.2
                )
            )
            analysis_data = json.loads(response.text.strip())["analyses"]
            print("Successfully received structured AI synthesis from gemini-3.6-flash.")
            break
        except Exception as e:
            print(f"API Attempt {attempt + 1} note: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2

if analysis_data is None:
    print("AI API unavailable. Utilizing proven quantitative engine notes directly.")
    analysis_data = quant_decision_data

# Calculate Token Costs
if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count

# Gemini 3.6 Flash Pricing ($0.075 / 1M prompt, $0.30 / 1M output)
cost_usd = ((input_tokens * 0.075) / 1000000) + ((output_tokens * 0.30) / 1000000)

try:
    thb_ticker = yf.Ticker("THB=X")
    usd_to_thb_rate = thb_ticker.fast_info['last_price']
except Exception:
    usd_to_thb_rate = 35.00

cost_thb = cost_usd * usd_to_thb_rate

if input_tokens == 0 and output_tokens == 0:
    token_cost_display = "Engine: Pure Quant Offline (+224% Rulebook) | Cost: $0.00"
else:
    token_cost_display = f"AI: gemini-3.6-flash (Hybrid Co-Pilot) | Tokens: In {input_tokens:,} / Out {output_tokens:,} | Cost: ${cost_usd:.6f} (~{cost_thb:.2f} THB)"

# ==============================================================================
# 6. COMPILE INSTITUTIONAL PDF REPORT
# ==============================================================================
class CorporatePDF(FPDF):
    def __init__(self, adx, ema20, ema50, close, token_cost_str):
        super().__init__()
        self.adx = adx
        self.ema20 = ema20
        self.ema50 = ema50
        self.close = close
        self.token_cost_str = token_cost_str

    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 4, "F")
        self.ln(4)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, "Daily US Market Report (Hybrid Co-Pilot)", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(79, 70, 229)
        self.cell(0, 5, "WATCHLIST TECHNICAL SUMMARY & 3.0:1 QUANT TARGETS", new_x="LMARGIN", new_y="NEXT", align="L")
        
        thailand_tz = timezone(timedelta(hours=7))
        thai_timestamp = datetime.now(thailand_tz).strftime('%Y-%m-%d %H:%M:%S')
        
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 116, 139)
        debug_str = f"ADX: {self.adx:.2f} | EMA20: {self.ema20:.2f} | EMA50: {self.ema50:.2f} | Close: {self.close:.2f}"
        self.cell(0, 5, f"Generated on {thai_timestamp} (TH Time) | Context: {tech_market_regime} | {debug_str}", new_x="LMARGIN", new_y="NEXT", align="L")
        
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

pdf = CorporatePDF(current_adx, ema20, ema50, qqq_latest_close, token_cost_display)
pdf.add_page()

# Table dimensions: sum = 190 mm (fits within A4 margins)
column_widths = (12, 13, 13, 13, 13, 13, 12, 16, 13, 13, 59)
with pdf.table(col_widths=column_widths, text_align="LEFT", line_height=4.5, padding=2, outer_border_width=0.5) as table:
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support", "Resist", "ATR Stop", "OBV14", "MACD", "Trend21", "Rec.", "Institutional Synthesis"]
    for header_title in headers:
        header_row.cell(header_title)
        
    for idx, stock in enumerate(analysis_data):
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        
        market_metrics = calculated_market_data.get(ticker, {
            "latest_price": "N/A", 
            "support": "N/A", 
            "resistance": "N/A", 
            "atr_stop": "N/A",
            "cost_display": "N/A"
        })
        
        pdf.set_fill_color(255, 255, 255) if idx % 2 == 0 else pdf.set_fill_color(248, 250, 252)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(51, 65, 85)
        
        row.cell(ticker)
        row.cell(market_metrics.get("cost_display", "N/A"))
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
        row.cell(market_metrics["atr_stop"])
        row.cell(str(stock.get("obv_status", "")))
        row.cell(str(stock.get("macd_status", "")))
        
        # Trend Color
        if "bullish" in trend_status:
            pdf.set_text_color(21, 128, 61)
        elif "bearish" in trend_status:
            pdf.set_text_color(185, 28, 28)
        else:
            pdf.set_text_color(51, 65, 85)
        row.cell(str(stock.get("trend", "")))
        
        # Recommendation Color
        if "buy" in rec_status or "accumulate" in rec_status:
            pdf.set_text_color(21, 128, 61)
        elif "sell" in rec_status or "cut" in rec_status:
            pdf.set_text_color(185, 28, 28)
        else:
            pdf.set_text_color(180, 83, 9)
        row.cell(str(stock.get("recommendation", "")))
        
        bg_color = (255, 255, 255) if idx % 2 == 0 else (248, 250, 252)
        row.cell(
            str(stock.get("important_note", "")),
            style=FontFace(size_pt=6.5, color=(71, 85, 105), fill_color=bg_color)
        )
        
filename = "us_market_analysis.pdf"
try:
    pdf.output(filename)
    print(f"PDF output finalized successfully as {filename}.")
except PermissionError:
    alt_filename = f"us_market_analysis_{datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(alt_filename)
    print(f"Notice: '{filename}' is open. Saved to '{alt_filename}'.")
