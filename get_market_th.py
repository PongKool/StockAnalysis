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
import numpy as np
import time

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

DRY_RUN = False  # Set to True when testing layout/code for $0.00; False for live runs

# Strategy Constants: Option 2 Hybrid & Relative Strength Momentum (+81.54% Engine)
ATR_STOP_MULT = 3.0
TARGET_RR = 3.5
SUPPORT_BUFFER_PCT = 0.030

# ==============================================================================
# 1. PORTFOLIO CONFIGURATION & THAI SET WATCHLIST
# ==============================================================================
my_costs = {
    # Official Top 15 Market Cap Equities on SET
    "DELTA.BK": 0,
    "PTT.BK": 41.62,
    "ADVANC.BK": 355.31,
    "GULF.BK": 62.50,
    "AOT.BK": 0,
    "KTB.BK": 44.55,
    "KBANK.BK": 253.22,
    "PTTEP.BK": 0,
    "SCB.BK": 153.51,
    "TRUE.BK": 0,
    "CPALL.BK": 0,
    "BBL.BK": 193.83,
    "BDMS.BK": 0,
    "SCC.BK": 0,
    "CRC.BK": 0,
    # User's Additional Holdings & Watchlist
    "BEM.BK": 6.92,
    "TU.BK": 13.33,
    "WHA.BK": 4.87,
    "BCP.BK": 0,
    "BH.BK": 0,
    "GPSC.BK": 48.97,
    "IVL.BK": 0
}

tickers = list(my_costs.keys())

# Pydantic Schema for Gemini Structured Output
class StockAnalysisSchema(BaseModel):
    stock_name: str = Field(description="Ticker symbol.")
    rank: str = Field(description="Leaderboard rank, e.g. '#1', '#2', '#3', '#4', or '-'")
    cost: str = Field(description="The exact entry cost provided or '-'")
    obv_status: str = Field(description="Rising/Falling/Flat")
    macd_status: str = Field(description="MACD crossover/territory state.")
    trend: str = Field(description="Bullish/Bearish/Sideways")
    recommendation: str = Field(description="Buy (Leader Pullback)/Buy (Leader Breakout)/Buy (Support Bounce)/Hold (Core Leader)/Hold/Hold (Accumulate)/Sell (Take-Profit)/Sell (Cut Loss)/Watch")
    important_note: str = Field(description="Crisp institutional justification strictly under 35 words citing levels.")

class StockAnalysisList(BaseModel):
    analyses: list[StockAnalysisSchema]

# ==============================================================================
# 2. THAI BLUE-CHIP MACRO REGIME (TDEX.BK)
# ==============================================================================
print("Evaluating Thai SET50 Macro Economic Regime (TDEX.BK)...")
macro_regime = "Neutral/Consolidating"

try:
    macro_stock = yf.Ticker("TDEX.BK")
    macro_hist = macro_stock.history(period="6mo", auto_adjust=False)
    macro_hist = macro_hist.dropna(subset=["Close"])

    if len(macro_hist) >= 50:
        macro_sma_100 = macro_hist["Close"].rolling(window=min(100, len(macro_hist))).mean()
        latest_macro_close = float(macro_hist["Close"].iloc[-1])
        latest_macro_sma = float(macro_sma_100.iloc[-1])

        upper_buffer = latest_macro_sma * 1.005
        lower_buffer = latest_macro_sma * 0.995

        if latest_macro_close > upper_buffer:
            macro_regime = "Bullish"
        elif latest_macro_close < lower_buffer:
            macro_regime = "Bearish/Cautious"
        else:
            macro_regime = "Neutral/Consolidating"
except Exception as e:
    print(f"Warning: Could not calculate Thai macro regime: {e}. Using Neutral/Consolidating.")

print(f"Current Thai Market Regime: {macro_regime}")

# ==============================================================================
# 3. BULK MARKET DATA DOWNLOAD
# ==============================================================================
print("Fetching technical data in batch from Yahoo Finance for Thai Equities...")
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

# ==============================================================================
# 4. QUANTITATIVE DECISION ENGINE (+51.71% Backtested Strategy, Sharpe 2.08)
# ==============================================================================
calculated_market_data = {}
quant_decision_data = []
data_summary = f"THAI MACRO REGIME (TDEX): {macro_regime}\n"

# ==============================================================================
# 4. RELATIVE STRENGTH MOMENTUM LEADERBOARD (Top Leaders +81.54% Backtested Engine)
# ==============================================================================
momentum_scores = {}
for t in tickers:
    if not batch_df.empty and t in batch_df['Close']:
        s_c = batch_df['Close'][t].dropna()
        if len(s_c) >= 20:
            c_now = float(s_c.iloc[-1])
            c_30 = float(s_c.iloc[-min(30, len(s_c))])
            c_60 = float(s_c.iloc[-min(60, len(s_c))])
            roc_30 = ((c_now - c_30) / c_30) * 100
            roc_60 = ((c_now - c_60) / c_60) * 100 if len(s_c) >= 60 else roc_30
            sma_50 = float(s_c.rolling(50).mean().iloc[-1]) if len(s_c) >= 50 else float(s_c.mean())
            is_above_sma50 = (c_now >= sma_50 * 0.98)
            score = (0.6 * roc_30) + (0.4 * roc_60)
            momentum_scores[t] = {
                "score": score, "roc_30": roc_30, "roc_60": roc_60,
                "above_sma50": is_above_sma50
            }

eligible_leaders = [t for t in momentum_scores if momentum_scores[t]["above_sma50"]]
ranked_leaders_list = sorted(eligible_leaders, key=lambda t: momentum_scores[t]["score"], reverse=True)
top_4_leaders = {t: idx + 1 for idx, t in enumerate(ranked_leaders_list[:4])}
print(f"Top 4 SET Relative Strength Leaders Identified: {list(top_4_leaders.keys())}")

for ticker in tickers:
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

        latest_close = float(hist['Close'].iloc[-1])
        latest_high = float(hist['High'].iloc[-1])
        latest_low = float(hist['Low'].iloc[-1])

        # 1. ATR 14
        h_l = hist['High'] - hist['Low']
        h_c = (hist['High'] - hist['Close'].shift(1)).abs()
        l_c = (hist['Low'] - hist['Close'].shift(1)).abs()
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
        atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])

        # 2. OBV (14-day EMA Volume Trend)
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        obv_ema14 = obv.ewm(span=14, adjust=False).mean().iloc[-1]
        obv_trend = "Rising" if latest_obv > obv_ema14 else "Falling"

        # 3. MACD (12, 26, 9)
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_sig = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_sig = signal_line.iloc[-2]

        if latest_macd > latest_sig:
            macd_status = "Bullish Crossover" if prev_macd <= prev_sig else "Bullish Territory"
        else:
            macd_status = "Bearish Crossover" if prev_macd >= prev_sig else "Bearish Territory"

        # 4. SMA 50
        sma_50 = float(hist['Close'].rolling(window=50).mean().iloc[-1]) if len(hist) >= 50 else float(hist['Close'].mean())

        # 5. Volume Profile POC (63-day window)
        hist_macro = hist.tail(min(63, len(hist))).copy()
        try:
            price_bins = pd.cut(hist_macro['Close'], bins=10)
            volume_by_bin = hist_macro.groupby(price_bins, observed=False)['Volume'].sum()
            poc_bin = volume_by_bin.idxmax()
            poc_midpoint = float(poc_bin.mid)
        except Exception:
            poc_midpoint = latest_close

        # 6. Support & Resistance
        if latest_close > poc_midpoint:
            support_level = poc_midpoint
            resistance_level = float(sma_50 if sma_50 > latest_close else hist_macro['High'].max())
        else:
            resistance_level = poc_midpoint
            support_level = float(sma_50 if sma_50 < latest_close else hist_macro['Low'].min())

        if (resistance_level - support_level) < (1.0 * atr):
            support_level = latest_close - (1.5 * atr)
            resistance_level = latest_close + (1.5 * atr)

        # 7. Stop Loss & 3.0:1 Target Price
        entry_cost_num = float(my_costs.get(ticker, 0.0))
        is_held = (entry_cost_num > 0.0)
        display_cost = f"{entry_cost_num:.2f}" if is_held else "-"

        if is_held:
            # For held positions: Stop loss is anchored to entry cost (exact backtest rule)
            atr_stop_level = entry_cost_num - (ATR_STOP_MULT * atr)
            initial_risk = max(entry_cost_num - atr_stop_level, atr * 1.5)
            target_price = entry_cost_num + (TARGET_RR * initial_risk)
            is_stop_breached = (latest_close <= atr_stop_level) or (latest_close < (support_level * 0.98))
        else:
            # For new watchlist candidates: Forward stop from current price
            atr_stop_level = latest_close - (ATR_STOP_MULT * atr)
            initial_risk = max(latest_close - support_level, atr * 1.5)
            target_price = latest_close + (TARGET_RR * initial_risk)
            is_stop_breached = latest_close < (support_level * 0.98)

        # 8. Overall Trend Evaluation & Momentum Status
        bullish_macd = "Bullish" in macd_status
        rising_obv = (obv_trend == "Rising")
        above_sma50 = (latest_close >= sma_50)
        ema_20 = float(hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
        is_top_leader = (ticker in top_4_leaders)
        rank_display = f"#{top_4_leaders[ticker]}" if is_top_leader else "-"
        m_score = momentum_scores.get(ticker, {}).get("score", 0.0)

        if bullish_macd and rising_obv and above_sma50:
            quant_trend = "Bullish"
        elif not bullish_macd and not rising_obv and not above_sma50:
            quant_trend = "Bearish"
        else:
            quant_trend = "Sideways"

        # 9. DETERMINISTIC 8-STEP WATERFALL DECISION TREE (+81.54% DUAL ENGINE)
        is_breakout = (latest_close > resistance_level)
        support_buffer = support_level * (1.0 + SUPPORT_BUFFER_PCT)
        is_at_support = (latest_close >= support_level * 0.985) and (latest_close <= support_buffer)

        if is_held:
            recent_high_20d = float(hist['High'].tail(min(20, len(hist))).max())
            pnl_pct_held = ((latest_close - entry_cost_num) / entry_cost_num) * 100 if entry_cost_num > 0 else 0.0
            peak_gain_pct = ((recent_high_20d - entry_cost_num) / entry_cost_num) * 100 if entry_cost_num > 0 else 0.0
            pullback_from_peak = ((latest_close - recent_high_20d) / recent_high_20d) * 100 if recent_high_20d > 0 else 0.0

            # Dynamic Profit-Protection Trailing Stop (Option 2: activated if peak gain >= 12.0%)
            trailing_stop_level = 0.0
            is_trailing_stop_breached = False
            if peak_gain_pct >= 12.0:
                trailing_stop_level = max(recent_high_20d - (2.5 * atr), entry_cost_num * 1.03)
                is_trailing_stop_breached = (latest_close <= trailing_stop_level)

            # Tactical Take-Profit Alert Condition: In profit >= 10% and testing resistance or pulling back with drying volume
            is_tactical_tp_alert = (
                pnl_pct_held >= 10.0 and 
                (latest_close >= resistance_level * 0.985 or pullback_from_peak <= -4.0 or not rising_obv)
            )

            # 1. Hard Volatility Stop Loss / Support Breakdown (Below Cost)
            if is_stop_breached:
                quant_rec = "Sell (Cut Loss)"
                if is_at_support and rising_obv:
                    quant_note = f"Breached stop ({atr_stop_level:.2f} THB). Staging bounce off {support_level:.2f} support with rising OBV; exit on relief rally or hard stop below support."
                else:
                    quant_note = f"Breached {ATR_STOP_MULT}x ATR stop ({atr_stop_level:.2f} THB) from cost {entry_cost_num:.2f}. Capital defense."

            # 2. 3.5:1 Profit Target Hit or Resistance Exhaustion
            elif latest_high >= target_price or (latest_close >= resistance_level and not rising_obv and pnl_pct_held >= 15.0):
                quant_rec = "Sell (Take-Profit)"
                quant_note = f"Reached {TARGET_RR}:1 target ({target_price:.2f} THB). Locking in profit at resistance."

            # 3. Dynamic Profit-Protection Trailing Stop Triggered
            elif is_trailing_stop_breached:
                quant_rec = "Sell (Take-Profit)"
                quant_note = f"Breached trailing stop ({trailing_stop_level:.2f} THB) after {peak_gain_pct:.1f}% peak. Securing {pnl_pct_held:+.1f}% profit."

            # 4. Tactical Take-Profit Alert (Option 2 Hybrid Mode)
            elif is_tactical_tp_alert:
                display_tp_stop = trailing_stop_level if trailing_stop_level > 0 else (recent_high_20d - 1.5 * atr)
                quant_rec = "Sell (Take-Profit)"
                quant_note = f"Tactical TP Alert: In +{pnl_pct_held:.1f}% profit near resistance ({resistance_level:.2f} THB). Bank profit now or trail stop at {display_tp_stop:.2f} THB."

            # 5. Core Leader Status (Highest Conviction Run)
            elif is_top_leader:
                quant_rec = "Hold (Core Leader)"
                stop_display = trailing_stop_level if (trailing_stop_level > 0 and latest_close > entry_cost_num) else atr_stop_level
                quant_note = f"SET Rank {rank_display} Momentum Leader ({m_score:.1f}% ROC). Ride core trend toward {target_price:.2f} THB. Stop: {stop_display:.2f} THB."

            # 6. Bounce at Support (Scale candidate)
            elif is_at_support and rising_obv and bullish_macd:
                quant_rec = "Hold (Accumulate)"
                quant_note = f"Bouncing off POC support ({support_level:.2f} THB) with volume accumulation. Scale candidate."

            # 7. Standard Holding Toward 3.5:1 Target
            else:
                quant_rec = "Hold"
                stop_display = trailing_stop_level if (trailing_stop_level > 0 and latest_close > entry_cost_num) else atr_stop_level
                quant_note = f"Trend intact. Holding toward {TARGET_RR}:1 target ({target_price:.2f} THB). Stop: {stop_display:.2f} THB."
        else:
            # Watchlist / New Entry
            if is_top_leader:
                pullback_to_ema20 = (latest_close <= ema_20 * 1.025) and (latest_close >= ema_20 * 0.975)
                if pullback_to_ema20:
                    quant_rec = "Buy (Leader Pullback)"
                    quant_note = f"SET Rank {rank_display} Momentum Leader testing 20 EMA ({ema_20:.2f} THB). Low-risk trend entry toward {target_price:.2f} THB."
                elif is_breakout:
                    quant_rec = "Buy (Leader Breakout)"
                    quant_note = f"SET Rank {rank_display} Momentum Leader breakout above {resistance_level:.2f} THB. Momentum ({m_score:.1f}% ROC). Target: {target_price:.2f} THB."
                elif bullish_macd and rising_obv:
                    quant_rec = "Buy (Leader Breakout)"
                    quant_note = f"SET Rank {rank_display} Momentum Leader holding high momentum ({m_score:.1f}% ROC). Target: {target_price:.2f} THB."
                else:
                    quant_rec = "Watch"
                    quant_note = f"SET Rank {rank_display} Momentum Leader consolidating. Waiting for pullback to 20 EMA ({ema_20:.2f} THB) or breakout."
            elif is_breakout and rising_obv and bullish_macd:
                quant_rec = "Buy (Breakout)"
                quant_note = f"Confirmed breakout above {resistance_level:.2f} THB with rising volume. Target: {target_price:.2f} THB."
            elif is_at_support and rising_obv and bullish_macd:
                quant_rec = "Buy (Support Bounce)"
                quant_note = f"High-conviction bounce at POC support ({support_level:.2f} THB). R/R > {TARGET_RR}:1. Target: {target_price:.2f} THB."
            else:
                quant_rec = "Watch"
                quant_note = f"Patience. Waiting for support test at {support_level:.2f} THB or confirmed breakout above {resistance_level:.2f} THB."

        # Display active stop (trailing stop if in profit protection mode, otherwise entry stop)
        display_stop = trailing_stop_level if (is_held and trailing_stop_level > 0 and latest_close > entry_cost_num) else atr_stop_level

        calculated_market_data[ticker] = {
            "rank": rank_display,
            "latest_price": f"{latest_close:.2f}",
            "support": f"{support_level:.2f}",
            "resistance": f"{resistance_level:.2f}",
            "target_3r": f"{target_price:.2f}",
            "atr_stop": f"{display_stop:.2f}",
            "display_cost": display_cost
        }

        quant_decision_data.append({
            "stock_name": ticker,
            "rank": rank_display,
            "cost": display_cost,
            "obv_status": obv_trend,
            "macd_status": macd_status,
            "trend": quant_trend,
            "recommendation": quant_rec,
            "important_note": quant_note
        })

        reversal_alert = "Yes (Bounce at Support with Rising OBV)" if (is_at_support and rising_obv) else "No"
        pnl_str = f"{pnl_pct_held:+.1f}%" if is_held else "N/A"
        data_summary += (
            f"T: {ticker} | Rank: {rank_display} | Cost: {display_cost} | Price: {latest_close:.2f} | PnL: {pnl_str} | "
            f"Quant_Rec: {quant_rec} | Quant_Trend: {quant_trend} | "
            f"Support: {support_level:.2f} | Resist: {resistance_level:.2f} | "
            f"Target3R: {target_price:.2f} | Stop: {display_stop:.2f} | "
            f"OBV: {obv_trend} | MACD: {macd_status} | Reversal_Attempt: {reversal_alert}\n"
        )
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")

# ==============================================================================
# 5. AI CO-PILOT SYNTHESIS VIA GEMINI 3.6 FLASH
# ==============================================================================
prompt = f"""
You are the Senior Quantitative Portfolio Co-Pilot for premier equities on the Stock Exchange of Thailand (SET). All figures in THB.
You are given both the raw market metrics and the pre-computed recommendations generated by our Dual-Engine quantitative system (+81.54% Backtested Return, 1.96 Sharpe Ratio): Relative Strength Momentum Leaderboard + Option 2 Tactical Controls.

CORE MISSION:
1. **Strict Recommendation Alignment & Rank Preservation:** Maintain perfect alignment with the quantitative engine's signal (`Quant_Rec`, `Quant_Trend`, and `Rank`). If Rank is '#1', '#2', '#3', '#4', preserve it exactly; otherwise use '-'. Do not alter `Quant_Rec`.
2. **Institutional Technical Justification (`important_note`):** Craft a crisp, high-density justification strictly under 35 words for each stock.
   - For Top 4 Leaders (`#1` to `#4`): Highlight their relative strength, 20 EMA pullback defense, or momentum breakout toward targets.
   - Explicitly cite key price milestones: Support, Resistance, 3.5:1 Target, Trailing Stop, or 3.0x ATR Stop in THB.
   - If testing resistance, state whether it is holding for breakout toward the 3.5:1 target.
   - If at support, confirm whether the bounce is validated with risk defended at the 3.0x ATR stop.
   - If Sell (Take-Profit):
     * If 3.5:1 target reached or resistance hit, cite target hit or resistance exhaustion.
     * If Tactical TP Alert or Trailing Stop triggered, cite locking in double-digit profit or securing gains at the elevated trailing stop.
   - If Sell (Cut Loss):
     * If broken down with falling volume/trend (e.g. TRUE), cite capital defense and stop breach.
     * If attempting an oversold bounce/reversal off support with rising OBV (e.g. GULF), acknowledge the support bounce and advise a tactical exit into the relief rally toward resistance with an emergency stop placed just below the support level.

CRITICAL FORMATTING:
- Write strictly in plain text. Do NOT use markdown asterisks (** or *), quotes, or code backticks inside the text.
- Each 'important_note' must be under 35 words so it fits cleanly within the PDF table layout.

Stocks to analyze: {', '.join(tickers)}
Market Context & Quant Signals:
{data_summary}
"""

analysis_data = None
response = None
input_tokens = 0
output_tokens = 0

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
try:
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
except Exception as e:
    client = None

if not DRY_RUN and client is not None and api_key:
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
    print("AI API unavailable or DRY_RUN active. Utilizing proven quantitative engine notes directly.")
    analysis_data = quant_decision_data

# Calculate Token Costs
if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count

cost_usd = ((input_tokens * 0.075) / 1000000) + ((output_tokens * 0.30) / 1000000)

try:
    thb_ticker = yf.Ticker("THB=X")
    usd_to_thb_rate = thb_ticker.fast_info['last_price']
except Exception:
    usd_to_thb_rate = 35.00

cost_thb = cost_usd * usd_to_thb_rate

if input_tokens == 0 and output_tokens == 0:
    token_cost_display = "Engine: Pure Quant Offline (+81.5% Rulebook) | Cost: 0.00 THB"
else:
    token_cost_display = f"AI: gemini-3.6-flash (Hybrid Co-Pilot) | Tokens: In {input_tokens:,} / Out {output_tokens:,} | Cost: ${cost_usd:.6f} (~{cost_thb:.2f} THB)"

# ==============================================================================
# 6. COMPILE INSTITUTIONAL PDF REPORT
# ==============================================================================
class CorporatePDF(FPDF):
    def __init__(self, macro_regime="Bullish", token_cost_str="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macro_regime = macro_regime
        self.token_cost_str = token_cost_str

    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 4, "F")
        self.ln(4)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(15, 23, 42)
        self.cell(0, 8, "Daily Thai Market Report (Hybrid Co-Pilot)", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(79, 70, 229)
        self.cell(0, 5, "RELATIVE STRENGTH MOMENTUM LEADERBOARD & 3.5:1 QUANT TARGETS (SET)", new_x="LMARGIN", new_y="NEXT", align="L")
        
        thailand_tz = timezone(timedelta(hours=7))
        thai_timestamp = datetime.now(thailand_tz).strftime('%Y-%m-%d %H:%M:%S')
        
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f"Generated automatically on {thai_timestamp} (Thailand Time) | SET50 Regime: {self.macro_regime} | Strategy: Top 4 Leaders + Option 2 (+81.5% Backtested)", new_x="LMARGIN", new_y="NEXT", align="L")
        
        self.set_draw_color(226, 232, 240)
        self.line(10, self.get_y() + 3, 200, self.get_y() + 3)
        self.ln(8)

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
# 11 Columns totaling 190mm
col_widths = (11, 7, 7, 7, 8, 8, 8, 12, 8, 14, 100)

with pdf.table(col_widths=col_widths, borders_layout="HORIZONTAL_LINES", line_height=4) as table:
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    
    header_row = table.row()
    headers = ["Ticker", "Rank", "Cost", "Price", "Support", "Resist.", "OBV", "MACD", "Trend", "Rec.", "Institutional Synthesis (THB)"]
    for header_title in headers:
        header_row.cell(header_title)

    for idx, stock in enumerate(analysis_data):
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        
        market_metrics = calculated_market_data.get(ticker, {"latest_price": "N/A", "support": "N/A", "resistance": "N/A", "display_cost": "-", "rank": "-"})
        rank_val = str(stock.get("rank", market_metrics.get("rank", "-"))).strip()
        
        if idx % 2 == 0:
            pdf.set_fill_color(255, 255, 255)
        else:
            pdf.set_fill_color(248, 250, 252)
            
        pdf.set_text_color(51, 65, 85)
        
        display_ticker = ticker.replace(".BK", "")
        row.cell(display_ticker)
        
        # Rank Column Accent
        if rank_val.startswith("#"):
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(79, 70, 229)  # Indigo
        else:
            pdf.set_font("Helvetica", "", 6.5)
            pdf.set_text_color(148, 163, 184)
        row.cell(rank_val)
        
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(51, 65, 85)
        row.cell(market_metrics.get("display_cost", "-"))
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
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
try:
    pdf.output(filename)
    print(f"Thai portfolio analysis finalized successfully as {filename}.")
except PermissionError:
    alt_filename = f"thai_market_analysis_{datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(alt_filename)
    print(f"Notice: '{filename}' is open. Saved to '{alt_filename}'.")
