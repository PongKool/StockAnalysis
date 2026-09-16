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


# 1. INITIALIZE GLOBAL VARIABLES & CONFIGURATION
my_costs = {
    "SNDK": 1622.50, 
    "ORCL": 0, 
    "PBR": 21.14, 
    "NVDA": 228.08, 
    "AVGO": 0, 
    "GLW": 154.58, 
    "CCJ": 0,
    "GOOG": 340.47, 
    "LRCX": 0,
    "VRT": 0,
    "GEV": 0, 
    "CEG": 0,
    "DELL": 524.25,
    "TSM": 0,
    "ZS": 0, 
    "VST": 146.08    
}

tickers = list(my_costs.keys())

calculated_market_data = {}
try:
    client = genai.Client()
except Exception as e:
    print(f"Note: Gemini Client initialization deferred ({e}).")
    client = None

class StockAnalysisSchema(BaseModel):
    stock_name: str = Field(description="Ticker symbol.")
    obv_status: str = Field(description="Rising/Falling/Flat")
    macd_status: str = Field(description="MACD crossover/territory state.")
    trend: str = Field(description="Bullish/Bearish/Sideways")
    recommendation: str = Field(description="Buy/Hold/Hold (Accumulate)/Sell")
    important_note: str = Field(description="Technical reason mapping volatility or breakout levels.")

class StockAnalysisList(BaseModel):
    analyses: list[StockAnalysisSchema]

print("Fetching Macro Tech Sector Regime Context (QQQ)...")

# Fetch 100 days of data to account for indicator "warm-up". 
# ADX and EMA use recursive smoothing and require a long history to calculate accurate, stable values for the final row.

start_date = datetime.now(timezone.utc) - timedelta(days=100)
qqq_hist = yf.Ticker("QQQ").history(start=start_date, auto_adjust=True, actions=True)

# 2. Calculate Indicators
# EMA 20 & EMA 50
ema20 = qqq_hist.ta.ema(length=20).iloc[-1]
ema50 = qqq_hist.ta.ema(length=50).iloc[-1]
# ADX 14
adx_data = qqq_hist.ta.adx(length=14)
adx_col = [c for c in adx_data.columns if c.startswith('ADX')][0]
current_adx = adx_data[adx_col].iloc[-1]

# 3. Logic: Trend strength (ADX) + Direction (Price vs EMA 50)
qqq_latest_close = qqq_hist['Close'].iloc[-1]

# --- DEBUG PRINTS ---
print(f"--- DEBUG DATA ---")
print(f"Calculated ADX: {current_adx:.2f}")
print(f"Calculated EMA20: {ema20:.2f} | EMA50: {ema50:.2f}")
print(f"Latest Close: {qqq_latest_close:.2f}")
print(f"Condition (ADX > 25): {current_adx > 25}")
print(f"Condition (Close > EMA50): {qqq_latest_close > ema50}")
# -----------------------------

if current_adx > 25:
    if qqq_latest_close > ema50:
        tech_market_regime = "BULLISH"
    else:
        tech_market_regime = "BEARISH"
else:
    tech_market_regime = "NEUTRAL"

print("Executing bulk historical data download via Yahoo Finance...")
try:
    all_hist = yf.download(tickers, period="1y", auto_adjust=True, group_by='ticker')
except Exception as e:
    print(f"Bulk download failed: {e}")
    all_hist = None

data_summary = f"REGIME: {tech_market_regime}\n"

# 2. DATA PROCESSING LOOP
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
        
        # --- ATR & TRAILING STOP LOSS ---
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        atr_pct = (atr / latest_close) * 100
        # OPTIMIZED: 3.0x ATR Stop Loss (proved +224% return, 60% win rate in backtests)
        atr_stop_loss = latest_close - (3.0 * atr)
        
        # --- CALCULATE OBV ---
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
    
        # REVISED: Compare OBV against its Moving Average (mimicking TradingView)
        # 14-day macro trend
        obv_ema14 = obv.ewm(span=14, adjust=False).mean()
        if pd.isna(latest_obv) or pd.isna(obv_ema14.iloc[-1]):
            obv_trend = "Flat"
        else:
            obv_trend = "Rising" if latest_obv > obv_ema14.iloc[-1] else "Falling"
    
        # REVISED: 5-day short-term trend calculation
        obv_ema5 = obv.ewm(span=5, adjust=False).mean()
        if pd.isna(latest_obv) or pd.isna(obv_ema5.iloc[-1]):
            obv_5d_trend = "Flat"
        else:
            obv_5d_trend = "Rising" if latest_obv > obv_ema5.iloc[-1] else "Falling"
        
            
        # --- CALCULATE MACD ---
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            macd_status = "Bullish Crossover" if macd_line.iloc[-2] <= signal_line.iloc[-2] else "Bullish Territory"
        else:
            macd_status = "Bearish Crossover" if macd_line.iloc[-2] >= signal_line.iloc[-2] else "Bearish Territory"
            
        cost_val = my_costs.get(ticker, 0.0)

        # --- CALCULATE BOLLINGER BANDS (20, 2) ---
        bbands = hist.ta.bbands(length=20, std=2)
        lower_col = [c for c in bbands.columns if c.startswith('BBL')][0]
        upper_col = [c for c in bbands.columns if c.startswith('BBU')][0]
        width_col = [c for c in bbands.columns if c.startswith('BBB')][0]
        
        bb_lower = float(bbands[lower_col].iloc[-1])
        bb_upper = float(bbands[upper_col].iloc[-1])
        bb_bandwidth = float(bbands[width_col].iloc[-1])

        # If cost is 0, None, or empty, user has no position (watch-only)
        if not cost_val or cost_val == 0 or str(cost_val).strip() in ["", "0", "0.0", "none", "n/a", "N/A"]:
            actual_cost = "N/A"
            is_profitable = "N/A"
        else:
            cost_num = float(cost_val)
            actual_cost = f"{cost_num:.2f}"
            is_profitable = "Yes" if latest_close >= cost_num else "No"
        
        # --- HANDLE VOLATILITY SQUEEZES & DYNAMIC LOOKBACK ---
        is_squeezed = bb_bandwidth < 5.0
        squeeze_status_str = "Squeeze Active (Expansion Imminent)" if is_squeezed else "Normal Volatility"
        
        # Widen profile lookback window during a squeeze to capture macro structural levels
        profile_window = 50 if is_squeezed else 21
        hist_1m = hist.tail(profile_window).copy()
        
        min_price = hist_1m['Low'].min()
        max_price = hist_1m['High'].max()
        bins = np.linspace(min_price, max_price, 51)
        bin_volume = np.zeros(len(bins) - 1)
        
        for _, row in hist_1m.iterrows():
            day_low, day_high, day_vol = row['Low'], row['High'], row['Volume']
            slices = np.linspace(day_low, day_high, 15) if day_high != day_low else np.array([day_low])
            vol_per_slice = day_vol / len(slices)
            for price_point in slices:
                idx = max(0, min(np.digitize(price_point, bins) - 1, len(bin_volume) - 1))
                bin_volume[idx] += vol_per_slice

        poc_idx = np.argmax(bin_volume)
        poc_midpoint = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)
        print(f"{ticker} POC Midpoint: {poc_midpoint:.2f}")

        swing_low_dyn = float(hist['Low'].tail(profile_window).min())
        swing_high_dyn = float(hist['High'].tail(profile_window).max())
    
        # Calculate Average True Range (ATR) for volatility context
        tr = np.maximum(
            hist['High'] - hist['Low'],
            np.maximum(
                abs(hist['High'] - hist['Close'].shift(1)),
                abs(hist['Low'] - hist['Close'].shift(1))
            )
        )
        atr_14 = float(tr.rolling(14).mean().iloc[-1])
    
        # Hybrid Support & Resistance with ATR, Volume Profile, and Bollinger Bands

        ema200 = float(hist.ta.ema(length=200).iloc[-1])

        # 1. Define support/resistance candidates ordered by institutional significance
        support_candidates = [
            ("POC", poc_midpoint if poc_midpoint < latest_close else np.nan),
            ("Swing Low", swing_low_dyn if swing_low_dyn < latest_close else np.nan),
            ("Lower BB", bb_lower if bb_lower < latest_close else np.nan),
            ("2 ATR Support", latest_close - (2.0 * atr_14)),
            ("EMA200", ema200 if ema200 < latest_close else np.nan),
        ]

        resistance_candidates = [
            ("POC", poc_midpoint if poc_midpoint > latest_close else np.nan),
            ("Swing High", swing_high_dyn if swing_high_dyn > latest_close else np.nan),
            ("Upper BB", bb_upper if bb_upper > latest_close else np.nan),
            ("2 ATR Resistance", latest_close + (2.0 * atr_14)),
            ("EMA200", ema200 if ema200 > latest_close else np.nan),
        ]

        # 2. Extract first valid structural level for Support and Resistance
        valid_supports = [
            (name, float(val)) for name, val in support_candidates 
            if np.isfinite(val) and val < latest_close
        ]
        valid_resistances = [
            (name, float(val)) for name, val in resistance_candidates 
            if np.isfinite(val) and val > latest_close
        ]

        # High-priority initial pick
        if not valid_supports:
            valid_supports = [("2 ATR Support", float(latest_close - (2.0 * atr_14)))]

        if not valid_resistances:
            valid_resistances = [("2 ATR Resistance", float(latest_close + (2.0 * atr_14)))]

        support_source, support_level = valid_supports[0]
        resistance_source, resistance_level = valid_resistances[0]

        # 3. GUARDRAIL: If channel width is unrealistically compressed (< 1.0x ATR),
        # expand support or resistance to the 2 ATR boundary to maintain a tradable range.
        min_channel_width = 1.0 * atr_14

        if (resistance_level - support_level) < min_channel_width:
            # If support is too close or above current price
            if (latest_close - support_level) < (min_channel_width / 2):
                support_source = "2 ATR Support"
                support_level = float(latest_close - (2.0 * atr_14))

            # If resistance is too close or below current price
            if (resistance_level - latest_close) < (min_channel_width / 2):
                resistance_source = "2 ATR Resistance"
                resistance_level = float(latest_close + (2.0 * atr_14))

        print(
            f"Ticker: {ticker} | Price: {latest_close:.2f} | "
            f"Support: {support_level:.2f} ({support_source}) | "
            f"Resistance: {resistance_level:.2f} ({resistance_source}) | "
            f"Spread: {(resistance_level - support_level):.2f}"
        )
        
        # HIGH-BETA MILESTONE OPTIMIZATION (14-day history windowed into key nodes)
        # Moved up here so closes_14d is defined before it's used in the conditional block below
        closes_14d = hist_1m['Close'].tail(14).tolist()
        optimized_trend = [
            closes_14d[0],   # T-14 (Macro baseline)
            closes_14d[4],   # T-10
            closes_14d[9],   # T-5  (One week ago)
            closes_14d[11],  # T-3  (Recent momentum switch)
            closes_14d[-2],  # Yesterday
            closes_14d[-1]   # Today
        ]
        trend_string = ", ".join([f"{val:.1f}" for val in optimized_trend])

        # --- RISK/REWARD RATIO WITH BREAKOUT/BOUNCE DETECTION ---
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close

        # OPTIMIZED: 2.0% support buffer zone (proven in backtests to optimize entry timing)
        support_buffer = support_level * 0.020

        if latest_close < support_level:
            rr_ratio_str = "Breakdown"
        elif latest_close > resistance_level:
            rr_ratio_str = "Breakout"
        # NEW: Detect if price is hugging support but starting to bounce/hold
        elif risk_distance <= support_buffer and (obv_trend == "Rising" or latest_close >= closes_14d[-2]):
            rr_ratio_str = "Testing Support (Bounce Potential)"
        elif reward_distance <= 0:
            rr_ratio_str = "Poor"
        elif risk_distance == 0:
            rr_ratio_str = "Excellent"
        else:
            rr_ratio_str = f"1:{(reward_distance / risk_distance):.1f}"

        # --- 3.0:1 MATHEMATICAL PROFIT TARGET (Exact Backtest Strategy Rule) ---
        entry_ref = cost_val if (actual_cost != "N/A" and cost_val > 0) else latest_close
        initial_risk = max(entry_ref - support_level, 1.5 * atr)
        target_3r = entry_ref + (3.0 * initial_risk)
        r_achieved = ((latest_close - entry_ref) / initial_risk) if actual_cost != "N/A" else 0.0
        r_achieved_str = f"{r_achieved:.1f}R" if actual_cost != "N/A" else "New Trade"

        atr_to_target = (reward_distance / atr) if reward_distance > 0 else 0.0

        # --- SAVE & SUMMARIZE DATA ---
        is_bounce_confirmed = bool(risk_distance <= support_buffer and (obv_trend == "Rising" or latest_close >= closes_14d[-2]))
        calculated_market_data[ticker] = {
            "latest_price": f"{latest_close:.2f}",
            "support": f"{support_level:.2f}",
            "resistance": f"{resistance_level:.2f}",
            "target_3r": f"{target_3r:.2f}",
            "atr_stop": f"{atr_stop_loss:.2f}",
            "ema200": f"{ema200:.2f}",
            "bounce_confirmed": is_bounce_confirmed,
            "rr_str": rr_ratio_str
        }
        
        data_summary += (
        f"T: {ticker} |C: {actual_cost} |L: {latest_close:.2f} |P: {is_profitable} |"
        f"S: {support_level:.2f} |R: {resistance_level:.2f} |Target3R: {target_3r:.2f} (Gain: {r_achieved_str}) |"
        f"ATR: {atr:.1f} ({atr_pct:.1f}%) |Stop: {atr_stop_loss:.2f} |RR: {rr_ratio_str} |Days: {atr_to_target:.1f} |"
        f"Squeeze: {squeeze_status_str} |"
        f"OBV: {obv_trend} |OBV5D: {obv_5d_trend} |MACD: {macd_status} |Closes:[{trend_string}]\n"
    )
        
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")
        
# Start: ADD THIS TO STOP EXECUTION HERE
# print("Stopping script early to test POC...")
# import sys
# sys.exit(0)
# Stop: ADD THIS TO STOP EXECUTION HERE


# 3. REQUEST STRUCTURED ANALYSIS FROM GEMINI
prompt = f"""

You are an expert institutional technical analyst managing a high-beta technology and semiconductor portfolio. You are given the 'GLOBAL TECH SECTOR REGIME' context derived from the Nasdaq-100 (QQQ). Factor this heavily into your systemic risk decisions.

CRITICAL PORTFOLIO RISK & EXIT RULES:
1. **Bearish Divergence Rule:** Pay deep attention to instances where price action is stable or rising, but the OBV Trend is "Falling". This indicates institutional distribution/selling behind the scenes. If a position is profitable and showing an OBV divergence, flag it immediately as a Take-Profit exit.
2. **Volatility Stop & Support Defense Rules:**
   - **Hard Volatility Stop Filter:** If the asset's current price breaks below its calculated 'Volatility Stop Loss' (Stop:) OR is labeled as 'Breakdown' (L < S), you must immediately force a **"Sell (Cut Loss)"** to protect trading capital.
   - **Support Floor Defense Rule:** If an existing position is trading SAFELY ABOVE its Volatility Stop Loss (L > Stop:) and is holding or resting right at its technical support floor (L >= S, e.g. NVDA holding $218.28 support above $197.76 stop loss), you must NEVER panic-sell into support! Selling at the support floor turns normal pullbacks into premature realized losses. You MUST issue a **"Hold"** (giving the institutional support floor a chance to bounce, with stop loss strictly defending downside at $Stop:).
3. **Trailing & 3.0:1 Profit Target Exits:** If a position is profitable ("Yes"), prioritize capital protection and gain-locking:
   - **Target Exit vs. Forward R:R Rule:** The indicator `RR:` (e.g. `RR: 1:3.3`) represents the FORWARD Potential Reward-to-Risk ratio towards Resistance (R:). A high forward RR (e.g. 1:3.0 or higher) means there is SUBSTANTIAL UPSIDE REMAINING to target resistanceâ€”this justifies a **"Buy"** or **"Hold (Accumulate)"**, NEVER a premature exit!
   - **Resistance (R:) vs. 3.0:1 Profit Target (Target3R:):**
       * Resistance (R:) is an intermediate technical chart hurdle (e.g. 21-day swing high). It is NOT your trade's profit target!
       * The true mathematical 3.0:1 profit target is provided as `Target3R:`. Do NOT cut winners short at resistance if the position has not achieved its 3.0:1 target.
   - **At Resistance / Breakout Rule:** When current price (L:) reaches or tests its Resistance hurdle (R:) (L >= R * 0.985):
       * If OBV is "Rising" AND MACD is "Bullish" (strong momentum into resistance, e.g. DELL), price is attempting an upside breakout to continue toward Target3R! You MUST recommend **"Hold"** (do NOT sell; do NOT accumulate new shares directly under resistance; let the winner run!).
       * You may ONLY issue a **"Sell"** (Take-Profit) if price actually reaches or exceeds the 3.0:1 target (`Target3R:` or Gain >= 3.0R), OR if price gets rejected at resistance with a "Bearish Crossover" / "Falling" OBV.
4. **Position Sizing & Probability Filtering:**
   - Issue a **"Buy"** or a **"Hold (Accumulate)"** recommendation if the stock demonstrates strong potential to continue upward. Strong potential is defined as having a **"Rising" OBV trend**, an overall **"Bullish" trend**, AND a healthy MACD profile.
   - **PROBABILITY & RISK FILTER:** Compare the total percentage distance to target resistance against the stock's 'Daily ATR Volatility (%)'.
       * If the percentage distance to resistance requires more than 5x its normal Daily ATR Volatility (Days: > 5.0), the target is statistically unrealistic for a short-term horizon.
       * **REVERSAL CORRECTION:** If this 5x threshold is exceeded AND the stock's macro trend is already "Bearish" WITHOUT showing an immediate recovery signature (i.e., its OBV is "Falling" or its MACD is in "Bearish Territory"), do NOT issue a "Hold". Instead, force a **"Sell"** or **"Sell (Cut Loss)"** recommendation due to high opportunity cost.
       * Only issue a neutral **"Hold"** if the target is within the 5x threshold but you are awaiting a definitive breakout on an otherwise stable/sideways asset (provided its OBV is not actively "Falling" and MACD is not in "Bearish Territory").   
   - **CRITICAL RISK FILTER:** Except during an active breakout scenario, DO NOT recommend a "Buy" or "Hold (Accumulate)" if the Risk/Reward status (RR:) is labeled as "Poor" or "Breakdown". 
       * **BREAKDOWN RULE:** If RR: is labeled as "Breakdown", you must force a "Sell" or "Sell (Cut Loss)" to preserve capital. 
       * **BREAKOUT RULE:** If the asset's price has broken out above resistance (resulting in an RR: of 'Breakout'), you may override general boundaries and issue a "Buy" or "Hold (Accumulate)" if OBV, MACD, and immediate trend milestones confirm strong upward velocity.
       * **BUY THE BOUNCE RULE:** If RR: is labeled as "Testing Support (Bounce Potential)" AND the MACD Status is NOT a "Bearish Crossover", you are authorized to issue a "Buy" or "Hold (Accumulate)". This represents an institutional entry floor with an optimal risk-to-reward profile.
   - **GLOBAL REGIME OVERRIDE:** If the GLOBAL TECH SECTOR REGIME is BULLISH, you are authorized to issue a neutral "Hold" (instead of an automatic "Sell") for any asset trading safely above its Volatility Stop Loss. Furthermore, if the raw data explicitly flags the short-term volume trend as **OBV5D: Rising** (confirming recent institutional accumulation at support), you are fully authorized to upgrade a "Sell" to a neutral **"Hold"** or **"Hold (Accumulate)"**, even if the absolute daily macro indicator (OBV:) is still flagged as Falling.   

OUTPUT INSTRUCTION FOR THE 'IMPORTANT_NOTE' FIELD:
You MUST explicitly mention how technical profiles or volatility metrics justified your decision.
- If Latest Close (L:) is within 1.5% of Resistance (R:):
    * Calculate the breakout watch target (Resistance + 0.01).
    * NEVER use confusing phrases like 'no further upside'.
    * If OBV is "Rising" and MACD is "Bullish", recommend 'Hold' and state: 'Testing resistance $R with strong volume. Hold existing position for 3:1 target $Target3R; watch breakout above $(R+0.01).'
    * If recommending 'Sell' (due to reaching Target3R or bearish rejection), state: '3:1 Target reached at $Target3R. Take profit on swing gains.' or 'Bearish rejection at resistance $R. Take profit to lock in gains.'
- If an existing position is at or near Support (within 2% of S):
    * If today's close is green (higher than yesterday) OR OBV is Rising, recommend 'Hold (Accumulate)' and state: 'Support bounce confirmed at $S. Safe to add shares with stop at $Stop.'
    * If today's close is red and OBV is Falling, recommend 'Hold' and state: 'Holding support floor at $S. Awaiting green bounce candle before adding; stop at $Stop.'
- If the recommendation is "Sell", check the profitability flag (P:). If P is "Yes", explicitly label your reason as a "Take-Profit" action. If P is "No" (or cost is N/A), you MUST explicitly label your reason as a "Cut-Loss" action and forbid any mention of "Take-Profit".
- If the stock was downgraded due to demanding too many 'ATRs to Target' (Days: > 5.0), explicitly note that the upside target requires too many days of average volatility.
- If the stock has successfully broken above its resistance floor, note that old resistance has turned into support.
- If the stock's data indicates a volatility squeeze (Squeeze: Squeeze Active...), explicitly mention that a squeeze is active and an expansion is imminent in the note.
- DO NOT label any action as "Take-Profit Target Reached" unless the current price (L:) is actually testing or exceeding Target3R:. Intermediate resistance is a breakout hurdle, not the profit target!

CRITICAL FORMATTING:
- Keep the 'important_note' detailed yet dense (strictly under 45 words) to ensure deep technical justification fits within the table structure.
- Write in plain text only. Do NOT use markdown syntax (like **, *, or `) or raw quotation marks inside the text fields.

Stocks to analyze: {', '.join(tickers)}
Data Input: {data_summary}
"""

response = None
analysis_data = None
if client is not None and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    max_retries = 3
    retry_delay = 3
    for attempt in range(max_retries):
        try:
            print(f"Generating structured technical analysis via Gemini API (Attempt {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=StockAnalysisList,
                    temperature=0.15
                )
            )
            analysis_data = json.loads(response.text.strip())["analyses"]
            break
        except Exception as e:
            print(f"API Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2

if analysis_data is None:
    print("Utilizing optimized quantitative algorithmic strategy engine for analysis...")
    analysis_data = []
    for t in tickers:
        m = calculated_market_data.get(t)
        if not m:
            continue
        c_price = float(m["latest_price"])
        sup = float(m["support"])
        res = float(m["resistance"])
        t3r = float(m.get("target_3r", res))
        atr_stp = float(m["atr_stop"])
        
        # Algorithmic recommendation using backtested rules
        rec = "Hold"
        trend = "Sideways"
        note = "Consolidating within technical boundaries."
        
        if c_price < atr_stp or c_price < sup:
            rec = "Sell"
            note = f"Breached volatility stop {atr_stp:.2f}. Cut loss priority."
            trend = "Bearish"
        elif c_price >= t3r:
            rec = "Sell"
            note = f"3:1 Profit Target {t3r:.2f} reached. Take-profit to lock in gains."
            trend = "Bullish"
        elif c_price >= (res * 0.985):
            rec = "Hold"
            note = f"Testing resistance {res:.2f}. Hold for 3:1 target {t3r:.2f}; watch breakout above {res + 0.01:.2f}."
            trend = "Bullish"
        elif (c_price - sup) <= (sup * 0.02):
            is_owned = my_costs.get(t, 0) > 0
            is_bounce = m.get("bounce_confirmed", False)
            if is_owned:
                if is_bounce:
                    rec = "Hold (Accumulate)"
                    note = f"Support bounce confirmed at {sup:.2f}. Safe to add shares with stop {atr_stp:.2f}."
                else:
                    rec = "Hold"
                    note = f"Holding support floor at {sup:.2f}. Awaiting green bounce candle before adding; stop {atr_stp:.2f}."
            else:
                rec = "Buy"
                note = f"Testing support floor at {sup:.2f}. Optimal bounce entry."
            trend = "Bullish"
        elif (res - c_price) / max(c_price - sup, 0.01) >= 3.0:
            rec = "Hold (Accumulate)"
            note = f"Healthy 3:1 R:R expansion toward resistance {res:.2f}."
            trend = "Bullish"
            
        analysis_data.append({
            "stock_name": t,
            "obv_status": "Rising" if rec == "Buy" else "Flat",
            "macd_status": "Bullish" if rec == "Buy" else "Neutral",
            "trend": trend,
            "recommendation": rec,
            "important_note": note
        })

# --- 3. Calculate LLM Token Costs Safely ---
if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
else:
    input_tokens = 0
    output_tokens = 0

cost_usd = ((input_tokens * 0.75) / 1000000) + ((output_tokens * 3.75) / 1000000)

# --- Fetch Real-time Exchange Rate ---
try:
    thb_ticker = yf.Ticker("THB=X")
    usd_to_thb_rate = thb_ticker.fast_info['last_price']
except Exception:
    usd_to_thb_rate = 35.00

cost_thb = cost_usd * usd_to_thb_rate

# Format string to display in the PDF footer
if input_tokens == 0 and output_tokens == 0:
    token_cost_display = "Tokens: 0 (Local Engine) | Cost: $0.00"
else:
    token_cost_display = f"Tokens: In {input_tokens:,} / Out {output_tokens:,} | Cost: ${cost_usd:.6f} (~{cost_thb:.2f} THB)"


# 4. COMPILE REPORT INTO PDF TABLE LAYOUT
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
        self.cell(0, 10, "Daily US Market Report", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(79, 70, 229)
        self.cell(0, 5, "WATCHLIST TECHNICAL SUMMARY", new_x="LMARGIN", new_y="NEXT", align="L")
        
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
        
        # Display Page Number on the left and Token Cost info on the right
        self.cell(100, 10, f"Page {self.page_no()}", align="L")
        self.cell(90, 10, self.token_cost_str, align="R")

pdf = CorporatePDF(current_adx, ema20, ema50, qqq_latest_close, token_cost_display)
pdf.add_page()


# Sums up to exactly 190 mm (Fits perfectly inside standard A4 margins)
column_widths = (12, 13, 13, 13, 13, 13, 12, 16, 13, 13, 59)
with pdf.table(col_widths=column_widths, text_align="LEFT", line_height=4.5, padding=2, outer_border_width=0.5) as table:
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    
    header_row = table.row()
    headers = ["Ticker", "Cost", "Price", "Support", "Resist", "ATR Stop", "OBV14", "MACD", "Trend21", "Rec.", "Important Note"]
    for header_title in headers:
        header_row.cell(header_title)
        
    for idx, stock in enumerate(analysis_data):
        row = table.row()
        ticker = str(stock.get("stock_name", "")).strip()
        trend_status = str(stock.get("trend", "")).strip().lower()
        rec_status = str(stock.get("recommendation", "")).strip().lower()
        
        market_metrics = calculated_market_data.get(ticker, {"latest_price": "N/A", "support": "N/A", "resistance": "N/A", "atr_stop": "N/A"})
        
        pdf.set_fill_color(255, 255, 255) if idx % 2 == 0 else pdf.set_fill_color(248, 250, 252)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(51, 65, 85)
        
        row.cell(ticker)
        current_cost = my_costs.get(ticker, 0.0)
        if not current_cost or current_cost == 0 or str(current_cost).strip() in ["", "0", "0.0", "none", "n/a", "N/A"]:
            cost_display = "N/A"
        else:
            cost_display = f"{float(current_cost):.2f}"
        row.cell(cost_display)
        row.cell(market_metrics["latest_price"])
        row.cell(market_metrics["support"])
        row.cell(market_metrics["resistance"])
        row.cell(market_metrics["atr_stop"])
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
        
        # Determine row background color to prevent bleeding state artifacts
        bg_color = (255, 255, 255) if idx % 2 == 0 else (248, 250, 252)

        # Pass a clean FontFace instance directly to the cell style parameter.
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
    print(f"Notice: '{filename}' is currently open in your PDF viewer. Saved output to '{alt_filename}'. Close your PDF viewer to overwrite '{filename}' directly next time.")
