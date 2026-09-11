import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf

# 1. Configuration & Tickers
my_costs = {
    "SNDK": 1620.50, "ORCL": 156.03, "PBR": 21.14, "NVDA": 228.08, 
    "VRT": 283.89, "CEG": 287.02, "DELL": 524.25, "TSM": 421.41, "VST": 146.08
}
tickers = list(my_costs.keys())

print("Downloading 1-year historical data for backtest simulation...")
all_hist = yf.download(tickers, period="1y", auto_adjust=True, group_by='ticker')

backtest_results = {}

# 2. Backtest Loop per Ticker
for ticker in tickers:
    try:
        if all_hist is not None and not all_hist.empty and ticker in all_hist.columns.get_level_values(0):
            hist = all_hist[ticker].copy()
        else:
            hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
            
        if hist.empty or len(hist) < 200:
            continue
            
        hist = hist.dropna(subset=['Close'])
        
        # We need at least 150 days of warm-up data for indicators (EMA200, volume profile, ATR)
        start_idx = 150
        trades = []
        in_position = False
        entry_price = 0.0
        entry_date = None
        
        # Walk forward day-by-day (No look-ahead bias)
        for i in range(start_idx, len(hist)):
            hist_slice = hist.iloc[:i+1].copy()
            current_date = hist_slice.index[-1]
            latest_close = hist_slice['Close'].iloc[-1]
            
            # --- CALCULATE INDICATORS ON SLICE ---
            # ATR (14)
            high_low = hist_slice['High'] - hist_slice['Low']
            high_close = (hist_slice['High'] - hist_slice['Close'].shift()).abs()
            low_close = (hist_slice['Low'] - hist_slice['Close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = float(true_range.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
            atr_stop_loss = latest_close - (2.5 * atr_14)
            
            # OBV & EMA14 Trend
            direction = hist_slice['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            obv = (direction * hist_slice['Volume']).cumsum()
            obv_ema14 = obv.ewm(span=14, adjust=False).mean()
            obv_trend = "Rising" if not pd.isna(obv.iloc[-1]) and not pd.isna(obv_ema14.iloc[-1]) and obv.iloc[-1] > obv_ema14.iloc[-1] else "Falling"
            
            # Volume Profile (18-day window within slice)
            hist_1m = hist_slice.tail(18).copy()
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
            
            # Support and Resistance Levels
            swing_low_21d = float(hist_slice['High'].tail(21).min())
            swing_high_21d = float(hist_slice['High'].tail(21).max())
            
            support_level = max(poc_midpoint if poc_midpoint < latest_close else 0, swing_low_21d)
            resistance_level = min(poc_midpoint if poc_midpoint > latest_close else float('inf'), swing_high_21d)
            
            # Fallback if boundaries collapse
            if (resistance_level - support_level) < (1.0 * atr_14):
                support_level = latest_close - (2.0 * atr_14)
                resistance_level = latest_close + (2.0 * atr_14)
                
            risk_distance = latest_close - support_level
            support_buffer = support_level * 0.015
            closes_14d = hist_1m['Close'].tail(14).tolist()
            
            # Risk/Reward String Determination
            if latest_close < support_level:
                rr_ratio_str = "Breakdown"
            elif latest_close > resistance_level:
                rr_ratio_str = "Breakout"
            elif risk_distance <= support_buffer and (obv_trend == "Rising" or latest_close >= closes_14d[-2]):
                rr_ratio_str = "Testing Support (Bounce Potential)"
            else:
                rr_ratio_str = "Normal"
                
            # --- SIMULATE TRADE EXECUTION RULES ---
            if not in_position:
                # Buy Entry Trigger: Testing support bounce or clean breakout with rising OBV
                if rr_ratio_str == "Testing Support (Bounce Potential)" or (rr_ratio_str == "Breakout" and obv_trend == "Rising"):
                    in_position = True
                    entry_price = latest_close
                    entry_date = current_date
            else:
                # Exit Triggers: Hit ATR stop loss or breakdown condition
                if latest_close <= atr_stop_loss or rr_ratio_str == "Breakdown":
                    exit_price = latest_close
                    pft_pct = ((exit_price - entry_price) / entry_price) * 100
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": pft_pct,
                        "win": pft_pct > 0
                    })
                    in_position = False
                    
        backtest_results[ticker] = trades
        print(f"Completed backtest for {ticker}: {len(trades)} simulated trades found.")
        
    except Exception as e:
        print(f"Error backtesting {ticker}: {e}")

# 3. Aggregate and Print Performance Summary
print("\n--- BACKTEST PERFORMANCE SUMMARY ---")
total_trades = 0
total_wins = 0
all_returns = []

for ticker, trades in backtest_results.items():
    if not trades:
        continue
    ticker_wins = sum(1 for t in trades if t['win'])
    ticker_total = len(trades)
    ticker_returns = [t['return_pct'] for t in trades]
    total_trades += ticker_total
    total_wins += ticker_wins
    all_returns.extend(ticker_returns)
    
    win_rate = (ticker_wins / ticker_total) * 100 if ticker_total > 0 else 0
    print(f"{ticker} -> Trades: {ticker_total} | Win Rate: {win_rate:.1f}% | Avg Return: {np.mean(ticker_returns):.2f}%")

if total_trades > 0:
    overall_win_rate = (total_wins / total_trades) * 100
    print(f"\nOverall Portfolio Win Rate: {overall_win_rate:.1f}% across {total_trades} trades")
    print(f"Cumulative Simulated Strategy Return: {sum(all_returns):.2f}%")
else:
    print("\nNo trades were triggered across the selected timeframe and parameters.")
