import pandas as pd
import numpy as np
import yfinance as yf

tickers = ["SNDK", "ORCL", "PBR", "NVDA", "VRT", "CEG", "DELL", "TSM", "VST"]

print("Downloading 1-year historical data for backtest simulation...")
all_hist = yf.download(tickers, period="1y", auto_adjust=True, group_by='ticker')

backtest_results = {}

for ticker in tickers:
    try:
        if all_hist is not None and not all_hist.empty and ticker in all_hist.columns.get_level_values(0):
            hist = all_hist[ticker].copy()
        else:
            hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
            
        if hist.empty or len(hist) < 200:
            continue
            
        hist = hist.dropna(subset=['Close'])
        start_idx = 150
        trades = []
        in_position = False
        entry_price = 0.0
        entry_date = None
        
        for i in range(start_idx, len(hist)):
            hist_slice = hist.iloc[:i+1].copy()
            current_date = hist_slice.index[-1]
            latest_close = float(hist_slice['Close'].iloc[-1])
            
            # ATR (14) via pure Pandas
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
            
            # Volume Profile (18-day window)
            hist_1m = hist_slice.tail(18).copy()
            min_price = float(hist_1m['Low'].min())
            max_price = float(hist_1m['High'].max())
            bins = np.linspace(min_price, max_price, 51)
            bin_volume = np.zeros(len(bins) - 1)
            
            for _, row in hist_1m.iterrows():
                day_low, day_high, day_vol = float(row['Low']), float(row['High']), float(row['Volume'])
                slices = np.linspace(day_low, day_high, 15) if day_high != day_low else np.array([day_low])
                vol_per_slice = day_vol / len(slices)
                for price_point in slices:
                    idx = max(0, min(np.digitize(price_point, bins) - 1, len(bin_volume) - 1))
                    bin_volume[idx] += vol_per_slice
                    
            poc_idx = np.argmax(bin_volume)
            poc_midpoint = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)
            
            swing_low_21d = float(hist_slice['High'].tail(21).min())
            swing_high_21d = float(hist_slice['High'].tail(21).max())
            
            support_level = max(poc_midpoint if poc_midpoint < latest_close else 0, swing_low_21d)
            resistance_level = min(poc_midpoint if poc_midpoint > latest_close else float('inf'), swing_high_21d)
            
            if (resistance_level - support_level) < (1.0 * atr_14):
                support_level = latest_close - (2.0 * atr_14)
                resistance_level = latest_close + (2.0 * atr_14)
                
            risk_distance = latest_close - support_level
            support_buffer = support_level * 0.015
            closes_14d = hist_1m['Close'].tail(14).tolist()
            
            if latest_close < support_level:
                rr_ratio_str = "Breakdown"
            elif latest_close > resistance_level:
                rr_ratio_str = "Breakout"
            elif risk_distance <= support_buffer and (obv_trend == "Rising" or latest_close >= closes_14d[-2]):
                rr_ratio_str = "Testing Support (Bounce Potential)"
            else:
                rr_ratio_str = "Normal"
                
            if not in_position:
                if rr_ratio_str == "Testing Support (Bounce Potential)" or (rr_ratio_str == "Breakout" and obv_trend == "Rising"):
                    in_position = True
                    entry_price = latest_close
                    entry_date = current_date
            else:
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

print("\n--- BACKTEST PERFORMANCE SUMMARY ---")
total_trades, total_wins = 0, 0
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
    print(f"\nOverall Portfolio Win Rate: {(total_wins / total_trades) * 100:.1f}% across {total_trades} trades")
    print(f"Cumulative Simulated Strategy Return: {sum(all_returns):.2f}%")
else:
    print("\nNo trades were triggered across the selected timeframe and parameters.")
