for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        # Fetching 3 months of price context to accurately calculate 26-period EMA for MACD
        hist = stock.history(period="3mo")
        
        if hist.empty or len(hist) < 26:
            continue
            
        # --- THE FIX: Clean out any incomplete live/placeholder rows containing NaN ---
        hist = hist.dropna(subset=['Close'])
        
        # --- CALCULATE OBV ---
        # OBV = Previous OBV + Current Volume (if close up) or - Current Volume (if close down)
        direction = hist['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (direction * hist['Volume']).cumsum()
        latest_obv = obv.iloc[-1]
        obv_trend = "Rising" if obv.tail(5).diff().mean() > 0 else "Falling"
        
        # --- CALCULATE MACD (State + Trajectory Logic) ---
        exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        
        # Check previous day to determine if a fresh crossover occurred
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
                
        # Format actual cost to 2 decimal places to save tokens
        cost_val = my_costs.get(ticker, 0.0)
        actual_cost = f"{cost_val:.2f}" if cost_val > 0 else "N/A"
        
        # Calculate true Support and Resistance using the trailing 1 month (approx last 21 trading days)
        hist_1m = hist.tail(21)
        support_level = hist_1m['Low'].min()
        resistance_level = hist_1m['High'].max()
        
        # This will now safely grab the final valid, completely settled price row
        latest_close = hist_1m['Close'].iloc[-1]
        
        # --- CALCULATE RISK/REWARD RATIO ---
        risk_distance = latest_close - support_level
        reward_distance = resistance_level - latest_close
        
        # Handle edge case where price is exactly at or above resistance to prevent division by zero
        if reward_distance <= 0:
            rr_ratio_str = "Poor (At Resistance)"
        elif risk_distance <= 0:
            rr_ratio_str = "Excellent (At Support)"
        else:
            # Expressed as a single decimal ratio (e.g., Risking $1 to make $2.50 -> 1:2.50)
            calculated_ratio = reward_distance / risk_distance
            rr_ratio_str = f"1:{calculated_ratio:.2f}"
            
        # Compress the recent 14 days of closing prices into a minimal string layout
        recent_closes = hist_1m.tail(14)
        trend_string = ", ".join([f"{row['Close']:.2f}" for _, row in recent_closes.iterrows()])
        
        # Token-optimized data line representing the stock status including OBV, MACD, and Risk/Reward
        data_summary += (
            f"Ticker: {ticker} | Entry Cost: {actual_cost} | Latest Close: {latest_close:.2f} | "
            f"1Mo Support: {support_level:.2f} | 1Mo Resistance: {resistance_level:.2f} | "
            f"Risk/Reward: {rr_ratio_str} | "
            f"OBV: {latest_obv:.0f} ({obv_trend}) | MACD: {latest_macd:.2f} (Signal: {latest_signal:.2f}, {macd_status}) | "
            f"Recent Close Trend: [{trend_string}]\n"
        )
    except Exception as e:
        print(f"Error gathering data for {ticker}: {e}")
