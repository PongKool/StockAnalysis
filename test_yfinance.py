import pandas as pd
import yfinance as yf

# Use a few sample tickers from your list
test_tickers = ["MU", "NVDA", "AAPL", "SNDK"]

print("=" * 50)
print("STARTING YFINANCE DIAGNOSTIC TEST")
print("=" * 50)

for ticker in test_tickers:
    print(f"\nFetching data for: {ticker}")
    try:
        ticker_obj = yf.Ticker(ticker)
        # Replicating your exact history fetch
        hist = ticker_obj.history(period="3mo")
        
        if hist.empty:
            print(f"❌ ALERT: DataFrame is completely EMPTY for {ticker}. (Likely yfinance API block or delisted stock)")
            continue
            
        print(f"Total rows returned: {len(hist)}")
        
        # Look at the very last 3 rows returned by Yahoo Finance
        print("Last 3 rows of raw data:")
        print(hist[['Open', 'High', 'Low', 'Close']].tail(3))
        
        # Test your exact slicing logic
        hist_1m = hist.tail(21)
        latest_close = hist_1m['Close'].iloc[-1]
        
        print(f"Result of .iloc[-1]: {latest_close}")
        if pd.isna(latest_close):
            print(f"❌ ALERT: Calculated price is NaN! Check if the last row above is empty.")
        else:
            print(f"✅ SUCCESS: Valid price retrieved: {latest_close:.2f}")
            
    except Exception as e:
        print(f"💥 CRITICAL ERROR running ticker {ticker}: {e}")

print("\n" + "=" * 50)
print("DIAGNOSTIC COMPLETE")
print("=" * 50)
