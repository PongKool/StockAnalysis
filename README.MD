# Stock Analysis AI Agent 📊

This project uses an *AI agent* to extract market data from Yahoo Finance, calculate institutional-grade technical indicators, and compile a structured financial report.

## Key Features:
- Extracts current stock prices and tracks historical data arrays.
- Computes key support and resistance levels.
- Evaluates volume confirmation (**OBV**) and structural momentum (**MACD**).
- Computes real-time **Risk/Reward Ratios** based on 1-month market boundaries.
- Analyzes trend data to generate *Buy / Hold / Sell* recommendations tailored to your personal entry cost basis.

## 🧠 How AI is Used in This Project

To maximize token efficiency and lower operational costs, this project splits tasks between *local Python code* and *the Gemini AI agent*:

### 1. What Local Python Handles (Non-AI)
- **Data Fetching:** Python connects directly to Yahoo Finance's public API using `yfinance` to download historical market data.
- **Mathematical Operations:** Python calculates exact technical metrics natively. It scans the historical data to compute Support, Resistance, cumulative On-Balance Volume (OBV), Moving Average Convergence Divergence (MACD), and the exact **Risk/Reward Ratio** without wasting AI tokens on raw math.
- **PDF Compilation:** Once the analysis is complete, Python uses `fpdf2` to dynamically build, color-code, and render the final `morning_market_analysis.pdf` report.

### 2. What the Gemini AI Agent Handles
- **Strategic Financial Reasoning:** The AI reviews the compiled technical summary alongside your hardcoded Entry Cost. It applies professional financial logic to determine an actionable recommendation (*Buy*, *Hold*, or *Sell*)—such as blocking a "Buy" rating if a stock's Risk/Reward ratio is poor, even if momentum looks bullish.
- **Dynamic Commentary:** It generates an individualized, human-readable technical summary (`important_note`) explaining the current market state, volume health, and momentum context for each ticker.
- **Structural Enforcement:** The AI acts as a data translator, formatting its complex analytical output into a strict, clean JSON array matching an explicit schema so Python can parse it without errors.

## ⏱️ Timeframes & Analysis Metrics

Because this system tracks highly volatile, tech-heavy stocks (e.g., *NVDA, AMD, AVGO, TSM*), it analyzes market movements using multiple distinct timeframes to eliminate short-term market noise and isolate clean technical setups:

| *Metric* | *Timeframe* | *How It Works* |
| ----- | ----- | ----- |
| *Support & Resistance* | *1-Month Lookback* | Python evaluates a full month of daily data (~21 trading days) to find the absolute highest ceiling and lowest floor (`hist['Low'].min()`). This provides a robust, macro price floor where buyers historically step in. |
| *Risk/Reward Ratio* | *Structural Range* | Evaluates the mathematical distance from the latest close to its 1-month support floor vs. its resistance ceiling. Expressed as an objective ratio (e.g., `1:2.50`) to enforce strict trade discipline. |
| *OBV Trend* | *5-Day Momentum* | Evaluates the running cumulative volume total over the trailing 5 trading days. A positive average slope flags the metric as *Rising*, providing short-term volume validation to support recent price activity. |
| *MACD Status* | *3-Month Context* | Leverages a longer historical lookback to accurately calculate institutional-standard moving averages (12-day Fast EMA, 26-day Slow EMA, and a 9-day Signal EMA). Dynamically flags structural states as *Bullish/Bearish Territory* or immediate *Crossovers*. |
| *Trend Analysis* | *14 Trading Days* | Gemini is supplied with a compressed list of the last 14 closing prices (~3 weeks of market action). This allows the AI to determine if the short-term direction is *Bullish*, *Bearish*, or *Sideways* while smoothing out random single-day tech volatility. |
