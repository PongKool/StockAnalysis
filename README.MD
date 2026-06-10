# Stock Analysis AI Agent 📊

This project uses an *AI agent* and localized automation pipelines to extract market data from Yahoo Finance, calculate institutional-grade technical indicators, and compile structured financial reports across multiple global stock markets.

## Key Features:
- **Multi-Market Pipeline:** Sequentially processes both US Equities (`get_market.py`) and Thai SET Equities (`get_market_th.py` via `.BK` extensions).
- **Exact Pricing Safeguards:** Directly queries live metadata blocks (`stock.info`) to isolate the exact unadjusted market closing bell prices, completely preventing `NaN` data skips or split-adjusted pricing bugs.
- **Support & Resistance Mapping:** Computes macro price boundaries based on 1-month historical lookup metrics.
- **Volume & Momentum Filters:** Natively evaluates On-Balance Volume (OBV) trend slopes and MACD territorial environments/crossovers.
- **Risk/Reward Ratios:** Calculates structural target margins before rendering a final investment evaluation.

## 🧠 How AI is Used in This Project
To maximize token efficiency and lower operational costs, this project splits tasks between *local Python architectures* and *the Gemini AI agent*:

### 1. What Local Python Handles (Non-AI)
- **Data Fetching:** Python connects directly to Yahoo Finance's public API using `yfinance` to download historical market data and live summary metadata.
- **Mathematical Operations:** Python calculates exact technical metrics natively. It scans the historical data to compute Support, Resistance, cumulative On-Balance Volume (OBV), Moving Average Convergence Divergence (MACD), and the exact *Risk/Reward Ratio* without wasting AI tokens on raw math.
- **Isolated Multi-PDF Composition:** Compiles, color-codes, and exports two distinct, production-ready document tracking models sequentially every morning: `morning_market_analysis.pdf` (USD) and `thai_market_analysis.pdf` (THB).

### 2. What the Gemini AI Agent Handles
- **Strategic Financial Reasoning:** The AI reviews the compiled technical summary alongside your hardcoded Entry Cost. It applies professional financial logic to determine an actionable recommendation (*Buy*, *Hold*, or *Sell*)—such as blocking a "Buy" rating if a stock's Risk/Reward ratio is poor, even if momentum looks bullish.
- **Localized Regional Insights:** Adapts its financial commentary contextually to prioritize asset metrics in their respective native currencies (USD vs. Thai Baht structural nuances).
- **JSON Payload Synchronization:** Packs strategic technical summaries into strict structural JSON arrays for frictionless execution pipelines inside your automated GitHub workspace.

## ⏱️ Timeframes & Analysis Metrics
Because this system tracks highly volatile tech stocks (e.g., *NVDA, AMD, AVGO, TSM*) alongside premier Thai equities (e.g., *DELTA, KBANK, ADVANC*), it analyzes market movements using multiple distinct timeframes to eliminate short-term noise:

| Metric | Timeframe | How It Works |
| :--- | :--- | :--- |
| **Pricing Engine** | Live Metadata Close | Queries the core `stock.info` payload directly. This bypasses structural historical row array distortions to lock onto the precise market closing bell price. |
| **Support & Resistance** | 1-Month Lookback | Python evaluates a full month of daily data (~21 trading days) to find the absolute highest ceiling and lowest floor (`hist['Low'].min()`). This provides a robust, macro price floor where buyers historically step in. |
| **Risk/Reward Ratio** | Structural Range | Evaluates the mathematical distance from the latest close to its 1-month support floor vs. its resistance ceiling. Expressed as an objective ratio (e.g., `1:2.50`) to enforce strict trade discipline. |
| **OBV Trend** | 5-Day Momentum | Evaluates the running cumulative volume total over the trailing 5 trading days. A positive average slope flags the metric as *Rising*, providing short-term volume validation to support recent price activity. |
| **MACD Status** | 3-Month Context | Leverages a longer historical lookback to accurately calculate institutional-standard moving averages (12-day Fast EMA, 26-day Slow EMA, and a 9-day Signal EMA). Dynamically flags structural states as *Bullish/Bearish Territory* or immediate *Crossovers*. |
| **Trend Analysis** | 14 Trading Days | Gemini is supplied with a compressed list of the last 14 closing prices (~3 weeks of market action). This allows the AI to determine if the short-term direction is *Bullish*, *Bearish*, or *Sideways* while smoothing out random single-day volatility. |
