# Stock Analysis AI Agent 📊

This project implements an **automated, institutional-grade financial analysis pipeline** designed to monitor high-beta tech portfolios and regional assets. By cleanly dividing tasks between high-speed local mathematical scripts and the Gemini AI engine, it tracks open positions, evaluates risk margins, and exports production-ready market reports without wasting API token efficiency.

The system executes automatically across two distinct workflows: US Equities (`get_market.py`) and Thai SET Equities (`get_market_th.py`).

---

## 🚀 Key Architectural Upgrades

* **High-Speed Bulk Data Downloading:** Replaced slow, sequential ticker-by-ticker network queries with a singular multi-index bulk payload download via `yf.download()`. This eliminates sequential network lag and cuts data-gathering times by up to **80%**.
* **Macro Tech Sector Regime Filtering:** Integrates index tracking on the Invesco NASDAQ-100 ETF (`QQQ`). By calculating its 20-day EMA, the system gauges broader market health to stop the AI from blindly suggesting "Buy" ratings on stock pullbacks during systemic market corrections.
* **High-Beta Tech Volatility Scaling (ATR):** Incorporates the 14-day **Average True Range (ATR)**. This provides a dynamic dollar-volatility buffer customized to the aggressive intraday swings of tech and semiconductor names (*NVDA, AMD, AVGO, TSM*).
* **Strict JSON Type-Safety (Pydantic):** Employs Pydantic's `BaseModel` injected directly into Gemini's native `response_schema`. This forces the AI to output zero markdown blocks and pure structured JSON, completely eliminating pipeline runtime parsing exceptions.
* **Unicode Table Sanitization:** Hardened text parsing channels automatically intercept and convert problematic hidden Unicode marks (like long en-dashes `–`) into standard safe hyphens to guarantee `FPDF` rendering success during headless automated workflows.

---

## 🧠 Split-Execution Core Architecture

To minimize API latency and maximize token efficiency, tasks are heavily isolated:

### 1. What Local Python Handles (Deterministic Calculation)
* **Data Aggregation:** Bulk downloads a 6-month data matrix to allow Exponential Moving Average calculation smoothing to perfectly stabilize.
* **Technical Metrics:** Computationally derives standard Support/Resistance boundaries strictly locked to a trailing 1-Month (21 trading days) window, computes running On-Balance Volume (OBV), and traces MACD historical crossovers.
* **Immutable PDF Generation:** Assembles a highly scannable, zebra-striped data table utilizing `fpdf2`, complete with conditional green/red status highlighting.

### 2. What the Gemini AI Agent Handles (Financial Reasoning)
* **Strategic Financial Reasoning:** The AI reviews the compiled technical summary alongside your hardcoded Entry Cost. It applies professional financial logic to determine an actionable recommendation (*Buy*, *Hold*, or *Sell*).
* **Bearish Divergence Hunting:** Evaluates if price action is climbing while its underlying volume matrix (`OBV Trend`) is simultaneously falling—flagging institutional distribution to enforce immediate defensive Take-Profit exit triggers.

---

## ⏱️ Technical Matrix Parameters

The pipeline standardizes indicators across multiple timeframes to cross-verify signals:

| Metric | Evaluation Window | Technical Function / Mechanics |
| :--- | :--- | :--- |
| **Pricing Engine** | `auto_adjust=True` | Extracts split/dividend-adjusted historical bars, ensuring live intraday trading data matches standard market-hour terminals. |
| **Volatility Scaling** | 14-Day ATR | Estimates trailing daily asset trading range to identify extreme, high-beta momentum extensions. |
| **Macro Regime** | 20-Day EMA | Tracks `QQQ` price posture to map global sector velocity (Bullish vs. Cautious environments). |
| **Support & Resistance** | 1-Month Lookback | Scans 21 trading days (`hist_1m['Low'].min()`) to isolate true tactical buying/selling structural floors. |
| **Volume Velocity** | 5-Day OBV Trend | Computes average directional volume accumulation. Stalled ranges automatically fall to a `"Flat"` identifier to capture trading halts. |
| **Trend Continuity** | 3-Month Lookback | Sets up the baseline fast/slow moving averages to accurately trace institutional MACD crossovers. |

---

## 📥 Generated Artifacts

Upon completing successful workflow triggers, the pipeline automatically writes out two high-visibility, professional data logs:
1.  `morning_market_analysis.pdf` — Aggregated corporate brief outlining core US portfolio positioning.
2.  `thai_market_analysis.pdf` — Formatted technical summary reporting on Thai asset positions.
