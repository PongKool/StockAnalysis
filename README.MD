# Dual-Engine Stock Analysis AI Agent 📊

This project implements an **automated, institutional-grade financial analysis pipeline** designed to monitor two distinct portfolios: high-beta US tech equities and conservative Thai blue chips. By cleanly dividing processing tasks between high-speed local mathematical scripts and the Gemini AI engine, it evaluates risk margins, spots institutional distribution, and exports production-ready market reports without wasting API token efficiency.

The system executes automatically across two isolated workflows: US Equities (`get_market.py`) and Thai SET Equities (`get_market_th.py`).

[View Latest Market Analysis (Direct)](https://raw.githubusercontent.com/PongKool/StockAnalysis/main/morning_market_analysis.pdf)
(https://raw.githubusercontent.com/PongKool/StockAnalysis/main/thai_market_analysis.pdf)

---

## 🚀 Key Architectural Upgrades

* **High-Speed Bulk Data Downloading:** Replaced slow, sequential ticker-by-ticker network queries with a singular multi-index bulk payload download via `yf.download()`. This eliminates sequential network lag and cuts data-gathering times by up to **80%**.
* **Dual Macro Regime Filtering:** * **US Engine:** Integrates index tracking on the Invesco NASDAQ-100 ETF (`QQQ`) using a 20-day EMA to map global sector momentum.
  * **Thai Engine:** Integrates benchmark tracking on the SET50 Index ETF (`TDEX.BK`) using a 20-day EMA to track domestic blue-chip liquidity.
* **Targeted Risk Profiles:**
  * **US Tech Volatility:** Incorporates a 14-day **Average True Range (ATR)** to scale dynamic stop boundaries customized to aggressive intraday tech swings (*NVDA, AMD, AVGO*).
  * **Thai Value Structural Maps:** Relies strictly on raw horizontal **1-Month Support/Resistance boundaries** to locate structural entry floors for heavy cash-flow blue chips (*BBL, ADVANC, PTT*).
* **Strict JSON Type-Safety (Pydantic):** Employs Pydantic's `BaseModel` injected directly into Gemini's native `response_schema`. This forces the AI to output pure structured JSON with zero markdown code blocks, eliminating pipeline runtime parsing exceptions.
* **Unicode Table Sanitization:** Hardened text parsing channels automatically intercept and convert problematic hidden Unicode marks (like long en-dashes `–`) into standard safe hyphens to guarantee headless `fpdf2` rendering success.

---

## 🧠 Institutional Decision Engines

The Gemini AI processes each portfolio through custom financial lenses, explicitly printing the relationship between the macro regime and the asset's risk/reward setup in the final report columns:

### 1. US Tech Framework (High-Beta Momentum & Volatility)
* **QQQ Guardrail:** If `QQQ` trades below its 20-day EMA, systemic tech risk is high. The AI overrides local indicators, pauses pullback "Buy" ratings, and tightens trailing profit stops on open positions.
* **ATR Volatility Slicing:** Uses ATR to ensure entries are not initiated at overextended price peaks, requiring clear risk-asymmetry relative to the 1-month range before declaring a trade valid.

### 2. Thai Blue-Chip Framework (Value & Capital Preservation)
* **SET50 Liquidity Filter:** Uses the `TDEX.BK` 20-day EMA to gauge institutional capital flows in Bangkok. If the broader market is in a structural markdown, the AI stops "Buy" recommendations on value pullbacks to keep from catching falling knives.
* **Defensive Trailing Exits:** Prioritizes guarding principal capital over waiting for speculative breakouts. If a position is profitable while its OBV trend falls or the MACD logs a bearish crossover, it enforces an immediate exit recommendation (**Sell** or **Hold**) to lock in gains.

---

## 🧠 Split-Execution Core Architecture

To minimize API latency and maximize token efficiency, tasks are heavily isolated:

### 1. What Local Python Handles (Deterministic Calculation)
* **Data Aggregation:** Bulk-downloads historical matrices to allow EMA calculation smoothing to perfectly stabilize.
* **Technical Metrics:** Computationally derives standard Support/Resistance boundaries locked to a trailing 1-Month (21 trading days) window, computes running On-Balance Volume (OBV), and traces MACD historical crossovers.
* **Immutable PDF Generation:** Assembles a highly scannable, math-aligned data table utilizing `fpdf2` ($190\text{ mm}$ printable width grid), complete with conditional green/red status highlighting.

### 2. What the Gemini AI Agent Handles (Financial Reasoning)
* **Strategic Financial Reasoning:** Reviews the compiled technical summaries alongside hardcoded Entry Costs to output clear, actionable ratings (**Buy**, **Hold**, or **Sell**).
* **Bearish Divergence Hunting:** Evaluates if price action is climbing while its underlying volume matrix (`OBV Trend`) is simultaneously falling—flagging institutional distribution to enforce immediate defensive take-profit exits.

---

## ⏱️ Technical Matrix Parameters

The pipeline standardizes indicators across both execution streams:

| Metric | Evaluation Window | Technical Function / Mechanics |
| :--- | :--- | :--- |
| **Pricing Engine** | `auto_adjust=True` | Extracts split/dividend-adjusted historical bars for live intraday calculations. |
| **US Volatility Scaling** | 14-Day ATR | Estimates daily tech trading ranges to identify extreme momentum extensions. |
| **US Macro Regime** | 20-Day EMA | Tracks `QQQ` price posture to map global tech sector velocity. |
| **Thai Macro Regime** | 20-Day EMA | Tracks `TDEX.BK` price posture to map domestic blue-chip liquidity. |
| **Support & Resistance** | 1-Month Lookback | Scans 21 trading days (`hist_1m['Low'].min()`) to isolate structural floor and ceiling zones. |
| **Volume Velocity** | 5-Day OBV Trend | Computes directional volume accumulation to flag institutional buying vs. distribution. |
| **Trend Continuity** | 3-Month Lookback | Sets up baseline fast/slow moving averages to accurately trace institutional MACD crossovers. |

---

## 📥 Generated Artifacts

Upon completing successful workflow triggers, the pipeline automatically generates two professional data logs:
1.  `morning_market_analysis.pdf` — Aggregated corporate brief outlining core US portfolio positioning.
2.  `thai_market_analysis.pdf` — Formatted technical summary reporting on Thai asset positions.
