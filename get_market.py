import os
import yfinance as yf
import pandas as pd
from google import genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import datetime

# Initialize Gemini Client (using recommended google-genai SDK)
client = genai.Client()

# --- CONFIGURATION (YOUR FULL 14-STOCK LIST) ---
WATCHLIST = ["AVGO", "ANET", "TPR", "IONQ", "PLTR", "WMT", "NVDA", "NOW", "VST", "LRCX", "SNDK", "MSFT", "MU", "TSM"]
PORTFOLIO_COSTS = {
    "AVGO": 451.22, "ANET": 169.94, "TPR": 150.20, "IONQ": 64.31, 
    "PLTR": 134.49, "WMT": 120.47, "NVDA": 217.03, "NOW": 107.68, 
    "VST": 153.65, "LRCX": 319.36, "SNDK": 2094.39, "MSFT": 459.63, 
    "MU": 1020.16, "TSM": 424.30
}

def get_market_regime():
    """Determine overall market regime using QQQ relative to its 20 EMA."""
    try:
        qqq = yf.Ticker("QQQ")
        hist = qqq.history(period="50d")
        if hist.empty:
            return "UNKNOWN"
        hist['EMA20'] = hist['Close'].ewm(span=20, adjust=False).mean()
        latest_close = hist['Close'].iloc[-1]
        latest_ema = hist['EMA20'].iloc[-1]
        return "BULLISH" if latest_close > latest_ema else "BEARISH"
    except Exception as e:
        print(f"Error fetching market regime: {e}")
        return "UNKNOWN"

def calculate_technical_metrics(ticker):
    """Fetch data and calculate core technical indicators for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        if hist.empty:
            return None

        # Clean up empty rows at the end of historical data
        hist = hist.dropna(subset=['Close'])

        if len(hist) < 20:
            return None

        # Backup strategy if the latest close is missing or NaN
        latest_close = hist['Close'].iloc[-1]
        if pd.isna(latest_close) or latest_close <= 0:
            try:
                latest_close = stock.fast_info['lastPrice']
            except Exception:
                print(f"Skipping {ticker}: Pricing data completely unavailable.")
                return None

        # 1. Price & Support/Resistance
        support = hist['Low'].tail(20).min()
        resistance = hist['High'].tail(20).max()

        # 2. 14-Day Average True Range (ATR) & Volatility %
        high_low = hist['High'] - hist['Low']
        high_close = (hist['High'] - hist['Close'].shift()).abs()
        low_close = (hist['Low'] - hist['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        atr_pct = (atr / latest_close) * 100

        # 3. ATR-Based Trailing Stop (2.5x ATR)
        atr_stop = latest_close - (2.5 * atr)
        distance_to_resist = max(0, resistance - latest_close)
        atr_to_target = distance_to_resist / atr if atr > 0 else 0

        # 4. On-Balance Volume (OBV) Trend
        obv = (true_range * 0).copy()
        direction = hist['Close'].diff()
        obv.iloc[0] = 0
        for i in range(1, len(hist)):
            if direction.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + hist['Volume'].iloc[i]
            elif direction.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - hist['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        obv_ema = obv.ewm(span=10, adjust=False).mean()
        obv_trend = "Rising" if obv.iloc[-1] > obv_ema.iloc[-1] else "Falling"

        # 5. MACD (12, 26, 9)
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_sig = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_sig = signal_line.iloc[-2]

        if latest_macd > 0:
            macd_status = "Bullish Territory"
        elif prev_macd < prev_sig and latest_macd > latest_sig:
            macd_status = "Bullish Crossover"
        elif prev_macd > prev_sig and latest_macd < latest_sig:
            macd_status = "Bearish Crossover"
        else:
            macd_status = "Bearish Territory"

        # 6. Structural Trend
        ema20 = hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
        trend = "Bullish" if latest_close > ema20 else "Bearish"

        # Structural Support Status Matrix
        support_buffer = support + (0.25 * atr)
        if latest_close < support:
            support_status = "BROKEN"
        elif latest_close <= support_buffer:
            support_status = "TESTING SUPPORT"
        else:
            support_status = "SAFE"

        return {
            "price": latest_close, "support": support, "support_status": support_status,
            "resistance": resistance, "atr": atr, "atr_pct": atr_pct, 
            "atr_to_target": atr_to_target, "atr_stop": atr_stop, "obv": obv_trend, 
            "macd": macd_status, "trend": trend
        }
    except Exception as e:
        print(f"Error calculating metrics for {ticker}: {e}")
        return None

def generate_batch_ai_suggestions(all_stocks_data, regime):
    """Sends entire watchlist technical matrix to Gemini in 1 request, minimizing token overhead."""
    data_payload = ""
    for item in all_stocks_data:
        data_payload += (
            f"Ticker: {item['ticker']} | Cost: {item['cost']:.2f} | Price: {item['price']:.2f} | "
            f"Support: {item['support']:.2f} ({item['support_status']}) | Resist: {item['resistance']:.2f} | "
            f"Stop: {item['atr_stop']:.2f} | OBV: {item['obv']} | MACD: {item['macd']} | Trend: {item['trend']}\n"
        )

    prompt = f"""
    You are an elite institutional risk-management system. Analyze the following portfolio dataset and generate a strict execution action ("Buy", "Hold", "Hold (Accumulate)", "Sell", or "Sell (Cut Loss)") along with a single-sentence reason for EVERY ticker.

    CRITICAL PORTFOLIO RISK & EXIT RULES:
    1. **TAKE PROFIT RULE:** If profitable but MACD shows Bearish Crossover/Territory and OBV is "Falling", force "Sell".
    2. **TRAILING STOP RULE:** If Price falls below 'ATR Trailing Stop Floor', force "Sell (Cut Loss)".
    3. **REGIME COUPLING:** If Market Regime is "BEARISH" and Stock Trend is "Bearish", default to "Sell".
    4. **PROBABILITY FILTER:** If distance to resistance is > 5x Daily ATR Volatility AND support is "BROKEN", force "Sell (Cut Loss)". If support is "TESTING SUPPORT", use "Hold (Watch Support)".

    DATA TO EVALUATE:
    Broad Market Regime: {regime}
    {data_payload}

    Return EXACTLY a valid JSON array of objects matching this structure:
    [
      {{
        "ticker": "TICKER_SYMBOL",
        "recommendation": "YOUR_RECOMMENDATION",
        "note": "YOUR_SINGLE_SENTENCE_REASON"
      }}
    ]
    """
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        suggestions = pd.read_json(io.StringIO(clean_text))
        return suggestions.set_index('ticker').to_dict(orient='index')
    except Exception as e:
        print(f"Batch AI analysis failed: {e}")
        return {}

def build_pdf_report(data_matrix, regime):
    """Generate professional PDF summary report using ReportLab styling."""
    pdf_filename = "morning_market_analysis.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=24, leading=28, textColor=colors.HexColor("#1A365D"), spaceAfter=6)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#4A5568"), spaceAfter=20)
    cell_style = ParagraphStyle('GridCell', parent=styles['Normal'], fontSize=8, leading=10, textColor=colors.HexColor("#2D3748"))
    header_style = ParagraphStyle('GridHead', parent=styles['Normal'], fontSize=9, leading=11, bold=True, textColor=colors.white)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("Daily Market Report", title_style))
    story.append(Paragraph(f"WATCHLIST TECHNICAL SUMMARY<br/>Generated automatically on {timestamp} (Thailand Time) | Regime: {regime} (QQQ relative to 20EMA)", subtitle_style))

    headers = ["Ticker", "Cost", "Price", "Support", "Resist.", "ATR Stop", "OBV", "MACD", "Trend", "Rec.", "Important Note"]
    table_data = [[Paragraph(h, header_style) for h in headers]]

    for item in data_matrix:
        rec_color = "#C53030" if "Sell" in item['rec'] else ("#2F855A" if "Buy" in item['rec'] else "#D69E2E")
        rec_style = ParagraphStyle('RecText', parent=cell_style, textColor=colors.HexColor(rec_color), bold=True)
        
        row = [
            Paragraph(item['ticker'], cell_style),
            Paragraph(f"{item['cost']:.2f}", cell_style),  # Gaps removed from f-strings
            Paragraph(f"{item['price']:.2f}", cell_style),
            Paragraph(f"{item['support']:.2f}", cell_style),
            Paragraph(f"{item['resistance']:.2f}", cell_style),
            Paragraph(f"{item['atr_stop']:.2f}", cell_style),
            Paragraph(item['obv'], cell_style),
            Paragraph(item['macd'].replace(" ", "<br/>"), cell_style),  # Dynamic 2-line break configuration
            Paragraph(item['trend'], cell_style),
            Paragraph(item['rec'], rec_style),
            Paragraph(item['note'], cell_style),
        ]
        table_data.append(row)

    # 540pt mathematically balanced layout grid
    col_widths = [38, 44, 44, 44, 44, 44, 40, 50, 50, 46, 96]
    
    summary_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    
    story.append(summary_table)
    doc.build(story)
    print(f"Successfully compiled and saved live analysis to {pdf_filename}")

if __name__ == "__main__":
    print("Initiating production quantitative processing suite...")
    regime_status = get_market_regime()
    pre_matrix = []
    
    # Step 1: Accumulate analytical data frames locally 
    for ticker in WATCHLIST:
        print(f"Processing structural health analytics for {ticker}...")
        metrics = calculate_technical_metrics(ticker)
        if metrics:
            pre_matrix.append({
                "ticker": ticker,
                "cost": PORTFOLIO_COSTS[ticker],
                **metrics
            })
            
    # Step 2: Push batched data through a single structured LLM evaluation pass
    if pre_matrix:
        print("Sending consolidated matrix to Gemini Risk Engine...")
        ai_batch_results = generate_batch_ai_suggestions(pre_matrix, regime_status)
        
        # Step 3: Combine datasets for the structural PDF reporter
        master_matrix = []
        for item in pre_matrix:
            ticker = item['ticker']
            ticker_ai = ai_batch_results.get(ticker, {"recommendation": "Hold", "note": "Pipeline fallback check applied."})
            
            master_matrix.append({
                **item,
                "rec": ticker_ai.get("recommendation", "Hold"),
                "note": ticker_ai.get("note", "No comment provided.")
            })
            
        build_pdf_report(master_matrix, regime_status)
