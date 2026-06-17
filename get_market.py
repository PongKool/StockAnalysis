import os
import yfinance as yf
import pandas as pd
from google import genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
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
        if len(hist) < 20:
            return None

        # 1. Price & Support/Resistance (Using 20-day High/Low Channel)
        latest_close = hist['Close'].iloc[-1]
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

        # Distance to resistance in terms of ATR days
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

        # 6. Structural Trend (Price vs 20 EMA)
        ema20 = hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
        trend = "Bullish" if latest_close > ema20 else "Bearish"

        # --- DYNAMIC ATR SUPPORT BUFFER LOGIC ---
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

def generate_ai_suggestion(ticker, metrics, regime):
    """Send structured data payload to Gemini to parse recommendations using fixed prompt criteria."""
    cost = PORTFOLIO_COSTS.get(ticker, 0)
    data_summary = (
        f"Ticker: {ticker} | Cost Basis: ${cost:.2f} | Current Price: ${metrics['price']:.2f} | "
        f"Support Floor: ${metrics['support']:.2f} | Support Status: {metrics['support_status']} | "
        f"Resistance Ceiling: ${metrics['resistance']:.2f} | "
        f"Daily ATR Volatility: {metrics['atr']:.2f} ({metrics['atr_pct']:.1f}%) | "
        f"ATRs Required to Hit Target Resistance: {metrics['atr_to_target']:.1f} days | "
        f"2.5x ATR Trailing Stop Floor: ${metrics['atr_stop']:.2f} | "
        f"On-Balance Volume (OBV): {metrics['obv']} | MACD Line Status: {metrics['macd']} | "
        f"Structural Trend: {metrics['trend']} | Broad Market Regime: {regime}"
    )

    prompt = f"""
    You are an elite institutional risk-management system. Analyze the following data for {ticker} and generate a highly strict execution action ("Buy", "Hold", "Hold (Accumulate)", "Sell", or "Sell (Cut Loss)") along with a single-sentence reason.

    CRITICAL PORTFOLIO RISK & EXIT RULES:
    1. **TAKE PROFIT RULE:** If the stock is highly profitable (Price is well above Cost Basis) but exhibits a Bearish Crossover or Bearish Territory MACD paired with "Falling" OBV, force a "Sell" or "Take Profit" to lock in gains. Do not stay exposed to volume distribution at the top.
    2. **TRAILING STOP RULE:** If the Current Price falls below the '2.5x ATR Trailing Stop Floor', the position is broken. Force a "Sell (Cut Loss)".
    3. **REGIME COUPLING:** If the Broad Market Regime is "BEARISH" and the stock's individual Structural Trend is "Bearish", do not buy or hold. Default to "Sell".
    4. **PROBABILITY & RISK FILTER:** Compare the total percentage distance to target resistance against the stock's 'Daily ATR Volatility (%)'.
       * Calculate the volatility multiplier needed to hit the target. If the percentage distance to resistance is greater than 500% (5x) of its normal Daily ATR Volatility, the target is statistically unrealistic.
       * **CRITICAL CORRECTION:** Only force an immediate **"Sell (Cut Loss)"** if the 5x threshold is exceeded AND the 'Support Status' is "BROKEN".
       * If the 5x threshold is exceeded but 'Support Status' is "TESTING SUPPORT", do NOT sell yet. Instead, issue a highly tactical **"Hold (Watch Support)"** recommendation to avoid panic-selling at the absolute bottom before a confirmed breakdown.

    DATA TO EVALUATE:
    {data_summary}

    Return EXACTLY a valid JSON object matching this structure:
    {{
      "recommendation": "YOUR_RECOMMENDATION_HERE",
      "note": "YOUR_SINGLE_SENTENCE_CRITICAL_REASON_HERE"
    }}
    """
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return pd.read_json(io.StringIO(clean_text), typ='series').to_dict()
    except Exception as e:
        return {"recommendation": "Hold", "note": "Analysis calculation failed pipeline."}

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
            Paragraph(f"${item['cost']:.2f}", cell_style),
            Paragraph(f"${item['price']:.2f}", cell_style),
            Paragraph(f"${item['support']:.2f}", cell_style),
            Paragraph(f"${item['resistance']:.2f}", cell_style),
            Paragraph(f"${item['atr_stop']:.2f}", cell_style),
            Paragraph(item['obv'], cell_style),
            Paragraph(item['macd'].replace(" ", "<br/>"), cell_style),  # Replaces space with line break for 2-line clean wrap
            Paragraph(item['trend'], cell_style),
            Paragraph(item['rec'], rec_style),
            Paragraph(item['note'], cell_style),
        ]
        table_data.append(row)

    # Balanced 540pt grid configuration: Cost, Price, Support, Resist, and ATR Stop are all exactly 44 points wide.
    col_widths = [36, 44, 44, 44, 44, 44, 40, 52, 52, 44, 96]
    
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
    master_matrix = []
    
    for ticker in WATCHLIST:
        print(f"Processing structural health analytics for {ticker}...")
        metrics = calculate_technical_metrics(ticker)
        if metrics:
            ai_output = generate_ai_suggestion(ticker, metrics, regime_status)
            master_matrix.append({
                "ticker": ticker,
                "cost": PORTFOLIO_COSTS[ticker],
                "price": metrics['price'],
                "support": metrics['support'],
                "resistance": metrics['resistance'],
                "atr_stop": metrics['atr_stop'],
                "obv": metrics['obv'],
                "macd": metrics['macd'],
                "trend": metrics['trend'],
                "rec": ai_output.get("recommendation", "Hold"),
                "note": ai_output.get("note", "No comment provided.")
            })
            
    if master_matrix:
        build_pdf_report(master_matrix, regime_status)
