import os
import yfinance as yf
from google import genai
from fpdf import FPDF
from datetime import datetime

class MorningReportPDF(FPDF):
    def header(self):
        # Draw a dark corporate header bar
        self.set_fill_color(15, 23, 42) # Slate-900 (#0f172a)
        self.rect(0, 0, 210, 35, 'F')
        
        # Header text styling
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.set_y(10)
        self.cell(0, 6, "AUTOMATED MARKET ANALYSIS REPORT", ln=True, align="L")
        
        # Meta timestamp alignment
        self.set_font("Helvetica", "", 10)
        self.set_text_color(148, 163, 184) # Slate-400
        current_time = datetime.now().strftime("%B %d, %Y | %I:%M %p UTC")
        self.cell(0, 6, f"Asset: ^GSPC (S&P 500 Index)   |   Generated: {current_time}", ln=True, align="L")
        self.set_y(45) # Lower the cursor position past the banner boundary

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, "Automated Cloud Pipeline - Powered by yfinance & Gemini API", align="center")

def run_pipeline():
    ticker = "^GSPC"
    print("Fetching raw market numbers...")
    try:
        df = yf.Ticker(ticker).history(period="1mo")
        data_snapshot = df.tail(14)[['Open', 'High', 'Low', 'Close']].to_string()
    except Exception as e:
        print(f"Market fetch error: {e}")
        return

    print("Requesting parsing from Gemini...")
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    Act as an institutional financial technical analyst. Analyze the following 14-day market data for {ticker}:
    {data_snapshot}
    
    Provide your final analysis broken cleanly into these exact headers:
    1. CURRENT MARKET TREND
    2. KEY STRUCTURAL LEVELS
    3. MORNING OPENING OUTLOOK
    
    Keep descriptions succinct and actionable. Do not use markdown bolding indicators (**) in the response body.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    analysis_text = response.text

    print("Compiling colorful PDF structure...")
    pdf = MorningReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Base layout style configuration
    pdf.set_text_color(51, 65, 85) # Slate-700
    pdf.set_font("Helvetica", "", 11)
    
    lines = analysis_text.split('\n')
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
            
        # Intercept main layout blocks to inject colored headers
        if "TREND" in clean_line.upper() or "LEVELS" in clean_line.upper() or "OUTLOOK" in clean_line.upper():
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(30, 58, 138) # Corporate Navy (#1e3a8a)
            pdf.cell(0, 8, clean_line, ln=True)
            # Underline bar for header structural depth
            pdf.set_fill_color(226, 232, 240)
            pdf.rect(pdf.get_x(), pdf.get_y(), 180, 1, 'F')
            pdf.ln(3)
            # Reset text block defaults
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(51, 65, 85)
        else:
            pdf.multi_cell(0, 6, clean_line)
            pdf.ln(1)

    pdf.output("morning_market_analysis.pdf")
    print("PDF output finalized successfully.")

if __name__ == "__main__":
    run_pipeline()
