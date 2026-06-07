import os
import yfinance as yf
from google import genai
from fpdf import FPDF
from datetime import datetime

class MorningReportPDF(FPDF):
    def header(self):
        # Dark corporate header bar
        self.set_fill_color(15, 23, 42) # Slate-900
        self.rect(0, 0, 210, 35, 'F')
        
        # Header text
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.set_y(10)
        self.cell(0, 6, "AUTOMATED MARKET ANALYSIS REPORT", ln=True, align="L")
        
        # Metadata timestamp
        self.set_font("Helvetica", "", 10)
        self.set_text_color(148, 163, 184) # Slate-400
        current_time = datetime.now().strftime("%B %d, %Y | %I:%M %p UTC")
        self.cell(0, 6, f"Asset: ^GSPC (S&P 500 Index)   |   Generated: {current_time}", ln=True, align="L")
        self.set_y(45) # Drop cursor past the header bar

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
    
    Format your response cleanly in Markdown using these exact headers:
    **1. CURRENT MARKET TREND**
    **2. KEY STRUCTURAL LEVELS**
    **3. MORNING OPENING OUTLOOK**
    
    Keep descriptions succinct, actionable, and use normal line spacing.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    analysis_text = response.text

    print("Compiling markdown layout...")
    pdf = MorningReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Configure document typography
    pdf.set_text_color(51, 65, 85) # Slate-700
    pdf.set_font("Helvetica", "", 11)
    
    # Render everything using native markdown wrapping
    # pdf.epw calculates effective page width (accounting for margins)
    pdf.multi_cell(w=pdf.epw, h=6, txt=analysis_text, markdown=True)

    pdf.output("morning_market_analysis.pdf")
    print("PDF output finalized successfully.")

if __name__ == "__main__":
    run_pipeline()
