from pdfminer.high_level import extract_text
import sys
import os

def extract(pdf_path):
    try:
        if not os.path.exists(pdf_path):
             print(f"Error: File not found at {pdf_path}")
             return

        text = extract_text(pdf_path)
        
        output_path = pdf_path.replace(".pdf", ".txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Text extracted to:", output_path)
    except Exception as e:
        print(f"Error extracting text: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_pdf_extract.py <pdf_path>")
    else:
        extract(sys.argv[1])