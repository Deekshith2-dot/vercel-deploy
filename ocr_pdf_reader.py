import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# SET PATHS
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\DEEKSHITH\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin"     # CHANGE THIS

PDF_PATH = "resume.pdf"  # keep your resume in same folder

def ocr_extract_text(pdf_path):
    print("Converting PDF pages to images...")
    images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)

    print("Running OCR on images...")
    text = ""
    for i, img in enumerate(images):
        print(f"OCR page {i+1}...")
        text += pytesseract.image_to_string(img, lang="eng") + "\n\n"

    return text


if __name__ == "__main__":
    text = ocr_extract_text(PDF_PATH)
    
    # save extracted text
    with open("resume_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("\nOCR DONE. Extracted text saved to resume_text.txt")
