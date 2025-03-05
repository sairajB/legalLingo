import fitz  # PyMuPDF
import docx

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor"""
        pass

    def extract_text_from_pdf(self, file_path):
        """Extract text from a PDF file"""
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_docx(self, file_path):
        """Extract text from a DOCX file"""
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return ""

    def extract_text_from_file(self, file_path, file_extension):
        """Extract text based on file type"""
        if file_extension.lower() == ".pdf":
            return self.extract_text_from_pdf(file_path)
        elif file_extension.lower() == ".docx":
            return self.extract_text_from_docx(file_path)
        elif file_extension.lower() == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return ""