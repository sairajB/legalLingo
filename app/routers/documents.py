import os
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

from app.services.document_processor import DocumentProcessor
from app.models.simplifier import SuperSimpleLegalDocumentParser

# Create router
router = APIRouter()

# Initialize the simplifier and processor
simplifier = SuperSimpleLegalDocumentParser(use_gpu=False)
document_processor = DocumentProcessor()

# Create upload directory if it doesn't exist
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Response model without what_this_means and main_purpose
class SimplificationResponse(BaseModel):
    document_type: str
    key_points: List[str]
    simple_explanation: str

@router.post("/simplify/", response_model=SimplificationResponse)
async def simplify_document(
    file: UploadFile = File(...),
    language: Optional[str] = Form("english")
):
    """
    Upload and simplify a legal document
    """
    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Please upload {', '.join(allowed_extensions)}"
        )
    
    # Save the uploaded file temporarily
    temp_file_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text from file
        document_text = document_processor.extract_text_from_file(temp_file_path, file_ext)
        
        if not document_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the document"
            )
        
        # Process document
        result = simplifier.create_plain_english_summary(document_text)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return result
        
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")