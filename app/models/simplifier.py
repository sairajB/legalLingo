import re
import torch
import spacy
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class SuperSimpleLegalDocumentParser:
    def __init__(self, use_gpu=True):
        """
        Initialize a legal document simplifier focused on maximum readability.
        """
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"Running on: {self.device}")
        
        # Load language processing tools
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading language model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
        
        # Load specialized legal model
        print("Loading InLegalBERT model...")
        self.model_name = "law-ai/InLegalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.legal_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)
        
        # Text simplification model
        print("Loading text simplification model...")
        self.simplifier = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=0 if self.device == "cuda" else -1
        )
        
        print("Ready to simplify documents!")
    
    def get_everyday_terms(self):
        """
        Maps legal terms to extremely simple everyday language
        """
        return {
            # Court terms
            "writ petition": "formal request to court",
            "petitioner": "person making the request",
            "respondent": "person who needs to answer",
            "versus": "against",
            "appellant": "person appealing",
            "affidavit": "sworn statement",
            "plaintiff": "person who started the case",
            "defendant": "person being sued",
            "jurisdiction": "court's power to decide",
            "injunction": "court order to stop something",
            
            # Common legal phrases
            "pursuant to": "according to",
            "hereinafter": "from now on",
            "aforementioned": "mentioned earlier",
            "notwithstanding": "despite",
            "in lieu of": "instead of",
            "inter alia": "among other things",
            "prima facie": "at first look",
            "suo motu": "on its own",
            "ultra vires": "beyond legal power",
            "caveat emptor": "buyer beware",
            
            # Indian legal terms
            "vakalatnama": "lawyer permission document",
            "lok adalat": "people's court",
            "in-camera proceeding": "private court meeting",
            "ex parte": "with only one side present",
            
            # Legislative references
            "section": "part",
            "article": "section",
            "sub-section": "smaller part",
            "clause": "point",
            "provision": "rule",
            "statute": "written law",
            
            # Contract terms
            "indemnify": "protect from loss",
            "covenant": "promise",
            "consideration": "payment or benefit",
            "arbitration": "settling disputes without court",
            "therein": "in that",
            "thereof": "of that",
            "hereunder": "under this document",
            "hereof": "of this document",
            "whereby": "by which",
            "wherefrom": "from which",
            "whereupon": "after which"
        }
    
    def clean_document(self, text: str) -> str:
        """
        Super clean the document text
        """
        # Remove extra spaces and formatting
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove case numbers and headers
        text = re.sub(r'(WRIT PETITION|CIVIL APPEAL|CRIMINAL CASE)[\s\w\d\(\)\.]+OF \d{4}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'IN THE .+ COURT OF .+?AT .+?[\r\n]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'IN THE MATTER OF:?', '', text, flags=re.IGNORECASE)
        
        # Remove party designations
        text = re.sub(r'\.\.\. PETITIONER[S]?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\.\.\. RESPONDENT[S]?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\.\.\. APPELLANT[S]?', '', text, flags=re.IGNORECASE)
        
        # Remove section references
        text = re.sub(r'[Ss]ection \d+(\([a-z]\))? of the .+? Act,? \d{4}', 'relevant law', text)
        
        return text
    
    def break_into_simple_parts(self, text: str) -> List[str]:
        """
        Break text into very simple chunks
        """
        doc = self.nlp(text)
        
        # Extract sentences
        sentences = [sent.text for sent in doc.sents]
        
        # Group into small chunks (3-5 sentences)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            # Keep chunks small for better simplification
            if len(current_chunk) >= 3:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        # Add any remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def super_simplify_text(self, text: str) -> str:
        """
        Create extremely simplified version of text with dynamic output length
        """
        # First replace all legal terms
        for term, simple_term in self.get_everyday_terms().items():
            text = re.sub(rf'\b{re.escape(term)}\b', simple_term, text, flags=re.IGNORECASE)
        
        try:
            # Count input length (in tokens)
            input_length = len(text.split())
            
            # Dynamically set max_length to be approximately half the input length
            # but ensure it's at least 20 tokens and not less than min_length
            max_length = max(20, min(input_length // 2, 100))
            min_length = max(15, max_length // 2)
            
            # Very aggressive simplification with dynamic length
            result = self.simplifier(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            
            return result
        except Exception as e:
            print(f"Error during simplification: {e}")
            return text
    
    def extract_key_points(self, text: str) -> List[str]:
        """
        Pull out only the most important points
        """
        doc = self.nlp(text)
        
        key_points = []
        
        # Look for sentences with important markers
        for sent in doc.sents:
            sentence = sent.text.strip()
            
            # Skip short or header-like sentences
            if len(sentence.split()) < 5:
                continue
                
            # Extract sentences that likely contain key points
            importance_markers = [
                "seek", "request", "violat", "right", "direct", 
                "order", "submit", "argu", "claim", "alleg"
            ]
            
            if any(marker in sentence.lower() for marker in importance_markers):
                key_points.append(sentence)
        
        # If we couldn't find key sentences, take the longest sentences
        if not key_points:
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 5]
            sentences.sort(key=lambda s: len(s.split()), reverse=True)
            key_points = sentences[:3]  # Get top 3 longest sentences
            
        return key_points
    
    def _identify_document_purpose(self, text: str):
        """
        Identify the document type and purpose in simple terms
        """
        text_lower = text.lower()
        
        # Identify document type
        doc_type = "legal document"
        purpose = "handle a legal matter"
        
        # Court case types
        if re.search(r'writ petition', text_lower):
            doc_type = "court petition"
            purpose = "ask the court to fix a problem with the government or authorities"
            
        elif re.search(r'versus|vs\.', text_lower) and re.search(r'appellant|petitioner|plaintiff', text_lower):
            doc_type = "court case"
            purpose = "resolve a dispute in court"
        
        # Agreements
        elif re.search(r'agreement|contract|memorandum of understanding', text_lower):
            doc_type = "agreement"
            purpose = "establish rules between people or companies"
            
        # Notice
        elif re.search(r'notice|notification', text_lower):
            doc_type = "legal notice"
            purpose = "formally tell someone about a legal matter"
            
        # More specific identification based on content
        if re.search(r'income tax|tax', text_lower):
            purpose = "handle a tax-related issue"
            
        elif re.search(r'property|land|premises', text_lower):
            purpose = "deal with property or land matters"
            
        elif re.search(r'employment|salary|compensation', text_lower):
            purpose = "handle an employment issue"
        
        return {
            "type": doc_type,
            "purpose": purpose
        }
    
    def create_plain_english_summary(self, document: str):
        """
        Create a super simple plain English summary
        """
        # Clean up document
        cleaned_doc = self.clean_document(document)
        
        # Break into manageable chunks
        chunks = self.break_into_simple_parts(cleaned_doc)
        
        # Simplify each chunk
        simplified_chunks = [self.super_simplify_text(chunk) for chunk in chunks]
        simplified_text = " ".join(simplified_chunks)
        
        # Extract key points
        key_points = self.extract_key_points(document)
        
        # Create a "bottom line" summary (extreme simplification)
        if len(simplified_text.split()) > 40:  # Only summarize if more than 40 words
            bottom_line = self.super_simplify_text(simplified_text)
        else:
            bottom_line = simplified_text
            
        # Create very simple explanation of document purpose
        doc_purpose = self._identify_document_purpose(document)
        
        return {
            "what_this_means": bottom_line,
            "key_points": key_points,
            "document_type": doc_purpose["type"],
            "main_purpose": doc_purpose["purpose"],
            "simple_explanation": simplified_text
        }