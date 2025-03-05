import re
import torch
import spacy
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from functools import lru_cache
import concurrent.futures

class SuperSimpleLegalDocumentParser:
    def __init__(self, use_gpu=True):
        """
        Initialize a legal document simplifier focused on maximum readability.
        """
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"Running on: {self.device}")
        
        # Load language model - use a smaller model for better performance
        try:
            # Use the smaller and faster model
            self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
        except OSError:
            print("Downloading language model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
        
        # Load models lazily when first needed
        self._tokenizer = None
        self._legal_model = None
        self._simplifier = None
        
        # Precompile common regular expressions
        self._compile_regex_patterns()
        
        print("Ready to simplify documents!")
    
    def _compile_regex_patterns(self):
        """Precompile regex patterns for better performance"""
        self.regex_patterns = {
            'extra_spaces': re.compile(r'\s+'),
            'case_numbers': re.compile(r'(WRIT PETITION|CIVIL APPEAL|CRIMINAL CASE)[\s\w\d\(\)\.]+OF \d{4}', re.IGNORECASE),
            'court_headers': re.compile(r'IN THE .+ COURT OF .+?AT .+?[\r\n]', re.IGNORECASE),
            'matter_of': re.compile(r'IN THE MATTER OF:?', re.IGNORECASE),
            'petitioner': re.compile(r'\.\.\. PETITIONER[S]?', re.IGNORECASE),
            'respondent': re.compile(r'\.\.\. RESPONDENT[S]?', re.IGNORECASE),
            'appellant': re.compile(r'\.\.\. APPELLANT[S]?', re.IGNORECASE),
            'section_refs': re.compile(r'[Ss]ection \d+(\([a-z]\))? of the .+? Act,? \d{4}'),
        }
    
    @property
    def tokenizer(self):
        """Lazy-load the tokenizer"""
        if self._tokenizer is None:
            print("Loading InLegalBERT tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                "law-ai/InLegalBERT",
                use_fast=True  # Use the faster tokenizer implementation
            )
        return self._tokenizer
    
    @property
    def legal_model(self):
        """Lazy-load the legal model"""
        if self._legal_model is None:
            print("Loading InLegalBERT model...")
            self._legal_model = AutoModelForSequenceClassification.from_pretrained(
                "law-ai/InLegalBERT",
                num_labels=2,
                torchscript=True  # Enable TorchScript for faster inference
            ).to(self.device)
            # Set to evaluation mode for better performance
            self._legal_model.eval()
        return self._legal_model
    
    @property
    def simplifier(self):
        """Lazy-load the simplifier pipeline"""
        if self._simplifier is None:
            print("Loading text simplification model...")
            self._simplifier = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1,
                model_kwargs={"low_cpu_mem_usage": True}  # Reduce memory usage
            )
        return self._simplifier
    
    @lru_cache(maxsize=1024)
    def get_everyday_terms(self):
        """
        Maps legal terms to extremely simple everyday language (cached for performance)
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
        Super clean the document text using precompiled regex patterns
        """
        # Remove extra spaces and formatting
        text = self.regex_patterns['extra_spaces'].sub(' ', text.strip())
        
        # Apply all document cleaning patterns
        text = self.regex_patterns['case_numbers'].sub('', text)
        text = self.regex_patterns['court_headers'].sub('', text)
        text = self.regex_patterns['matter_of'].sub('', text)
        text = self.regex_patterns['petitioner'].sub('', text)
        text = self.regex_patterns['respondent'].sub('', text)
        text = self.regex_patterns['appellant'].sub('', text)
        text = self.regex_patterns['section_refs'].sub('relevant law', text)
        
        return text
    
    def break_into_simple_parts(self, text: str) -> List[str]:
        """
        Break text into very simple chunks
        """
        # Disable unnecessary pipeline components for better performance
        doc = self.nlp(text)
        
        # Extract sentences
        sentences = [sent.text for sent in doc.sents]
        
        # Group into small chunks (optimized for faster processing)
        chunks = []
        chunk_size = 3  # Adjust based on your needs
        
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i+chunk_size]
            if chunk:  # Check if chunk is not empty
                chunks.append(" ".join(chunk))
            
        return chunks

    @torch.no_grad()  # Disable gradient calculation for inference
    def super_simplify_text(self, text: str) -> str:
        """
        Create extremely simplified version of text with dynamic output length
        """
        # First replace all legal terms
        everyday_terms = self.get_everyday_terms()
        
        # More efficient term replacement
        for term, simple_term in everyday_terms.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, simple_term, text, flags=re.IGNORECASE)
        
        try:
            # Count input length (in tokens)
            input_length = len(text.split())
            
            # Dynamically set max_length to be approximately half the input length
            # but ensure it's at least 20 tokens and not less than min_length
            max_length = max(20, min(input_length // 2, 100))
            min_length = max(15, max_length // 2)
            
            # Very aggressive simplification with dynamic length
            with torch.inference_mode():  # Even more efficient than no_grad
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
        
        # Importance markers (precompiled set for faster lookup)
        importance_markers = {"seek", "request", "violat", "right", "direct", 
                             "order", "submit", "argu", "claim", "alleg"}
        
        key_points = []
        sentence_scores = []
        
        # Look for sentences with important markers and score them
        for sent in doc.sents:
            sentence = sent.text.strip()
            word_count = len(sentence.split())
            
            # Skip short or header-like sentences
            if word_count < 5:
                continue
            
            # Calculate importance score
            score = 0
            for marker in importance_markers:
                if marker in sentence.lower():
                    score += 1
            
            # Add a small bonus for moderate length sentences (not too short or long)
            if 10 <= word_count <= 30:
                score += 0.5
                
            sentence_scores.append((sentence, score))
        
        # Sort by importance score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3-5 sentences
        key_points = [s[0] for s in sentence_scores[:5] if s[1] > 0]
        
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
        
        # Using precompiled patterns for better performance
        patterns = {
            "court_petition": (re.compile(r'writ petition'), 
                             "court petition", 
                             "ask the court to fix a problem with the government or authorities"),
            
            "court_case": (re.compile(r'versus|vs\.'), re.compile(r'appellant|petitioner|plaintiff'),
                         "court case",
                         "resolve a dispute in court"),
            
            "agreement": (re.compile(r'agreement|contract|memorandum of understanding'), 
                        "agreement",
                        "establish rules between people or companies"),
            
            "notice": (re.compile(r'notice|notification'),
                     "legal notice",
                     "formally tell someone about a legal matter"),
                    
            "tax": (re.compile(r'income tax|tax'),
                  None,
                  "handle a tax-related issue"),
                  
            "property": (re.compile(r'property|land|premises'),
                       None,
                       "deal with property or land matters"),
                       
            "employment": (re.compile(r'employment|salary|compensation'),
                         None,
                         "handle an employment issue")
        }
        
        # Check for document types
        if patterns["court_petition"][0].search(text_lower):
            doc_type = patterns["court_petition"][1]
            purpose = patterns["court_petition"][2]
            
        elif (patterns["court_case"][0].search(text_lower) and 
              patterns["court_case"][1].search(text_lower)):
            doc_type = patterns["court_case"][1]
            purpose = patterns["court_case"][2]
            
        elif patterns["agreement"][0].search(text_lower):
            doc_type = patterns["agreement"][1]
            purpose = patterns["agreement"][2]
            
        elif patterns["notice"][0].search(text_lower):
            doc_type = patterns["notice"][1]
            purpose = patterns["notice"][2]
            
        # Check for specific issue types (overrides purpose only)
        if patterns["tax"][0].search(text_lower):
            purpose = patterns["tax"][2]
            
        elif patterns["property"][0].search(text_lower):
            purpose = patterns["property"][2]
            
        elif patterns["employment"][0].search(text_lower):
            purpose = patterns["employment"][2]
        
        return {
            "type": doc_type,
            "purpose": purpose
        }
    
    def simplify_chunks_parallel(self, chunks):
        """Process chunks in parallel for better performance"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            simplified_chunks = list(executor.map(self.super_simplify_text, chunks))
        return simplified_chunks
    
    def create_plain_english_summary(self, document: str):
        """
        Create a super simple plain English summary with improved formatting
        """
        # Clean up document
        cleaned_doc = self.clean_document(document)
        
        # Break into manageable chunks
        chunks = self.break_into_simple_parts(cleaned_doc)
        
        # For small documents, don't bother with parallelization
        if len(chunks) <= 2:
            simplified_chunks = [self.super_simplify_text(chunk) for chunk in chunks]
        else:
            # Simplify chunks in parallel
            simplified_chunks = self.simplify_chunks_parallel(chunks)
        
        # Format the simplified text with better sentence spacing and paragraph breaks
        formatted_simplified_text = self._format_simplified_text(simplified_chunks)
        
        # Extract key points and identify document purpose concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            key_points_future = executor.submit(self.extract_key_points, document)
            doc_purpose_future = executor.submit(self._identify_document_purpose, document)
            
            # Create a "bottom line" summary (extreme simplification) while waiting
            if len(formatted_simplified_text.split()) > 40:  # Only summarize if more than 40 words
                bottom_line = self.super_simplify_text(formatted_simplified_text)
            else:
                bottom_line = formatted_simplified_text
                
            # Get results from parallel tasks
            key_points = key_points_future.result()
            doc_purpose = doc_purpose_future.result()
        
        # Format key points as a bulleted list
        formatted_key_points = self._format_key_points(key_points)
        
        return {
            "what_this_means": bottom_line,
            "key_points": formatted_key_points,
            "document_type": doc_purpose["type"],
            "main_purpose": doc_purpose["purpose"],
            "simple_explanation": formatted_simplified_text
        }

    def _format_simplified_text(self, simplified_chunks):
        """
        Format simplified text for better readability
        """
        # Join chunks with proper spacing
        text = " ".join(simplified_chunks)
        
        # Fix spacing after periods, question marks, and exclamation points
        text = re.sub(r'([.?!])\s*([A-Z])', r'\1\n\n\2', text)
        
        # Ensure proper spacing after commas
        text = re.sub(r',\s*', ', ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Make sure bullet points are properly formatted
        text = re.sub(r'([•-])\s*', r'\n\1 ', text)
        
        # Enhance readability of legal phrases
        text = re.sub(r'(section|article|clause)\s+(\d+)', r'\1 \2', text, flags=re.IGNORECASE)
        
        return text.strip()

    def _format_key_points(self, key_points):
        """
        Format key points as a properly formatted list
        """
        if not key_points:
            return []
        
        formatted_points = []
        for i, point in enumerate(key_points):
            # Clean up the point
            point = point.strip()
            if not point.endswith(('.', '!', '?')):
                point += '.'
            
            # Capitalize first letter if needed
            if point and not point[0].isupper():
                point = point[0].upper() + point[1:]
                
            formatted_points.append(point)
        
        return formatted_points