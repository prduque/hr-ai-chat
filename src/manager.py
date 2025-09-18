"""
Vector Database Manager
Main class for managing vector database operations with hybrid search and Document versioning
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*telemetry.*')
warnings.filterwarnings('ignore', message='.*capture.*')

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import shutil
import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
from collections import Counter

from dataclasses import dataclass, asdict

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hr_chat.log'),
        logging.StreamHandler()
    ]
)
logging.getLogger("chromadb").setLevel(logging.ERROR)


def get_project_root() -> Path:
    """Get the project root directory (where this repo was cloned)"""
    # Start from this file's location and go up to find project root
    current = Path(__file__).parent
    
    # Go up until we find the project root indicators
    while current.parent != current:  # Not at filesystem root
        # Look for project indicators
        if any((current / indicator).exists() for indicator in 
               ['requirements.txt', '.git', 'README.md', 'setup.py']):
            return current
        current = current.parent
    
    # Fallback: assume we're in src/vector_db/ and go up 2 levels
    return Path(__file__).parent.parent.parent


@dataclass
class DocumentMetadata:
    """Complete document metadata"""
    original_filename: str
    processed_filename: str
    file_hash: str
    processing_date: datetime
    file_size: int
    document_type: str
    version: int
    is_active: bool = True
    replaces_version: Optional[int] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

class KeywordExtractor:
    """Keyword and synonym extractor for hybrid search"""
    
    def __init__(self):
        self.portuguese_synonyms = {
            'férias': ['vacation', 'holidays', 'descanso', 'ausência', 'folga'],
            'feriado': ['holiday', 'dia feriado', 'festa nacional', 'ponte'],
            'ponte': ['feriado prolongado', 'dia extra', 'long weekend'],
            'marcar': ['agendar', 'solicitar', 'pedir', 'requisitar', 'booking'],
            'aprovação': ['autorização', 'permissão', 'validação', 'approval'],
            'política': ['regras', 'normas', 'procedimentos', 'regulamento', 'policy'],
            'colaborador': ['funcionário', 'employee', 'worker', 'staff'],
            'empresa': ['company', 'organização', 'tourtailors'],
            'horário': ['schedule', 'working hours', 'tempo', 'hora'],
            'trabalho': ['work', 'job', 'emprego', 'atividade'],
            'direito': ['entitlement', 'benefit', 'privilégio'],
            'regime': ['sistema', 'modalidade', 'tipo'],
            'helpline': ['apoio', 'suporte', 'assistência', 'help'],
            'disponibilidade': ['availability', 'turno', 'serviço']
        }
        
        self.stopwords_pt = {
            'a', 'o', 'e', 'de', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'por',
            'que', 'se', 'na', 'no', 'os', 'as', 'dos', 'das', 'ao', 'à', 'pelos',
            'pelas', 'este', 'esta', 'isso', 'aquele', 'aquela', 'como', 'quando',
            'onde', 'qual', 'quais', 'porque', 'então', 'mas', 'ou', 'já', 'ainda',
            'também', 'só', 'mais', 'muito', 'bem', 'pode', 'podem', 'ter', 'tem',
            'é', 'são', 'foi', 'foram', 'estar', 'sendo', 'sido'
        }
        
        self.stopwords_en = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'can', 'could', 'should', 'would',
            'this', 'these', 'they', 'them', 'their', 'there', 'where', 'when',
            'what', 'why', 'how', 'who', 'which', 'have', 'had', 'do', 'does',
            'did', 'been', 'being'
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Remove stopwords
        keywords = [word for word in words 
                   if word not in self.stopwords_pt 
                   and word not in self.stopwords_en
                   and len(word) > 2]
        
        return keywords
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms"""
        expanded = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.portuguese_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded.append(expanded_query)
        
        # Remove duplicates while maintaining order
        seen = set()
        result = []
        for item in expanded:
            if item not in seen:
                seen.add(item)
                result.append(item)
        
        return result[:5]  # Limit to avoid overloading

class DocumentProcessor:
    """Document processor with enhanced multimodal capabilities"""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.pdf'}
        self.keyword_extractor = KeywordExtractor()
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from .txt file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Extracted text from {file_path}: {len(content)} characters")
            return content
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
            logger.warning(f"Used latin-1 encoding for {file_path}")
            return content
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using pdfplumber + OCR when necessary"""
        try:
            import pdfplumber  
            import pytesseract
            from pathlib import Path
            
            text_content = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # First try to extract text directly
                    text = page.extract_text()
                    
                    if text and text.strip():
                        page_text = f"--- Page {page_num + 1} ---\n{text}"
                        text_content.append(page_text)
                    else:
                        # No text - try OCR
                        try:
                            page_image = page.to_image(resolution=300)
                            pil_image = page_image.original
                            
                            ocr_text = pytesseract.image_to_string(
                                pil_image, 
                                lang='por+eng',
                                config='--oem 3 --psm 6'
                            )
                            
                            if ocr_text.strip():
                                page_text = f"--- Page {page_num + 1} (OCR) ---\n{ocr_text}"
                                text_content.append(page_text)
                                logger.info(f"OCR applied to page {page_num + 1}")
                                
                        except Exception as ocr_error:
                            logger.error(f"OCR error page {page_num + 1}: {ocr_error}")
            
            content = "\n\n".join(text_content)
            logger.info(f"Extracted text from PDF {file_path}: {len(content)} characters, {len(text_content)} pages")
            
            return content
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            # Fallback to PyPDF2
            try:
                import PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = []
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                    content = "\n\n".join(text_content)
                    logger.info(f"PyPDF2 fallback used for {file_path}: {len(content)} characters")
                    return content
            except Exception as e2:
                logger.error(f"Error also with PyPDF2 for {file_path}: {e2}")
                return ""
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file to detect changes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def smart_chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 400) -> List[Dict]:
        """Intelligent chunking with additional metadata"""
        if not text.strip():
            return []
        
        # Detect document sections
        sections = self._detect_sections(text)
        
        chunks = []
        chunk_index = 0
        
        for section_type, section_content in sections:
            if len(section_content) <= chunk_size:
                # Small section - single chunk
                keywords = self.keyword_extractor.extract_keywords(section_content)
                chunks.append({
                    'content': section_content.strip(),
                    'keywords': keywords,
                    'section_type': section_type,
                    'chunk_index': chunk_index,
                    'size': len(section_content)
                })
                chunk_index += 1
            else:
                # Large section - split with overlap
                section_chunks = self._split_with_overlap(section_content, chunk_size, overlap)
                for chunk_content in section_chunks:
                    keywords = self.keyword_extractor.extract_keywords(chunk_content)
                    chunks.append({
                        'content': chunk_content.strip(),
                        'keywords': keywords,
                        'section_type': section_type,
                        'chunk_index': chunk_index,
                        'size': len(chunk_content)
                    })
                    chunk_index += 1
        
        logger.info(f"Text split into {len(chunks)} intelligent chunks")
        return chunks
    
    def _detect_sections(self, text: str) -> List[Tuple[str, str]]:
        """Detect document sections"""
        sections = []
        
        # Patterns to detect sections
        section_patterns = [
            (r'^#+ .+', 'header'),
            (r'^\d+\.?\s+[A-Z][^.]+', 'numbered_section'),
            (r'^[A-Z][A-Z\s]+:?$', 'title'),
            (r'---.*---', 'separator')
        ]
        
        lines = text.split('\n')
        current_section = []
        current_type = 'content'
        
        for line in lines:
            line_type = 'content'
            
            # Check if line matches any pattern
            for pattern, section_type in section_patterns:
                if re.match(pattern, line.strip()):
                    line_type = section_type
                    break
            
            # If section type changed, finalize previous one
            if line_type != current_type and current_section:
                sections.append((current_type, '\n'.join(current_section)))
                current_section = [line]
                current_type = line_type
            else:
                current_section.append(line)
        
        # Add last section
        if current_section:
            sections.append((current_type, '\n'.join(current_section)))
        
        return sections
    
    def _split_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text with intelligent overlap"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            if end < text_length:
                # Look for natural break
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + chunk_size // 2:
                    end = last_period + 1
                else:
                    last_newline = text.rfind('\n', start, end)
                    if last_newline != -1 and last_newline > start + chunk_size // 2:
                        end = last_newline
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + chunk_size - overlap, end)
        
        return chunks

class VectorDatabaseManager:
    """Main vector database manager with Hybrid Search"""
    
    def __init__(self, kb_path: str = None, processed_path: str = None, 
                 db_path: str = None, registry_file: str = None):
        # Get project root and set default paths
        self.project_root = get_project_root()
        
        # Use provided paths or defaults relative to project root
        self.kb_path = Path(kb_path) if kb_path else self.project_root / "data" / "kb"
        self.processed_path = Path(processed_path) if processed_path else self.project_root / "data" / "processed"
        self.db_path = Path(db_path) if db_path else self.project_root / "data" / "vector_db"
        self.registry_file = Path(registry_file) if registry_file else self.project_root / "data" / "document_registry.json"
        
        # Create directories if they don't exist
        self.kb_path.mkdir(exist_ok=True)
        self.processed_path.mkdir(exist_ok=True, parents=True)
        self.db_path.mkdir(exist_ok=True, parents=True)

        self.document_processor = DocumentProcessor()
        self.keyword_extractor = KeywordExtractor()
        
        self._is_initialized = False
        
    def _ensure_initialized(self):
        if self._is_initialized:
            return

        print("🚀 Starting HR Vectorization System with Hybrid Search")
        print("=" * 60)

        import chromadb
        from chromadb.config import Settings
        from chromadb.utils import embedding_functions
        from sentence_transformers import SentenceTransformer
        
        # Initialize components
        # Create specific embedding function
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,  
                allow_reset=True,
                is_persistent=True,
            )
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="hr_documents_hybrid",
            metadata={
                "description": "HR documents with hybrid semantic + keyword search",
                "embedding_dimensions": self.embedding_model.get_sentence_embedding_dimension()
            },
            embedding_function=sentence_transformer_ef
        )        
        
        # Load document registry
        self.document_registry = self.load_document_registry()
        
        self._is_initialized = True
        
        logger.info(f"VectorDatabaseManager initialized: {len(self.document_registry)} documents in registry")
        
    def prepare_for_searches(self):
        self._ensure_initialized()
    
    def load_document_registry(self) -> Dict:
        """Load processed documents registry"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                    
                # Convert date strings back to datetime
                for doc_id, doc_info in registry_data.items():
                    if 'processing_date' in doc_info:
                        doc_info['processing_date'] = datetime.fromisoformat(doc_info['processing_date'])
                    if 'valid_from' in doc_info and doc_info['valid_from']:
                        doc_info['valid_from'] = datetime.fromisoformat(doc_info['valid_from'])
                    if 'valid_to' in doc_info and doc_info['valid_to']:
                        doc_info['valid_to'] = datetime.fromisoformat(doc_info['valid_to'])
                
                return registry_data
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                return {}
        return {}
    
    def save_document_registry(self):
        """Save documents registry"""
        try:
            registry_to_save = {}
            for doc_id, doc_info in self.document_registry.items():
                doc_copy = doc_info.copy()
                if 'processing_date' in doc_copy and isinstance(doc_copy['processing_date'], datetime):
                    doc_copy['processing_date'] = doc_copy['processing_date'].isoformat()
                if 'valid_from' in doc_copy and doc_copy['valid_from'] and isinstance(doc_copy['valid_from'], datetime):
                    doc_copy['valid_from'] = doc_copy['valid_from'].isoformat()
                if 'valid_to' in doc_copy and doc_copy['valid_to'] and isinstance(doc_copy['valid_to'], datetime):
                    doc_copy['valid_to'] = doc_copy['valid_to'].isoformat()
                registry_to_save[doc_id] = doc_copy
            
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_to_save, f, indent=2, ensure_ascii=False)
            logger.info("Documents registry saved")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def hybrid_search(self, query: str, n_results: int = 8, 
                     semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> Dict:
        """Hybrid search combining semantic and keyword search"""
        print(f"🔍 Hybrid Search: '{query}' (sem:{semantic_weight}, kw:{keyword_weight})")
        
        # 1. Semantic search (ChromaDB)
        semantic_results = self._semantic_search(query, n_results * 2)
        
        # 2. Keyword search
        keyword_results = self._keyword_search(query, n_results * 2)
        
        # 3. Combine and re-rank results
        combined_results = self._combine_and_rerank(
            semantic_results, keyword_results, 
            semantic_weight, keyword_weight, n_results
        )
        
        return combined_results
    
    def _semantic_search(self, query: str, n_results: int) -> Dict:
        """Semantic search using embeddings"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"is_active": True}
            )
            
            return {
                'query': query,
                'results': results,
                'total_found': len(results['ids'][0]) if results['ids'] else 0,
                'search_type': 'semantic'
            }
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {'query': query, 'results': [], 'total_found': 0, 'error': str(e)}
    
    def _keyword_search(self, query: str, n_results: int) -> Dict:
        """Keyword search with synonym expansion"""
        # Expand query with synonyms
        expanded_queries = self.keyword_extractor.expand_query(query)
        
        # Extract keywords from query
        query_keywords = set(self.keyword_extractor.extract_keywords(query))
        
        # Search all expanded queries
        all_results = {}  # chunk_id -> (score, metadata, document)
        
        for exp_query in expanded_queries:
            exp_keywords = set(self.keyword_extractor.extract_keywords(exp_query))
            
            # Search chunks containing these keywords
            try:
                # Use semantic search as base and filter by keywords
                results = self.collection.query(
                    query_texts=[exp_query],
                    n_results=n_results * 3,
                    where={"is_active": True}
                )
                
                if results['ids'] and results['ids'][0]:
                    for i, (doc_id, doc, meta, distance) in enumerate(zip(
                        results['ids'][0],
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    )):
                        # Calculate score based on keywords
                        doc_keywords_str = meta.get('keywords', '')
                        doc_keywords = set(doc_keywords_str.split()) if doc_keywords_str else set()
                        keyword_overlap = len(query_keywords.intersection(doc_keywords))
                        
                        if keyword_overlap > 0:
                            # Hybrid score: keyword overlap + semantic proximity
                            keyword_score = keyword_overlap / len(query_keywords)
                            semantic_score = 1.0 - distance  # Convert distance to score
                            combined_score = (keyword_score * 0.6) + (semantic_score * 0.4)
                            
                            if doc_id not in all_results or combined_score > all_results[doc_id][0]:
                                all_results[doc_id] = (combined_score, meta, doc, distance)
                
            except Exception as e:
                logger.error(f"Error in keyword search for '{exp_query}': {e}")
        
        # Sort by score and limit results
        sorted_results = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)
        top_results = sorted_results[:n_results]
        
        # Format as ChromaDB structure
        if top_results:
            ids = [[item[0] for item in top_results]]
            metadatas = [[item[1][1] for item in top_results]]
            documents = [[item[1][2] for item in top_results]]
            distances = [[1.0 - item[1][0] for item in top_results]]  # Convert score back to distance
            
            formatted_results = {
                'ids': ids,
                'metadatas': metadatas,
                'documents': documents,
                'distances': distances
            }
        else:
            formatted_results = {'ids': [[]], 'metadatas': [[]], 'documents': [[]], 'distances': [[]]}
        
        return {
            'query': query,
            'results': formatted_results,
            'total_found': len(top_results),
            'search_type': 'keyword',
            'expanded_queries': expanded_queries
        }
    
    def _combine_and_rerank(self, semantic_results: Dict, keyword_results: Dict, 
                           semantic_weight: float, keyword_weight: float, n_results: int) -> Dict:
        """Combine semantic and keyword results"""
        combined_scores = {}  # chunk_id -> (final_score, metadata, document)
        
        # Process semantic results
        if semantic_results['total_found'] > 0:
            for i, (doc_id, doc, meta, distance) in enumerate(zip(
                semantic_results['results']['ids'][0],
                semantic_results['results']['documents'][0],
                semantic_results['results']['metadatas'][0],
                semantic_results['results']['distances'][0]
            )):
                semantic_score = 1.0 - distance
                combined_scores[doc_id] = (semantic_score * semantic_weight, meta, doc, distance)
        
        # Process keyword results
        if keyword_results['total_found'] > 0:
            for i, (doc_id, doc, meta, distance) in enumerate(zip(
                keyword_results['results']['ids'][0],
                keyword_results['results']['documents'][0],
                keyword_results['results']['metadatas'][0],
                keyword_results['results']['distances'][0]
            )):
                keyword_score = 1.0 - distance
                
                if doc_id in combined_scores:
                    # Combine scores
                    current_score = combined_scores[doc_id][0]
                    new_score = current_score + (keyword_score * keyword_weight)
                    combined_scores[doc_id] = (new_score, meta, doc, 1.0 - new_score)
                else:
                    # Only keyword score
                    combined_scores[doc_id] = (keyword_score * keyword_weight, meta, doc, distance)
        
        # Sort by final score and limit
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1][0], reverse=True)
        top_results = sorted_results[:n_results]
        
        print(f"   ✅ Combined: {len(top_results)} final results")
        
        # Format final result
        if top_results:
            ids = [[item[0] for item in top_results]]
            metadatas = [[item[1][1] for item in top_results]]
            documents = [[item[1][2] for item in top_results]]
            distances = [[item[1][3] for item in top_results]]
            
            final_results = {
                'ids': ids,
                'metadatas': metadatas,
                'documents': documents,
                'distances': distances
            }
        else:
            final_results = {'ids': [[]], 'metadatas': [[]], 'documents': [[]], 'distances': [[]]}
        
        return {
            'query': semantic_results['query'],
            'results': final_results,
            'total_found': len(top_results),
            'search_type': 'hybrid',
            'semantic_results': semantic_results['total_found'],
            'keyword_results': keyword_results['total_found'],
            'expanded_queries': keyword_results.get('expanded_queries', [])
        }
    
    def search_documents(self, query: str, n_results: int = 5, 
                        include_inactive: bool = False,
                        search_type: str = 'hybrid',
                        max_distance: Optional[float] = None) -> Dict:
        """Main search interface (maintains compatibility)"""
        if search_type == 'hybrid':
            return self.hybrid_search(query, n_results)
        elif search_type == 'semantic':
            return self._semantic_search(query, n_results)
        elif search_type == 'keyword':
            return self._keyword_search(query, n_results)
        else:
            # Fallback to hybrid
            return self.hybrid_search(query, n_results)
    
    def process_single_document(self, file_path: Path) -> bool:
        """Process single document with intelligent chunking"""
        try:
            original_filename = file_path.name
            file_hash = self.document_processor.calculate_file_hash(str(file_path))
            file_size = file_path.stat().st_size
            processing_date = datetime.now()
            
            logger.info(f"Processing: {original_filename}")
            
            # Check if document already exists
            existing_doc = self.find_existing_document(original_filename, file_hash)
            
            if existing_doc and existing_doc['file_hash'] == file_hash:
                logger.info(f"Document {original_filename} already exists with same content. Skipping.")
                return False
            
            # Extract text based on extension
            if file_path.suffix.lower() == '.txt':
                text_content = self.document_processor.extract_text_from_txt(str(file_path))
                document_type = 'text'
            elif file_path.suffix.lower() == '.pdf':
                text_content = self.document_processor.extract_text_from_pdf(str(file_path))
                document_type = 'pdf'
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return False
            
            if not text_content.strip():
                logger.warning(f"No text extracted from {original_filename}")
                return False
            
            # Intelligent chunking
            chunks_data = self.document_processor.smart_chunk_text(text_content)
            if not chunks_data:
                logger.warning(f"No chunks generated for {original_filename}")
                return False
            
            # Extract only content for embeddings
            chunks_content = [chunk['content'] for chunk in chunks_data]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks_content).tolist()
            
            # Determine version
            version = self.get_next_version_number(original_filename)
            is_new_version = version > 1
            
            # Generate processed filename and move
            processed_filename = self.generate_processed_filename(original_filename)
            processed_file_path = self.processed_path / processed_filename
            shutil.move(str(file_path), str(processed_file_path))
            
            # Create unique ID for this document version
            document_id = f"{original_filename}_v{version}_{processing_date.strftime('%Y%m%d%H%M%S')}"
            
            # Prepare base metadata
            base_metadata = {
                'document_id': document_id,
                'original_filename': original_filename,
                'processed_filename': processed_filename,
                'file_hash': file_hash,
                'processing_date': processing_date.isoformat(),
                'file_size': file_size,
                'document_type': document_type,
                'version': version,
                'is_active': True,
                'valid_from': processing_date.isoformat(),
                'total_chunks': len(chunks_data)
            }
            
            if is_new_version:
                base_metadata['replaces_version'] = version - 1
                self.deactivate_previous_versions(original_filename, version)
            
            # Prepare data for ChromaDB with enriched metadata
            chunk_ids = []
            chunk_metadatas = []
            
            for i, chunk_data in enumerate(chunks_data):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_data['chunk_index'],
                    'keywords': ' '.join(chunk_data['keywords']) if chunk_data['keywords'] else '',  # Convert list to string
                    'section_type': chunk_data['section_type'],
                    'chunk_size': chunk_data['size'],
                    'chunk_text_preview': chunk_data['content'][:100] + "..." if len(chunk_data['content']) > 100 else chunk_data['content']
                })
                
                chunk_ids.append(chunk_id)
                chunk_metadatas.append(chunk_metadata)
            
            # Add to vector database
            self.collection.add(
                documents=chunks_content,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            # Register in document registry
            self.document_registry[document_id] = {
                'original_filename': original_filename,
                'processed_filename': processed_filename,
                'file_hash': file_hash,
                'processing_date': processing_date,
                'file_size': file_size,
                'document_type': document_type,
                'version': version,
                'is_active': True,
                'valid_from': processing_date,
                'valid_to': None,
                'replaces_version': version - 1 if is_new_version else None,
                'total_chunks': len(chunks_data)
            }
            
            action = "New version added" if is_new_version else "Document added"
            logger.info(f"{action}: {original_filename} -> {document_id} ({len(chunks_data)} chunks)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False

    # Helper methods maintained from original version
    def generate_processed_filename(self, original_filename: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        return f"{timestamp}_{original_filename}"
    
    def find_existing_document(self, original_filename: str, file_hash: str) -> Optional[Dict]:
        for doc_id, doc_info in self.document_registry.items():
            if doc_info['original_filename'] == original_filename:
                return doc_info
        for doc_id, doc_info in self.document_registry.items():
            if doc_info['file_hash'] == file_hash:
                return doc_info
        return None
    
    def get_next_version_number(self, original_filename: str) -> int:
        max_version = 0
        for doc_info in self.document_registry.values():
            if doc_info['original_filename'] == original_filename:
                max_version = max(max_version, doc_info['version'])
        return max_version + 1
    
    def deactivate_previous_versions(self, original_filename: str, current_version: int):
        current_time = datetime.now()
        for doc_id, doc_info in self.document_registry.items():
            if (doc_info['original_filename'] == original_filename and 
                doc_info['version'] < current_version and 
                doc_info['is_active']):
                
                doc_info['is_active'] = False
                doc_info['valid_to'] = current_time
                
                try:
                    results = self.collection.get(where={"document_id": doc_id})
                    if results['ids']:
                        updated_metadatas = []
                        for metadata in results['metadatas']:
                            metadata['is_active'] = False
                            metadata['valid_to'] = current_time.isoformat()
                            updated_metadatas.append(metadata)
                        
                        self.collection.update(
                            ids=results['ids'],
                            metadatas=updated_metadatas
                        )
                except Exception as e:
                    logger.error(f"Error deactivating version {doc_id}: {e}")
                
                logger.info(f"Version deactivated: {doc_id}")
    
    def has_new_documents(self) -> bool:
        supported_files = []
        for ext in self.document_processor.supported_extensions:
            supported_files.extend(self.kb_path.glob(f"*{ext}"))
        
        found_docs = len(supported_files) > 0
        if found_docs:
            logger.info(f"Found {len(supported_files)} files to process")
        else:
            logger.info("No new documents to process")
        return found_docs
    
    def process_all_documents(self) -> Dict[str, int]:
        self._ensure_initialized()
        stats = {'processed': 0, 'skipped': 0, 'errors': 0}
        
        supported_files = []
        for ext in self.document_processor.supported_extensions:
            supported_files.extend(self.kb_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(supported_files)} files to process")
        
        for file_path in supported_files:
            try:
                if self.process_single_document(file_path):
                    stats['processed'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                stats['errors'] += 1
        
        self.save_document_registry()
        logger.info(f"Processing completed: {stats}")
        return stats
    
    def get_document_info(self, original_filename: str = None, 
                         document_id: str = None, 
                         include_inactive: bool = False) -> List[Dict]:
        results = []
        for doc_id, doc_info in self.document_registry.items():
            if document_id and doc_id != document_id:
                continue
            if original_filename and doc_info['original_filename'] != original_filename:
                continue
            if not include_inactive and not doc_info['is_active']:
                continue
            
            results.append({'document_id': doc_id, **doc_info})
        
        results.sort(key=lambda x: x['processing_date'], reverse=True)
        return results
    
    def get_database_stats(self) -> Dict:
        self._ensure_initialized()
        try:
            total_documents = len(self.document_registry)
            active_documents = sum(1 for doc in self.document_registry.values() if doc['is_active'])
            collection_count = self.collection.count()
            
            type_stats = {}
            for doc_info in self.document_registry.values():
                doc_type = doc_info['document_type']
                if doc_type not in type_stats:
                    type_stats[doc_type] = {'total': 0, 'active': 0}
                type_stats[doc_type]['total'] += 1
                if doc_info['is_active']:
                    type_stats[doc_type]['active'] += 1
            
            return {
                'total_documents': total_documents,
                'active_documents': active_documents,
                'total_chunks': collection_count,
                'type_statistics': type_stats,
                'registry_file': str(self.registry_file),
                'database_path': str(self.db_path),
                'embedding_model': 'paraphrase-multilingual-mpnet-base-v2',
                'search_type': 'hybrid'
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    def clear_database(self, confirm: bool = False) -> Dict[str, str]:
        self._ensure_initialized()
        if not confirm:
            return {
                'status': 'cancelled',
                'message': 'Operation cancelled. Use clear_database(confirm=True) to confirm.'
            }
        
        try:
            stats_before = self.get_database_stats()
            
            try:
                self.chroma_client.delete_collection(name="hr_documents_hybrid")
                logger.info("ChromaDB collection deleted")
            except Exception as e:
                logger.warning(f"Error deleting collection: {e}")
            
            self.collection = self.chroma_client.get_or_create_collection(
                name="hr_documents_hybrid",
                metadata={"description": "HR documents with hybrid search"}
            )
            
            self.document_registry = {}
            self.save_document_registry()
            
            if self.processed_path.exists():
                for file_path in self.processed_path.glob("*"):
                    try:
                        file_path.unlink()
                        logger.info(f"Processed file removed: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Error removing {file_path}: {e}")
            
            logger.info("Database cleared successfully")
            
            return {
                'status': 'success',
                'message': 'Database cleared successfully',
                'documents_removed': stats_before.get('total_documents', 0),
                'chunks_removed': stats_before.get('total_chunks', 0)
            }
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return {'status': 'error', 'message': f'Error clearing database: {str(e)}'}

