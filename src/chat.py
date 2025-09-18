"""
Hybrid RAG System
Main class for querying the vector database and getting answer from LLM
"""

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
from contextlib import redirect_stderr
from io import StringIO

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# Import our system
from .manager import VectorDatabaseManager

class HybridHRRAGSystem:
    """Advanced RAG System with Hybrid Search for Human Resources"""
    
    def __init__(self, 
                 llm_provider: str = "ollama",
                 model_name: str = "llama3.1:8b",
                 temperature: float = 0.2):
        
        print("🚀 Initializing RAG System with Hybrid Search...")
        
        # Initialize vector database with hybrid search
        print("📚 Loading hybrid database...")
        self.vector_db = VectorDatabaseManager()
        self.vector_db.prepare_for_searches()
        
        # Configure LLM
        print(f"🤖 Configuring LLM: {llm_provider} - {model_name}")
        self.llm = self._setup_llm(llm_provider, model_name, temperature)
        
        if self.llm is None:
            raise Exception("❌ LLM was not initialized correctly!")
        
        print("✅ LLM configured successfully")
        
        # Configure conversational memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=3000
        )
        
        # Prompts optimized for hybrid search
        self.system_prompts = self._setup_prompts()
        
        # Search configurations
        self.search_config = {
            'semantic_weight': 0.6,
            'keyword_weight': 0.4,
            'n_results': 8,
            'use_query_expansion': True
        }
        
        # Metrics
        self.conversation_stats = {
            'queries_processed': 0,
            'hybrid_searches': 0,
            'semantic_searches': 0,
            'keyword_searches': 0,
            'avg_retrieval_time': 0.0,
            'avg_context_size': 0.0
        }
        
        print("✅ Hybrid RAG System initialized successfully!")
    
    def _setup_llm(self, provider: str, model_name: str, temperature: float):
        """Configure LLM with robust checks"""
        if provider == "ollama":
            try:
                print("🔗 Testing Ollama connection...")
                
                llm = Ollama(
                    model=model_name,
                    base_url="http://localhost:11434",
                    temperature=temperature,
                    num_predict=3000,
                    num_ctx=6000,
                    num_gpu=1,
                    num_thread=14,
                )
                
                # Connectivity test
                test_response = llm.invoke("Respond: 'System functional'")
                print(f"✅ Ollama Test: {test_response[:50]}...")
                
                return llm
                
            except Exception as e:
                print(f"❌ Error connecting to Ollama: {e}")
                return None
                
        elif provider == "openai":
            try:
                return ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            except Exception as e:
                print(f"❌ Error configuring OpenAI: {e}")
                return None
        else:
            raise ValueError(f"Provider {provider} not supported")
    
    def _setup_prompts(self) -> Dict[str, str]:
        """Prompts optimized for hybrid search"""
        return {
            'hr_assistant': """You are an assistant specialized in TourTailors HR policies.

INSTRUCTIONS:
1. Carefully analyze ALL sections of the provided documents
2. Use document information to respond clearly and specifically
3. If you find specific rules or policies, cite them directly
4. Always indicate the source (document and section) of information
5. If there are multiple related rules, mention all relevant ones
6. For questions not covered by documents, clearly state you don't have that information
7. An employee is not a customer, unless he explicitly says the trip was provided by TourTailors

COMPANY DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

EMPLOYEE QUESTION: {question}

RESPONSE BASED ON COMPANY POLICIES, and in the same language as the employee question:""",

            'policy_analysis': """You are an expert in corporate policy analysis.

TASK: Analyze the provided policies and answer the question in detail.

DOCUMENTS AND POLICIES:
{context}

QUESTION: {question}

Provide an analysis that:
1. Identifies all relevant policies
2. Explains how they apply to the situation
3. Highlights any requirements or limitations
4. Cites specific sources
5. An employee is not a customer, unless he explicitly says the trip was provided by TourTailors

DETAILED ANALYSIS:""",

            'quick_answer': """Direct response based on TourTailors policies:

DOCUMENTS: {context}
QUESTION: {question}

Concise response (maximum 2 paragraphs) with source citation, and in the same language as the user question:"""
        }
    
    def intelligent_retrieval(self, query: str, search_strategy: str = 'auto') -> Tuple[List[str], List[Dict], Dict]:
        """Intelligent retrieval system with multiple strategies"""
        start_time = datetime.now()
        
        print(f"🔍 Intelligent retrieval: '{query}' (strategy: {search_strategy})")
        
        # Determine strategy automatically if necessary
        if search_strategy == 'auto':
            search_strategy = self._determine_search_strategy(query)
        
        # Execute search based on strategy
        if search_strategy == 'hybrid':
            search_results = self.vector_db.hybrid_search(
                query, 
                n_results=self.search_config['n_results'],
                semantic_weight=self.search_config['semantic_weight'],
                keyword_weight=self.search_config['keyword_weight']
            )
            self.conversation_stats['hybrid_searches'] += 1
            
        elif search_strategy == 'semantic':
            search_results = self.vector_db.search_documents(
                query, 
                n_results=self.search_config['n_results'],
                search_type='semantic'
            )
            self.conversation_stats['semantic_searches'] += 1
            
        elif search_strategy == 'keyword':
            search_results = self.vector_db.search_documents(
                query, 
                n_results=self.search_config['n_results'],
                search_type='keyword'
            )
            self.conversation_stats['keyword_searches'] += 1
        
        # Process results
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        if search_results['total_found'] > 0:
            documents = search_results['results']['documents'][0]
            metadatas = search_results['results']['metadatas'][0]
            
            # Filter and rank by relevance
            filtered_docs, filtered_metas = self._filter_and_rank_results(
                documents, metadatas, query
            )
            
            print(f"   ✅ Found {len(filtered_docs)} relevant chunks")
            for i, meta in enumerate(filtered_metas[:3]):
                filename = meta.get('original_filename', 'unknown')
                section = meta.get('section_type', 'content')
                chunk_idx = meta.get('chunk_index', '?')
                print(f"      {i+1}. {filename} (section: {section}, chunk: {chunk_idx})")
            
            # Update statistics
            self._update_stats(retrieval_time, len(filtered_docs), search_strategy)
            
            return filtered_docs, filtered_metas, search_results
        else:
            print("❌ No relevant documents found")
            return [], [], search_results
    
    def _determine_search_strategy(self, query: str) -> str:
        """Determine best search strategy based on query"""
        query_lower = query.lower()
        
        # Specific queries that benefit from keyword search
        keyword_indicators = [
            'how many days', 'how many hours', 'what time', 'when',
            'where', 'who', 'what is', 'how is',
            'quantos dias', 'quantas horas', 'que horas', 'quando',
            'onde', 'quem', 'qual é', 'como é'
        ]
        
        # Queries that benefit from semantic search
        semantic_indicators = [
            'can i', 'should i', 'is it allowed', 'is it mandatory',
            'how does it work', 'what is the process', 'explain',
            'posso', 'devo', 'é permitido', 'é obrigatório',
            'como funciona', 'qual o processo', 'explique'
        ]
        
        # Counters
        keyword_score = sum(1 for indicator in keyword_indicators if indicator in query_lower)
        semantic_score = sum(1 for indicator in semantic_indicators if indicator in query_lower)
        
        # Decision
        if keyword_score > semantic_score:
            return 'keyword'
        elif semantic_score > keyword_score:
            return 'semantic'
        else:
            return 'hybrid'  # Default to hybrid
    
    def _filter_and_rank_results(self, documents: List[str], metadatas: List[Dict], query: str) -> Tuple[List[str], List[Dict]]:
        """Filter and re-rank results by relevance"""
        # Combine documents with metadata for processing
        doc_meta_pairs = list(zip(documents, metadatas))
        
        # Filter chunks that are too small or only contact info
        filtered_pairs = []
        for doc, meta in doc_meta_pairs:
            # Filter contact/administrative info chunks
            if (len(doc.strip()) > 150 and 
                not any(term in doc.lower() for term in ['nipc', 'capital social', 'rua das lapas']) and
                meta.get('section_type', 'content') != 'separator'):
                filtered_pairs.append((doc, meta))
        
        # If filtered everything, use originals (fallback)
        if not filtered_pairs:
            filtered_pairs = doc_meta_pairs
        
        # Re-rank by contextual relevance
        ranked_pairs = self._contextual_rerank(filtered_pairs, query)
        
        # Separate documents and metadata
        filtered_docs = [doc for doc, meta in ranked_pairs]
        filtered_metas = [meta for doc, meta in ranked_pairs]
        
        return filtered_docs[:6], filtered_metas[:6]  # Limit to 6 chunks
    
    def _contextual_rerank(self, doc_meta_pairs: List[Tuple[str, Dict]], query: str) -> List[Tuple[str, Dict]]:
        """Re-rank results based on query context"""
        query_terms = set(query.lower().split())
        
        scored_pairs = []
        for doc, meta in doc_meta_pairs:
            score = 0
            doc_lower = doc.lower()
            
            # Score based on query keywords
            doc_terms = set(doc_lower.split())
            term_overlap = len(query_terms.intersection(doc_terms))
            score += term_overlap * 2
            
            # Bonus for section type
            section_type = meta.get('section_type', 'content')
            if section_type in ['numbered_section', 'title']:
                score += 3
            elif section_type == 'header':
                score += 2
            
            # Bonus for specific keywords in metadata
            chunk_keywords = meta.get('keywords', [])
            keyword_overlap = len(query_terms.intersection(set(chunk_keywords)))
            score += keyword_overlap * 1.5
            
            # Penalty for very long chunks (may have too much noise)
            if len(doc) > 2000:
                score -= 1
            
            scored_pairs.append((score, doc, meta))
        
        # Sort by score (highest first)
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Return without score
        return [(doc, meta) for score, doc, meta in scored_pairs]
    
    def create_enhanced_context(self, documents: List[str], metadatas: List[Dict], query: str) -> str:
        """Create enriched context for LLM"""
        if not documents:
            return "No relevant information found in company documents."
        
        # Group by original document
        docs_by_source = {}
        for doc, meta in zip(documents, metadatas):
            source = meta.get('original_filename', 'unknown')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append({
                'content': doc,
                'section_type': meta.get('section_type', 'content'),
                'chunk_index': meta.get('chunk_index', 0),
                'keywords': meta.get('keywords', [])
            })
        
        # Sort chunks by index within each document
        for source in docs_by_source:
            docs_by_source[source].sort(key=lambda x: x['chunk_index'])
        
        # Build structured context
        context_parts = []
        context_parts.append("=== TOURTAILORS POLICIES AND DOCUMENTS ===\n")
        
        for source, chunks in docs_by_source.items():
            # Give priority to rules document
            if 'regras-de-funcionamento' in source.lower() or 'rules' in source.lower():
                context_parts.insert(1, f"\n📋 MAIN DOCUMENT: {source}")
                context_parts.insert(2, "=" * 50)
            else:
                context_parts.append(f"\n📄 DOCUMENT: {source}")
                context_parts.append("-" * 30)
            
            for chunk in chunks:
                section_info = f"[Section {chunk['chunk_index']} - {chunk['section_type']}]"
                context_parts.append(f"\n{section_info}")
                context_parts.append(chunk['content'].strip())
                context_parts.append("")
        
        final_context = "\n".join(context_parts)
        
        # Update context statistics
        self.conversation_stats['avg_context_size'] = (
            (self.conversation_stats['avg_context_size'] * self.conversation_stats['queries_processed'] + len(final_context)) /
            (self.conversation_stats['queries_processed'] + 1)
        )
        
        print(f"📝 Context created: {len(documents)} sections, {len(final_context)} characters")
        return final_context
    
    def chat(self, user_query: str, search_strategy: str = 'auto', include_history: bool = True) -> Dict:
        """Main chat interface with hybrid search"""
        start_time = datetime.now()
        
        # Check if LLM is functional
        if self.llm is None:
            return {
                'answer': "❌ LLM system is not available. Check if Ollama is installed and running.",
                'sources': [],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': True
            }
        
        # 1. Intelligent retrieval
        documents, metadatas, search_info = self.intelligent_retrieval(user_query, search_strategy)
        
        if not documents:
            return {
                'answer': "I didn't find relevant information about this question in company documents. You can rephrase the question or contact HR directly for clarification.",
                'sources': [],
                'search_strategy': search_strategy,
                'search_info': search_info,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
        
        # 2. Create enriched context
        context = self.create_enhanced_context(documents, metadatas, user_query)
        
        # 3. Determine prompt type
        query_type = self._detect_query_type(user_query)
        prompt_template = self.system_prompts[query_type]
        
        # 4. Prepare history
        if include_history:
            chat_history = self.memory.chat_memory.messages[-4:] if self.memory.chat_memory.messages else []
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        else:
            history_str = "First query of this session."
        
        # 5. Build final prompt
        if query_type == "hr_assistant":
            final_prompt = prompt_template.format(
                context=context,
                chat_history=history_str,
                question=user_query
            )
        else:
            final_prompt = prompt_template.format(
                context=context,
                question=user_query
            )
        
        # 6. Generate response via LLM
        try:
            print("🤖 Generating response with LLM...")
            response = self.llm.invoke(final_prompt)
            
            if isinstance(response, str):
                answer = response
            else:
                answer = response.content if hasattr(response, 'content') else str(response)
            
            # 7. Save to memory
            if include_history:
                self.memory.chat_memory.add_user_message(user_query)
                self.memory.chat_memory.add_ai_message(answer)
            
            # 8. Prepare sources
            sources = self._prepare_sources(metadatas)
            
            # 9. Update statistics
            self.conversation_stats['queries_processed'] += 1
            
            return {
                'answer': answer.strip(),
                'sources': sources,
                'documents_used': len(documents),
                'query_type': query_type,
                'search_strategy': search_strategy,
                'search_info': {
                    'total_found': search_info['total_found'],
                    'search_type': search_info.get('search_type', 'unknown'),
                    'expanded_queries': search_info.get('expanded_queries', [])
                },
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': True
            }
    
    def _detect_query_type(self, query: str) -> str:
        """Detect query type to choose appropriate prompt"""
        query_lower = query.lower()
        
        analysis_indicators = [
            "compare", "diferença", "análise", "avalie", "analise",
            "difference", "analysis", "evaluate", "analyze"
        ]
        
        quick_indicators = [
            "quantos", "qual", "quando", "onde", "quem",
            "how many", "what", "when", "where", "who"
        ]
        
        if any(word in query_lower for word in analysis_indicators):
            return "policy_analysis"
        elif any(word in query_lower for word in quick_indicators):
            return "quick_answer"
        else:
            return "hr_assistant"
    
    def _prepare_sources(self, metadatas: List[Dict]) -> List[Dict]:
        """Prepare source information for response"""
        sources = []
        sources_seen = set()
        
        for meta in metadatas:
            source_key = f"{meta.get('original_filename', 'unknown')}_v{meta.get('version', '?')}"
            if source_key not in sources_seen:
                sources.append({
                    'filename': meta.get('original_filename', 'unknown'),
                    'version': meta.get('version', '?'),
                    'section_type': meta.get('section_type', 'content'),
                    'chunk_count': len([m for m in metadatas if m.get('original_filename') == meta.get('original_filename')])
                })
                sources_seen.add(source_key)
        
        return sources
    
    def _update_stats(self, retrieval_time: float, docs_retrieved: int, search_strategy: str):
        """Update system statistics"""
        current_queries = self.conversation_stats['queries_processed']
        
        # Update average retrieval time
        current_avg = self.conversation_stats['avg_retrieval_time']
        new_avg = ((current_queries * current_avg) + retrieval_time) / (current_queries + 1)
        self.conversation_stats['avg_retrieval_time'] = new_avg
    
    def get_stats(self) -> Dict:
        """Get complete system statistics"""
        db_stats = self.vector_db.get_database_stats()
        
        return {
            'conversation_stats': self.conversation_stats,
            'database_stats': {
                'total_documents': db_stats.get('total_documents', 0),
                'active_documents': db_stats.get('active_documents', 0),
                'total_chunks': db_stats.get('total_chunks', 0),
                'embedding_model': db_stats.get('embedding_model', 'unknown'),
                'search_type': db_stats.get('search_type', 'unknown')
            },
            'search_config': self.search_config,
            'memory_usage': len(self.memory.chat_memory.messages) if self.memory.chat_memory.messages else 0
        }
    
    def clear_memory(self):
        """Clear conversation history"""
        self.memory.clear()
        print("Conversation memory cleared.")
    
    def test_search_strategies(self, query: str) -> Dict:
        """Test different search strategies for comparison"""
        print(f"🧪 Testing strategies for: '{query}'")
        
        strategies = ['semantic', 'keyword', 'hybrid']
        results = {}
        
        for strategy in strategies:
            try:
                docs, metas, search_info = self.intelligent_retrieval(query, strategy)
                results[strategy] = {
                    'documents_found': len(docs),
                    'total_found': search_info['total_found'],
                    'search_info': search_info,
                    'top_sources': [meta.get('original_filename', 'unknown') for meta in metas[:3]]
                }
                print(f"   {strategy}: {len(docs)} relevant docs")
            except Exception as e:
                results[strategy] = {'error': str(e)}
                print(f"   {strategy}: Error - {e}")
        
        return results

