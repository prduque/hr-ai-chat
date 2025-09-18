# query_rag_llm.py - RAG System with Hybrid Search

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.manager import VectorDatabaseManager
from src.chat import HybridHRRAGSystem


def test_hybrid_system():
    """Complete test function for hybrid system"""
    print("🧪 HYBRID RAG SYSTEM TEST")
    print("=" * 50)
    
    try:
        # Initialize system
        system = HybridHRRAGSystem(
            llm_provider="ollama",
            model_name="llama3.1:8b",
            temperature=0.2
        )
        
        # Test queries for different scenarios
        test_queries = [
            ("Can I schedule a day off between a holiday and weekend?", "auto"),
            ("Can I schedule vacation during holidays?", "hybrid"),
            ("How many vacation days am I entitled to?", "keyword"), 
            ("How does the vacation approval process work?", "semantic"),
            ("What is the company policy on working hours?", "auto"),
            ("Posso marcar férias junto a feriados?", "auto"),
            ("Quero um itinerário para as minhas férias de duas semanas, em itália", "auto")
        ]
        
        print("\n📊 Initial statistics:")
        stats = system.get_stats()
        print(f"   Documents in DB: {stats['database_stats']['active_documents']}")
        print(f"   Total chunks: {stats['database_stats']['total_chunks']}")
        print(f"   Model: {stats['database_stats']['embedding_model']}")
        
        print("\n💬 Testing queries with different strategies:")
        for i, (query, strategy) in enumerate(test_queries, 1):
            print(f"\n{i}. 🤔 QUESTION: {query}")
            print(f"   📊 Strategy: {strategy}")
            print("-" * 40)
            
            response = system.chat(query, search_strategy=strategy)
            
            if not response.get('error'):
                print(f"✅ RESPONSE:")
                print(response['answer'])
                print(f"\n📚 SOURCES ({len(response['sources'])}):")
                for j, source in enumerate(response['sources'], 1):
                    print(f"   {j}. {source['filename']} (v{source['version']}) - {source['chunk_count']} sections")
                
                search_info = response.get('search_info', {})
                print(f"\n🔍 SEARCH INFO:")
                print(f"   Strategy used: {response.get('search_strategy', 'unknown')}")
                print(f"   Search type: {search_info.get('search_type', 'unknown')}")
                print(f"   Results found: {search_info.get('total_found', 0)}")
                if search_info.get('expanded_queries'):
                    print(f"   Expanded queries: {len(search_info['expanded_queries'])}")
                
                print(f"⏱️ Time: {response['processing_time']:.2f}s")
            else:
                print(f"❌ ERROR: {response['answer']}")
            
            print("=" * 50)
        
        # Final statistics
        print(f"\n📈 Final statistics:")
        final_stats = system.get_stats()
        conv_stats = final_stats['conversation_stats']
        print(f"   Queries processed: {conv_stats['queries_processed']}")
        print(f"   Hybrid searches: {conv_stats['hybrid_searches']}")
        print(f"   Semantic searches: {conv_stats['semantic_searches']}")
        print(f"   Keyword searches: {conv_stats['keyword_searches']}")
        print(f"   Average retrieval time: {conv_stats['avg_retrieval_time']:.3f}s")
        print(f"   Average context size: {conv_stats['avg_context_size']:.0f} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def interactive_chat():
    """Enhanced interactive interface"""
    print("🚀 Hybrid RAG System - Interactive Chat")
    print("Commands: 'quit', 'clear', 'stats', 'test <query>', 'strategy <semantic|keyword|hybrid|auto>'")
    print("=" * 70)
    
    try:
        system = HybridHRRAGSystem()
        current_strategy = 'auto'
        
        while True:
            try:
                user_input = input(f"\n🤔 You [{current_strategy}]: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    system.clear_memory()
                    continue
                elif user_input.lower() == 'stats':
                    stats = system.get_stats()
                    print(f"\n📊 System Statistics:")
                    for section, data in stats.items():
                        print(f"   {section}:")
                        if isinstance(data, dict):
                            for key, value in data.items():
                                print(f"     {key}: {value}")
                        else:
                            print(f"     {data}")
                    continue
                elif user_input.lower().startswith('strategy '):
                    new_strategy = user_input.split(' ', 1)[1].strip()
                    if new_strategy in ['semantic', 'keyword', 'hybrid', 'auto']:
                        current_strategy = new_strategy
                        print(f"✅ Strategy changed to: {current_strategy}")
                    else:
                        print("❌ Valid strategies: semantic, keyword, hybrid, auto")
                    continue
                elif user_input.lower().startswith('test '):
                    test_query = user_input.split(' ', 1)[1].strip()
                    test_results = system.test_search_strategies(test_query)
                    print(f"\n🧪 Results for '{test_query}':")
                    for strategy, result in test_results.items():
                        if 'error' in result:
                            print(f"   {strategy}: ❌ {result['error']}")
                        else:
                            print(f"   {strategy}: {result['documents_found']} docs ({result['total_found']} total)")
                    continue
                elif not user_input:
                    continue
                
                print("\n🔄 Processing...")
                response = system.chat(user_input, search_strategy=current_strategy)
                
                if not response.get('error'):
                    print(f"\n🤖 HR Assistant: {response['answer']}")
                    if response['sources']:
                        print(f"\n📚 Sources used:")
                        for i, source in enumerate(response['sources'], 1):
                            print(f"   {i}. {source['filename']} (v{source['version']})")
                    
                    # Debug info
                    search_info = response.get('search_info', {})
                    if search_info:
                        print(f"\n🔍 [{response.get('search_strategy', 'unknown')}] "
                              f"{search_info.get('search_type', 'unknown')} - "
                              f"{search_info.get('total_found', 0)} results")
                else:
                    print(f"\n❌ Error: {response['answer']}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 See you next time!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
    
    except Exception as e:
        print(f"❌ Initialization error: {e}")



def main():    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_hybrid_system()
    elif len(sys.argv) > 1 and sys.argv[1] == "--chat":
        interactive_chat()
    else:
        print("Usage:")
        print("  python query_rag_llm.py --test   # Run complete tests")
        print("  python query_rag_llm.py --chat   # Interactive chat")
        

if __name__ == "__main__":
    main()
                        