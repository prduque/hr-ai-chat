# test_vector_db.py - Test script for hybrid system

import time
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.manager import VectorDatabaseManager


def test_basic_functionality():
    """Basic functionality test"""
    print("🔧 BASIC FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        db_manager = VectorDatabaseManager()
        db_manager.prepare_for_searches()
        
        print(f"✅ System initialized")
        print(f"   Model: {db_manager.embedding_model}")
        print(f"   Dimensions: {db_manager.embedding_model.get_sentence_embedding_dimension()}")
        
        return db_manager
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return None

def test_search_strategies(db_manager):
    """Test different search strategies"""
    print(f"\n🔍 SEARCH STRATEGIES TEST")
    print("=" * 50)
    
    test_queries = [
        "Can I schedule vacation during holidays?",
        "How many vacation days am I entitled to?",
        "How does vacation approval work?",
        "Posso marcar férias junto a feriados?",  # Portuguese test
    ]
    
    strategies = ['semantic', 'keyword', 'hybrid']
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 30)
        
        for strategy in strategies:
            try:
                start_time = time.time()
                
                if strategy == 'hybrid':
                    results = db_manager.hybrid_search(query, n_results=5)
                else:
                    results = db_manager.search_documents(query, n_results=5, search_type=strategy)
                
                elapsed = time.time() - start_time
                
                print(f"   {strategy:>8}: {results['total_found']:>2} results ({elapsed:.3f}s)")
                
                # Show top 2 results
                if results['total_found'] > 0:
                    for i, meta in enumerate(results['results']['metadatas'][0][:2]):
                        filename = meta.get('original_filename', 'unknown')
                        section = meta.get('section_type', 'content')
                        print(f"            {i+1}. {filename} ({section})")
                
            except Exception as e:
                print(f"   {strategy:>8}: ❌ Error - {e}")
        
        print()

def interactive_search_test(db_manager):
    """Interactive search test"""
    print(f"\n🎯 INTERACTIVE TEST")
    print("=" * 50)
    print("Enter queries to test (empty enter to exit)")
    
    while True:
        query = input("\n🔍 Query: ").strip()
        if not query:
            break
        
        print(f"\nTesting: '{query}'")
        print("-" * 40)
        
        # Test all strategies
        for strategy in ['semantic', 'keyword', 'hybrid']:
            try:
                start_time = time.time()
                
                if strategy == 'hybrid':
                    results = db_manager.hybrid_search(query, n_results=3)
                else:
                    results = db_manager.search_documents(query, n_results=3, search_type=strategy)
                
                elapsed = time.time() - start_time
                
                print(f"\n📊 {strategy.upper()}:")
                print(f"   Results: {results['total_found']}")
                print(f"   Time: {elapsed:.3f}s")
                
                if results['total_found'] > 0:
                    print(f"   Top result:")
                    meta = results['results']['metadatas'][0][0]
                    doc = results['results']['documents'][0][0]
                    
                    filename = meta.get('original_filename', 'unknown')
                    section = meta.get('section_type', 'content')
                    keywords = meta.get('keywords', [])
                    
                    print(f"     📄 {filename} ({section})")
                    if keywords:
                        print(f"     🏷️  Keywords: {', '.join(keywords[:5])}")
                    print(f"     🔍 Preview: {doc[:150]}...")
                
            except Exception as e:
                print(f"   ❌ {strategy}: {e}")

def benchmark_performance(db_manager):
    """Performance benchmark"""
    print(f"\n⚡ PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    benchmark_queries = [
        "vacation holidays policy",
        "working hours schedule company",
        "approval director advance notice",
        "vacation holiday policy",
        "employee staff worker"
    ]
    
    strategies = ['semantic', 'keyword', 'hybrid']
    results = {}
    
    for strategy in strategies:
        print(f"\n🏃 Testing {strategy}...")
        times = []
        
        for query in benchmark_queries:
            try:
                start_time = time.time()
                
                if strategy == 'hybrid':
                    search_results = db_manager.hybrid_search(query, n_results=5)
                else:
                    search_results = db_manager.search_documents(query, n_results=5, search_type=strategy)
                
                elapsed = time.time() - start_time
                times.append(elapsed)
                
            except Exception as e:
                print(f"   ❌ Error in '{query}': {e}")
                times.append(float('inf'))
        
        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
            
            results[strategy] = {
                'avg': avg_time,
                'min': min_time,
                'max': max_time,
                'success_rate': len(valid_times) / len(times)
            }
            
            print(f"   ✅ Average: {avg_time:.3f}s | Min: {min_time:.3f}s | Max: {max_time:.3f}s")
        else:
            print(f"   ❌ All tests failed")
            results[strategy] = None
    
    # Benchmark summary
    print(f"\n📊 BENCHMARK SUMMARY:")
    print("-" * 30)
    for strategy, stats in results.items():
        if stats:
            print(f"{strategy:>8}: {stats['avg']:.3f}s (average) | {stats['success_rate']:.1%} success")
        else:
            print(f"{strategy:>8}: FAILED")

def main():
    """Main test function"""
    print("🧪 COMPLETE HYBRID SYSTEM TEST")
    print("=" * 60)
    
    # 1. Basic test
    db_manager = test_basic_functionality()
    if not db_manager:
        print("❌ Tests cancelled - system not initialized")
        return
    
    # 2. Strategy test
    test_search_strategies(db_manager)
    
    # 3. Benchmark
    benchmark_performance(db_manager)
    
    # 4. Interactive test (optional)
    choice = input(f"\nℹ️ Run interactive test? (y/N): ").strip().lower()
    if choice == 'y':
        interactive_search_test(db_manager)
    
    print(f"\n✅ Tests completed!")

if __name__ == "__main__":
    main()