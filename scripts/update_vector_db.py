# update_vector_db.py - Creation, update and search of Vector Database, with Hybrid Search and Document versioning

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.manager import VectorDatabaseManager


def main():
    """Main knowledge base update script with Hybrid Search"""
    
    # Initialize manager
    db_manager = VectorDatabaseManager()

    if db_manager.has_new_documents():
        print("\n📚 Processing documents with Hybrid Search...")
        stats = db_manager.process_all_documents()
        
        print(f"\n📊 Processing Results:")
        print(f"   ✅ Processed: {stats['processed']}")
        print(f"   ⭐ Skipped: {stats['skipped']}")
        print(f"   ❌ Errors: {stats['errors']}")
        
        print("\n📈 Database Statistics:")
        db_stats = db_manager.get_database_stats()
        for key, value in db_stats.items():
            if key != 'type_statistics':
                print(f"   {key}: {value}")
        
        if 'type_statistics' in db_stats:
            print("\n   Document Types:")
            for doc_type, stats in db_stats['type_statistics'].items():
                print(f"     {doc_type}: {stats['active']}/{stats['total']} (active/total)")
       
        print("\n✅ Processing completed with Hybrid Search!")
        print(f"📋 Registry saved at: {db_manager.registry_file}")
        print(f"🗃️ Database at: {db_manager.db_path}")
    else:
        print("\n📚 No documents to process!")
        
        
if __name__ == "__main__":
    main()
                        