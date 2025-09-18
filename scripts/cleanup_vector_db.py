# cleanup_vector_db.py - Vector database cleanup script

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.manager import VectorDatabaseManager


def main():
    """Main cleanup script"""
    parser = argparse.ArgumentParser(description='Vector Database Cleanup')
    
    parser.add_argument('--action', 
                       choices=['clear', 'reset', 'remove', 'stats'], 
                       required=True,
                       help='Action to execute')
    
    parser.add_argument('--document', 
                       help='Document name to remove (only for action=remove)')
    
    parser.add_argument('--all-versions', 
                       action='store_true',
                       help='Remove all document versions (only for action=remove)')
    
    parser.add_argument('--keep-processed', 
                       action='store_true',
                       help='Keep processed files (only for action=reset)')
    
    parser.add_argument('--confirm', 
                       action='store_true',
                       help='Confirm destructive action')
    
    args = parser.parse_args()
    
    # Initialize manager
    db_manager = VectorDatabaseManager()
    
    if args.action == 'stats':
        # Show current statistics
        print("📊 Database Statistics:")
        stats = db_manager.get_database_stats()
        for key, value in stats.items():
            if key == 'type_statistics':
                print(f"   Document Types:")
                for doc_type, type_stats in value.items():
                    print(f"     {doc_type}: {type_stats['active']}/{type_stats['total']} (active/total)")
            else:
                print(f"   {key}: {value}")
        
        # List documents
        print("\n📋 Documents in Database:")
        docs = db_manager.get_document_info(include_inactive=True)
        for doc in docs:
            status = "✅ ACTIVE" if doc['is_active'] else "❌ INACTIVE"
            print(f"   {status} {doc['original_filename']} v{doc['version']} ({doc['total_chunks']} chunks)")
    
    elif args.action == 'clear':
        # Complete cleanup
        if not args.confirm:
            print("⚠️  WARNING: This operation will delete the ENTIRE database!")
            print("   - All documents and versions")
            print("   - All chunks and embeddings")
            print("   - Files in processed/ folder")
            print("   - Document registry")
            print("\n🔥 To confirm, execute: python cleanup_database.py --action clear --confirm")
            return
        
        print("🗑️  Clearing database...")
        result = db_manager.clear_database(confirm=True)
        
        if result['status'] == 'success':
            print(f"✅ {result['message']}")
            print(f"   📄 Documents removed: {result['documents_removed']}")
            print(f"   🧩 Chunks removed: {result['chunks_removed']}")
        else:
            print(f"❌ {result['message']}")
    
    elif args.action == 'reset':
        # Reset (keep kb files)
        if not args.confirm:
            print("⚠️  WARNING: This operation will reset the database!")
            print("   - Delete all embeddings and metadata")
            print("   - Keep original documents in kb/ for reprocessing")
            processed_action = "keep" if args.keep_processed else "delete"
            print(f"   - {processed_action.capitalize()} files in processed/")
            print("\n🔥 To confirm, execute: python cleanup_database.py --action reset --confirm")
            return
        
        print("🔄 Resetting database...")
        result = db_manager.reset_database(keep_processed_files=args.keep_processed)
        
        if result['status'] == 'success':
            print(f"✅ {result['message']}")
            print(f"   📄 Documents removed: {result['documents_removed']}")
            print(f"   🧩 Chunks removed: {result['chunks_removed']}")
            print("\n💡 To reprocess: python vector_db_manager.py")
        else:
            print(f"❌ {result['message']}")
    
    elif args.action == 'remove':
        # Remove specific document
        if not args.document:
            print("❌ Error: --document is required for action=remove")
            print("   Example: python cleanup_database.py --action remove --document 'vacation_policy.txt'")
            return
        
        versions_text = "all versions" if args.all_versions else "active version"
        
        if not args.confirm:
            print(f"⚠️  WARNING: Will remove {versions_text} of document '{args.document}'")
            print("\n🔥 To confirm, add --confirm")
            return
        
        print(f"🗑️  Removing {versions_text} of '{args.document}'...")
        result = db_manager.remove_document(args.document, all_versions=args.all_versions)
        
        if result['status'] == 'success':
            print(f"✅ {result['message']}")
            print(f"   📄 Documents removed: {result['documents_removed']}")
            print(f"   🧩 Chunks removed: {result['chunks_removed']}")
        elif result['status'] == 'not_found':
            print(f"ℹ️ {result['message']}")
        else:
            print(f"❌ {result['message']}")

def interactive_cleanup():
    """Interactive cleanup interface"""
    print("🧹 INTERACTIVE DATABASE CLEANUP")
    print("=" * 40)
    
    db_manager = VectorDatabaseManager()
    
    # Show current statistics
    stats = db_manager.get_database_stats()
    print(f"\n📊 Current State:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Active documents: {stats['active_documents']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    
    print(f"\n🎯 Available options:")
    print("   1. View document details")
    print("   2. Remove specific document")
    print("   3. Complete reset (keep kb/)")
    print("   4. Total cleanup")
    print("   5. Exit")
    
    while True:
        try:
            choice = input("\n👉 Choose an option (1-5): ").strip()
            
            if choice == '1':
                # View details
                docs = db_manager.get_document_info(include_inactive=True)
                print("\n📋 Documents in Database:")
                for i, doc in enumerate(docs, 1):
                    status = "✅ ACTIVE" if doc['is_active'] else "❌ INACTIVE"
                    print(f"   {i}. {status} {doc['original_filename']} v{doc['version']}")
                    print(f"      📅 {doc['processing_date']}")
                    print(f"      🧩 {doc['total_chunks']} chunks")
            
            elif choice == '2':
                # Remove document
                docs = db_manager.get_document_info()
                if not docs:
                    print("ℹ️ No active documents in database.")
                    continue
                
                print("\n📋 Active documents:")
                for i, doc in enumerate(docs, 1):
                    print(f"   {i}. {doc['original_filename']} v{doc['version']}")
                
                try:
                    doc_choice = int(input("👉 Choose document to remove (number): ").strip()) - 1
                    if 0 <= doc_choice < len(docs):
                        filename = docs[doc_choice]['original_filename']
                        confirm = input(f"⚠️  Confirm removal of '{filename}'? (y/N): ").strip().lower()
                        if confirm == 'y':
                            result = db_manager.remove_document(filename)
                            print(f"✅ {result['message']}")
                        else:
                            print("❌ Operation cancelled.")
                    else:
                        print("❌ Invalid choice.")
                except ValueError:
                    print("❌ Please enter a valid number.")
            
            elif choice == '3':
                # Reset
                confirm = input("⚠️  Complete database reset (keep kb/)? (y/N): ").strip().lower()
                if confirm == 'y':
                    result = db_manager.reset_database()
                    print(f"✅ {result['message']}")
                else:
                    print("❌ Operation cancelled.")
            
            elif choice == '4':
                # Total cleanup
                confirm = input("⚠️  TOTAL CLEANUP (delete everything)? (y/N): ").strip().lower()
                if confirm == 'y':
                    result = db_manager.clear_database(confirm=True)
                    print(f"✅ {result['message']}")
                else:
                    print("❌ Operation cancelled.")
            
            elif choice == '5':
                print("👋 See you next time!")
                break
            
            else:
                print("❌ Invalid option. Choose 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 See you next time!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If executed without arguments, use interactive mode
        interactive_cleanup()
    else:
        # Use command line arguments
        main()