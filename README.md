# HR AI Chat 🤖💼

An intelligent HR assistant powered by Retrieval-Augmented Generation (RAG) with hybrid search capabilities. Built specifically for Human Resources departments to provide instant, accurate answers from company policies and documents.

## 🌟 Features

- **Hybrid Search**: Combines semantic and keyword search for optimal results
- **Multilingual Support**: Works with both English and Portuguese queries
- **Document Processing**: Supports PDF and TXT files with OCR capabilities
- **Version Management**: Automatic document versioning and update detection
- **Interactive Chat**: Command-line interface with conversation memory
- **Intelligent Chunking**: Smart document segmentation with contextual awareness

## 🚀 Quick Start

### Prerequisites

- Python 3.11/3.12 (Python 3.13 may cause compatibility issues)
- [Ollama](https://ollama.ai/) for local LLM support (optional)
- Tesseract OCR for PDF processing (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prduque/hr-ai-chat.git
   cd hr-ai-chat
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python tests/test_requirements.py
   ```

### Quick Setup

1. **Add your HR documents** to the `data/kb/` folder
2. **Initialize the database**:
   ```bash
   python scripts/setup_database.py
   ```
3. **Start chatting**:
   ```bash
   python scripts/chat.py
   ```

## 📖 Usage

### Adding Documents

Place your HR documents (PDF or TXT) in the `data/kb/` folder:
```
data/kb/
├── employee_handbook.pdf
├── vacation_policy.txt
├── code_of_conduct.pdf
└── benefits_guide.txt
```

### Processing Documents

```bash
# Process all documents in kb/ folder
python scripts/setup_database.py

# View database statistics
python scripts/cleanup_database.py --action stats
```

### Interactive Chat

```bash
python scripts/chat.py
```

Available commands:
- `quit` - Exit the chat
- `clear` - Clear conversation history  
- `stats` - Show system statistics
- `strategy <type>` - Change search strategy (semantic/keyword/hybrid/auto)
- `test <query>` - Test all search strategies

### Example Queries

```
🤔 You: How many vacation days am I entitled to?
🤔 You: Can I schedule vacation during holidays?
🤔 You: What is the company policy on remote work?
🤔 You: Posso marcar férias junto a feriados?  # Portuguese support
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │  Vector Database │    │   RAG System    │
│   (PDF/TXT)     │───▶│   (ChromaDB)     │───▶│   (LangChain)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Hybrid Search   │    │   Chat Interface│
                       │ Semantic+Keyword │    │   (CLI/Memory)  │
                       └──────────────────┘    └─────────────────┘
```

### Key Components

- **Vector Database Manager**: Document processing, embedding generation, hybrid search
- **RAG System**: Query processing, context creation, LLM integration  
- **Document Processor**: Text extraction, intelligent chunking, metadata enrichment
- **Keyword Extractor**: Multilingual synonym expansion, stopword filtering

## 🔧 Configuration

### Search Strategies

- **Auto**: Automatically selects best strategy based on query type
- **Hybrid**: Combines semantic (70%) + keyword (30%) search
- **Semantic**: Pure embedding-based search
- **Keyword**: Exact term matching with synonyms

### LLM Providers

- **Ollama** (default): Local LLM execution
- **OpenAI**: Cloud-based GPT models (requires API key)

## 🧪 Testing

```bash
# Test system requirements
python tests/test_requirements.py

# Test vector database
python tests/test_vector_db.py

# Run full system test
python scripts/test_system.py
```

## 📊 Database Management

```bash
# View database statistics
python scripts/cleanup_database.py --action stats

# Remove specific document
python scripts/cleanup_database.py --action remove --document "old_policy.pdf" --confirm

# Reset database (keep documents)
python scripts/cleanup_database.py --action reset --confirm

# Complete cleanup
python scripts/cleanup_database.py --action clear --confirm
```

## 🛠️ Development

### Project Structure

```
src/
├── vector_db/          # Database management
├── rag/               # RAG system logic
└── utils/             # Utilities

scripts/               # Executable scripts
tests/                # Test files
data/                 # Document storage
docs/                 # Documentation
```

### Adding New Features

1. **Custom Document Types**: Extend `DocumentProcessor`
2. **New Search Strategies**: Implement in `VectorDatabaseManager`
3. **Enhanced Prompts**: Modify `system_prompts` in RAG system
4. **Web Interface**: Build FastAPI wrapper around existing classes

## 🐛 Troubleshooting

### Common Issues

1. **NumPy 2.x Compatibility**
   ```bash
   pip install "numpy>=1.26.0,<2.0.0" --force-reinstall
   pip install --force-reinstall chromadb
   ```

2. **Ollama Connection Error**
   - Install Ollama: `winget install Ollama.Ollama`
   - Pull model: `ollama pull llama3.1:8b`

3. **PDF Processing Issues**
   - Install Tesseract OCR
   - Check system PATH configuration

See [docs/installation.md](docs/installation.md) for detailed troubleshooting.

## 📈 Performance

- **Average query time**: 0.5-2.0 seconds
- **Supported document size**: Up to 100MB per PDF
- **Concurrent users**: Single-user CLI (expandable to multi-user)
- **Memory usage**: ~2GB for typical HR document set

## 🛣️ Roadmap

- [ ] **Web Interface**: FastAPI + React frontend
- [ ] **Multi-user Support**: Authentication and user sessions
- [ ] **Document Analytics**: Usage statistics and insights
- [ ] **API Endpoints**: REST API for integration
- [ ] **Docker Support**: Containerized deployment
- [ ] **Advanced RAG**: Query rewriting, multi-step reasoning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit: `git commit -m 'Add feature-name'`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for RAG capabilities
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Ollama](https://ollama.ai/) for local LLM execution

---

**Made with ❤️ for HR teams who deserve better tools**