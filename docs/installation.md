# Installation Guide 🔧

Complete installation guide for HR AI Chat system.

## System Requirements

### Python Version
- **Recommended**: Python 3.11 or 3.12
- **Avoid**: Python 3.13 (compatibility issues with some dependencies)
- **Minimum**: Python 3.9

### Operating Systems
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu 18+, CentOS 7+)

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB
- **Storage**: 5GB free space for models and data
- **CPU**: Multi-core recommended for better performance

## Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/prduque/hr-ai-chat.git
cd hr-ai-chat
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Update pip first (IMPORTANT!)
python -m pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python tests/test_requirements.py
```

### 5. Setup System
```bash
# Add documents to data/kb/ folder
# Then process them:
python scripts/setup_database.py

# Start chatting:
python scripts/chat.py
```

## Detailed Installation

### Step 1: Python Setup

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
3. Verify: `python --version`

#### macOS
```bash
# Using Homebrew (recommended)
brew install python@3.12

# Or download from python.org
```

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# CentOS/RHEL
sudo yum install python3.12 python3.12-devel
```

### Step 2: Dependencies Installation

#### Critical Order (Windows)
```bash
# Step 1: Base tools
pip install --upgrade pip setuptools wheel

# Step 2: NumPy FIRST (for ChromaDB compatibility)
pip install "numpy>=1.26.0,<2.0.0"

# Step 3: ChromaDB (depends on NumPy < 2.0)
pip install "chromadb>=0.4.15,<0.5.0"

# Step 4: All other requirements
pip install -r requirements.txt
```

### Step 3: Optional Components

#### Tesseract OCR (for PDF processing)

**Windows:**
1. Download from [tesseract-ocr](https://github.com/UB-Mannheim/tesseract/wiki)
2. Add to system PATH
3. Verify: `tesseract --version`

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng
```

#### Ollama (for local LLM)

**Windows:**
```bash
# Using winget
winget install Ollama.Ollama

# Or download from https://ollama.ai/download/windows
```

**macOS/Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start service
ollama serve

# Install model
ollama pull llama3.1:8b
```

## Configuration

### Environment Variables (Optional)
Create a `.env` file in the project root:
```bash
# OpenAI API (if using OpenAI instead of Ollama)
OPENAI_API_KEY=your_api_key_here

# Logging level
LOG_LEVEL=WARNING

# Custom paths
KB_PATH=data/kb
DB_PATH=data/vector_db
```

### System Settings
Edit `config/settings.yaml` (optional):
```yaml
search:
  semantic_weight: 0.7
  keyword_weight: 0.3
  n_results: 8

llm:
  provider: ollama  # or openai
  model: llama3.1:8b
  temperature: 0.2

processing:
  chunk_size: 2000
  chunk_overlap: 400
```

## Verification

### Test System Components
```bash
# Test all requirements
python tests/test_requirements.py

# Test vector database
python tests/test_vector_db.py

# Test complete system
python scripts/test_system.py
```

### Expected Output
```
✅ NumPy: 1.26.4 (MUST be < 2.0.0)
✅ ChromaDB: OK
✅ SentenceTransformers: OK
✅ PDFPlumber: OK
✅ Tesseract version: 5.3.0
✅ LangChain: OK

🎉 SYSTEM READY FOR USE!
```

## Troubleshooting

### Common Issues

#### 1. NumPy 2.x Compatibility Error
```bash
# Problem: ChromaDB incompatible with NumPy 2.x
pip install "numpy>=1.26.0,<2.0.0" --force-reinstall
pip install --force-reinstall chromadb
```

#### 2. Pydantic Version Conflicts
```bash
# Clean install
pip uninstall pydantic langchain langchain-core langchain-community -y
pip install pydantic==2.5.3
pip install langchain==0.2.17 langchain-core==0.2.43 langchain-community==0.2.19
```

#### 3. ChromaDB Initialization Error
```bash
# Clear cache and reinstall
pip cache purge
pip install --only-binary=all chromadb sentence-transformers
```

#### 4. Tesseract Not Found
**Windows:**
- Add Tesseract to system PATH
- Common path: `C:\Program Files\Tesseract-OCR`

**Linux/macOS:**
```bash
# Check installation
which tesseract
tesseract --version

# If missing, reinstall
```

#### 5. Ollama Connection Error
```bash
# Check if Ollama is running
ollama serve

# Test connection
ollama run llama3.1:8b "Hello"

# Check port (default: 11434)
curl http://localhost:11434
```

#### 6. Permission Errors
**Windows:**
- Run command prompt as Administrator
- Check antivirus software

**Linux/macOS:**
```bash
# Fix permissions
chmod +x scripts/*.py
sudo chown -R $USER:$USER ~/.cache
```

### Platform-Specific Issues

#### Windows
- Use `python` instead of `python3`
- Use backslashes in paths: `venv\Scripts\activate`
- Install Visual C++ Build Tools if needed

#### macOS
- Use `python3` explicitly
- Install Xcode Command Line Tools: `xcode-select --install`
- Handle M1/M2 compatibility with Rosetta if needed

#### Linux
- Install development packages: `python3-dev`, `build-essential`
- Handle package manager differences (apt/yum/pacman)
- Check SELinux/AppArmor policies if needed

### Performance Issues

#### Slow Processing
1. **Hardware**: Ensure sufficient RAM (8GB+)
2. **Models**: Models download on first use (~2GB)
3. **CPU**: Use multi-threading settings in config

#### Memory Issues
```bash
# Monitor memory usage
python -c "
import psutil
print(f'RAM: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.1f}GB')
"
```

## Maintenance

### Update Dependencies
```bash
# Update safely (avoid breaking changes)
pip install --upgrade pip setuptools wheel

# Check for updates
pip list --outdated

# Update specific packages
pip install --upgrade sentence-transformers
```

### Clean Installation
```bash
# Remove virtual environment
rm -rf venv/  # Linux/macOS
rmdir /s venv  # Windows

# Clear cache
pip cache purge

# Start fresh
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

### Backup Configuration
```bash
# Export working environment
pip freeze > requirements-working.txt

# Backup data
cp -r data/ data_backup/
cp document_registry.json document_registry_backup.json
```

## Next Steps

After successful installation:

1. **Add Documents**: Place HR documents in `data/kb/`
2. **Process Data**: Run `python scripts/setup_database.py`
3. **Test System**: Run `python scripts/test_system.py`
4. **Start Chatting**: Run `python scripts/chat.py`

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review logs in `hr_chat.log`
3. Test system requirements
4. Check GitHub issues
5. Contact support with error details

---

**Ready to go? Start with: `python scripts/chat.py`** 🚀