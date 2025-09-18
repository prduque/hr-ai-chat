import sys
print(f'Python: {sys.version}')

import numpy
print(f'✅ Numpy: {numpy.__version__}')

import chromadb
print('✅ ChromaDB: OK')

from sentence_transformers import SentenceTransformer
print('✅ SentenceTransformers: OK')

import pdfplumber
print('✅ PDFPlumber: OK')

import pytesseract
from PIL import Image

# Test if Tesseract is configured
try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract version: {version}")
except Exception as e:
    print(f"❌ Tesseract Error: {e}")
    print("Check if Tesseract is installed and in PATH")
    exit()

import langchain;
print('✅ LangChain: OK')

print('\n🎉 SYSTEM READY FOR USE!')