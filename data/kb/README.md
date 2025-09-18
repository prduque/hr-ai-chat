# Knowledge Base Documents 📚

This folder is where you place your HR documents for processing by the AI system.

## Supported File Types

- **PDF files** (`.pdf`) - Employee handbooks, policies, procedures
- **Text files** (`.txt`) - Simple text documents, FAQs, guidelines

## How to Add Documents

1. **Copy your HR documents** to this folder
2. **Run the processing script**:
   ```bash
   python scripts/setup_database.py
   ```
3. **Start chatting** about your documents:
   ```bash
   python scripts/chat.py
   ```

## Document Examples

### Recommended HR Documents:
- `employee_handbook.pdf` - Complete employee handbook
- `vacation_policy.txt` - Vacation and leave policies  
- `code_of_conduct.pdf` - Code of conduct and ethics
- `benefits_guide.txt` - Employee benefits information
- `remote_work_policy.pdf` - Remote work guidelines
- `performance_review.txt` - Performance evaluation process
- `safety_procedures.pdf` - Workplace safety guidelines
- `onboarding_checklist.txt` - New employee onboarding

### File Organization Tips:
```
data/kb/
├── policies/
│   ├── vacation_policy.pdf
│   ├── remote_work_policy.pdf
│   └── code_of_conduct.pdf
├── handbooks/
│   ├── employee_handbook_2024.pdf
│   └── manager_handbook.pdf
├── procedures/
│   ├── onboarding_process.txt
│   ├── performance_review.txt
│   └── expense_reimbursement.pdf
└── benefits/
    ├── health_insurance.pdf
    ├── retirement_plan.txt
    └── employee_perks.pdf
```

## Document Processing

The system will:
1. **Extract text** from PDFs and TXT files
2. **Apply OCR** if PDFs contain scanned images
3. **Create intelligent chunks** for better search results
4. **Generate embeddings** for semantic search
5. **Extract keywords** for hybrid search capabilities

## Document Updates

When you update a document:
1. **Replace the old file** with the new version (same filename)
2. **Re-run processing**: `python scripts/setup_database.py`
3. The system will **automatically version** the documents
4. **Old versions** are deactivated but preserved

## Best Practices

### Document Quality:
- ✅ Use clear, well-formatted documents
- ✅ Include headers and section titles
- ✅ Avoid scanned PDFs when possible (or ensure good quality)
- ✅ Use consistent naming conventions

### Content Guidelines:
- ✅ Include official company policies
- ✅ Keep documents up-to-date
- ✅ Use clear language and terminology
- ✅ Include contact information where relevant

### File Management:
- ✅ Use descriptive filenames
- ✅ Organize by category/department
- ✅ Remove outdated documents
- ✅ Backup important files

## Privacy & Security

⚠️ **Important Security Notes:**
- Only add **publicly shareable** HR documents
- Remove any **confidential personal information**
- Avoid documents with **employee personal data**
- Consider **data privacy regulations** in your region

## Troubleshooting

### Common Issues:

1. **"No text extracted"**
   - Check if PDF is scanned (OCR required)
   - Ensure file is not corrupted
   - Try converting to TXT format

2. **"Processing failed"**
   - Check file permissions
   - Ensure file is not password-protected
   - Verify file format is supported

3. **"Poor search results"**
   - Improve document structure with headers
   - Use clear, descriptive language
   - Add more related documents

### Getting Help:

For issues with document processing:
1. Check the logs: `hr_chat.log`
2. Test system requirements: `python tests/test_requirements.py`
3. View database stats: `python scripts/cleanup_database.py --action stats`

---

**Ready to add your documents? Drop them in this folder and run the setup script!** 🚀