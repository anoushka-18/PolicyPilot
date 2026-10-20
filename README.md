# PolicyPilot — AI-Powered Policy and Compliance Assistant

PolicyPilot is a Retrieval-Augmented Generation (RAG) application that helps users search and understand policy and compliance documents using AI. Users can upload PDF and TXT files, ask questions in natural language, and receive answers based on the uploaded documents with source references.

## Features

- **Document Upload:** Upload PDF and TXT documents.
- **Knowledge Base Creation:** Extract, process, and split documents into searchable text chunks.
- **Semantic Search:** Use Sentence Transformers to convert text into embeddings and FAISS to retrieve relevant content.
- **AI-Powered Answers:** Generate contextual answers using the Google Gemini API.
- **Source References:** Display source evidence to help users verify answers against their documents.
- **Interactive Chat Interface:** Ask questions through a Streamlit-based interface.
- **Relevance Checking:** Reduce unsupported responses by checking whether retrieved content is relevant to the question.

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- FAISS
- PyPDF

## How It Works

1. Upload policy or compliance documents.
2. Extract text from the documents.
3. Split the extracted text into smaller, overlapping chunks.
4. Generate vector embeddings using the `all-MiniLM-L6-v2` Sentence Transformer model.
5. Store and search embeddings using FAISS.
6. Retrieve relevant text chunks for the user's question.
7. Send the question and retrieved context to Gemini to generate an answer.
8. Display the answer and supporting source references.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/anoushka-18/PolicyPilot.git
cd PolicyPilot
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install streamlit pypdf sentence-transformers faiss-cpu google-genai
```

### 4. Configure the Gemini API key

Create a file named `.streamlit/secrets.toml` and add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Replace the placeholder with your own API key. Never commit or share this file.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Project Structure

```text
PolicyPilot/
├── app.py
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml
```

The `.streamlit/secrets.toml` file is local configuration and should remain excluded from Git.

## Limitations

- Answer quality depends on the quality and relevance of the uploaded documents.
- Scanned PDFs may require OCR if they do not contain extractable text.
- AI-generated answers can contain errors and should be verified against the original documents.
- Gemini API access and usage may be subject to applicable quotas and charges.

## Future Improvements

- Deploy the application online.
- Add persistent document storage.
- Improve source-level citations and answer verification.
- Support additional document formats.
- Add document management and user authentication.

## Author

**Anoushka Jain**

Computer Engineering Student | AI, Data Science and Software Development

GitHub: [anoushka-18](https://github.com/anoushka-18)

---

*PolicyPilot is an educational project exploring Retrieval-Augmented Generation, semantic search, and AI-assisted document analysis.*
