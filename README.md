# 📘 StudyMate – AI Academic Assistant

**StudyMate** is a Streamlit-based AI assistant that helps you study and research efficiently.  
It allows you to upload PDF documents, extract text, split it into chunks, perform semantic search,  
and get concise AI-generated answers using **OpenAI** or **Hugging Face** models

## 🚀 Features
- 📥 **Upload PDFs** – Supports multiple PDFs at once.
- 📄 **Text Extraction** – Extracts searchable text from PDFs (supports scanned PDFs with future OCR integration).
- ✂ **Chunking** – Splits text into overlapping chunks for better semantic search.
- 🔎 **Semantic Search** – Uses **Sentence Transformers** for embeddings and optional **FAISS** for fast similarity search.
- 🤖 **AI-Powered Answers** – Generates concise, context-based answers from:
  - OpenAI Chat models (e.g., `gpt-3.5-turbo`)
  - Hugging Face Inference API (e.g., `Mistral-7B-Instruct`)
- 📚 **Source References** – Always provides document/chunk IDs in the answer

## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/studymate.git
cd studymate
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Activate it:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, create one with:
```txt
streamlit
python-dotenv
PyMuPDF
numpy
sentence-transformers
faiss-cpu
openai
requests
```

---

## 🔑 Environment Variables

Create a `.env` file in the root of your project:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-3.5-turbo
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

**⚠ Important:** Never commit your `.env` file to GitHub. Add it to `.gitignore`:
```txt
.env
```

---

## ▶ Running the App

Once your `.env` is set up and dependencies are installed, run:
```bash
streamlit run app.py
```

Open your browser at the URL shown in the terminal (usually http://localhost:8501).

---

## 📂 Project Structure
```
StudyMate/
│
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env                # API keys and config (not tracked in Git)
└── README.md           # Project documentation
```

---

## 💡 Usage
1. **Upload PDFs** – Select one or more PDF files.
2. **Extract Text** – Click "Read text from PDFs".
3. **Chunk Text** – Adjust chunk size/overlap, then "Create Chunks".
4. **Build Index** – Click "Build Semantic Index" for embeddings.
5. **Ask Questions** – Type your question and choose your AI provider.
6. **View Answer** – See AI-generated answers with source references.

---

## 📦 Optional: Installing FAISS for Faster Search
If FAISS isn’t available, NumPy fallback is used.  
To install FAISS CPU version:
```bash
pip install faiss-cpu
```

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI](https://platform.openai.com/)
- [Hugging Face](https://huggingface.co/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
