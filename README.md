# Bhagavad Gita Q&A Assistant

This project implements a Retrieval-Augmented Generation (RAG) system that can answer questions about the Bhagavad Gita. The system parses the PDF of the Bhagavad Gita, chunks the content, stores it locally with embeddings, and uses Google's Gemini 2.0 Flash LLM to provide detailed answers based on the text.

## Features

- **PDF Processing**: Extracts text from the Bhagavad Gita PDF and chunks it for efficient retrieval
- **Local Storage**: Stores text chunks and embeddings locally (no external vector database required)
- **Dual Interface**: 
  - Command line interface for quick questions
  - Web interface using Flask for a more user-friendly experience
- **Powerful AI**: Uses Google's Gemini 2.0 Flash LLM for high-quality, contextual answers
- **Source References**: Provides answers with context from the relevant parts of the text
- **Explain More Feature**: Request more detailed explanations with a single click or command

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- PDF of the Bhagavad Gita in the `Data` folder

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Bhagavad-Gita-Project
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Unix/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   ```

### Usage

#### Command Line Interface

Run the application in CLI mode:
```bash
python app.py
```

After receiving an answer, you can type `explain more` to get a more detailed explanation of the previous answer in about 50-60 words.

#### Web Interface

Start the Flask web server:
```bash
python app.py --web
```
Then open your browser and navigate to `http://localhost:5000`

For each answer provided by the assistant, you'll see an "Explain more" button below the response. Click this button to get a more detailed explanation (approximately 50-60 words) of the previous answer.

## Project Structure

- `app.py` - Main application with both CLI and web interfaces
- `chunks.py` - Module for processing PDF and storing chunks locally
- `llm_ai.py` - Module for initializing the Gemini LLM
- `data_blocks/` - Directory where chunks and embeddings are stored
- `Data/` - Contains the Bhagavad Gita PDF
- `templates/` - HTML templates for the web interface
- `agent_logs/` - Logs of queries and system operations

## How It Works

1. The system extracts text from the PDF using PyPDF2
2. The text is split into chunks using LangChain's text splitter
3. Embeddings are generated for each chunk using a Hugging Face model
4. Chunks and embeddings are stored locally in JSON files
5. When a question is asked, the system:
   - Converts the question to an embedding
   - Finds the most similar text chunks
   - Sends the question and relevant chunks to Gemini
   - Returns the AI-generated answer with context

## License

This project is provided for educational purposes only. The Bhagavad Gita is a sacred text, and this tool is intended to help people learn and understand its teachings.
