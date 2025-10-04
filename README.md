# Legal Chatbot (Free RAG starter)

**Purpose**: Starter project for an AI Legal Chatbot (MVP) focused on common public laws (consumer, motor vehicles, IT, labor). 
Built with free/open-source components: FAISS (local vector store), HuggingFace sentence-transformers (embeddings), and Groq LLM (optional free tier).

## What is included
- `data/` : placeholder text files (consumer_protection.txt, motor_vehicles.txt, it_act.txt, labor_rights.txt)
- `src/ingest.py` : loads text files, chunks, creates embeddings and builds FAISS index
- `src/chatbot.py` : simple retrieval and LLM call (Groq) helper
- `src/utils.py` : helper functions
- `app.py` : Streamlit UI to chat with the bot
- `requirements.txt` : Python dependencies (install with pip)
- `.env.example` : template for API key(s)
- `LICENSE` : Apache 2.0 license (replace name/year if desired)

## Quick setup (example)
1. Create a Python virtual environment and activate it (venv/conda)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your GROQ_API_KEY if you plan to use Groq for generation:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Put your legal text files (PDFs converted to text or plain .txt) inside `data/`.
5. Run ingestion to build the FAISS index:
   ```bash
   python src/ingest.py --data-dir data --index-path ./faiss_index
   ```
6. Start the Streamlit UI:
   ```bash
   streamlit run app.py
   ```

## Notes
- This is a starter kit. You will need to add real legal documents (PDF -> text) into `data/` to get useful answers.
- The Groq LLM call in `src/chatbot.py` is a simple example using REST. You can swap in any LLM provider (OpenAI, Anthropic, local Llama) you prefer.
- If internet-free operation is required, you can substitute an offline LLM and local embedding models.

## License
This project is provided under the Apache 2.0 license (see LICENSE file).
