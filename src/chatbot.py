# src/chatbot.py
import os
import pickle
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.preprocessing import normalize
import subprocess


class LegalChatbot:
    def __init__(self, index_path='./faiss_index',
                 embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model_name='google/flan-t5-base',
                 device=None):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Using device: {self.device}")

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.index_path = index_path

        # Load or build FAISS index
        self._load_index()

        # Load LLM model (with safe fallback)
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=hf_token)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                llm_model_name,
                token=hf_token,
                device_map="auto",
                torch_dtype="auto",
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print("⚠️ device_map loading failed, using CPU fallback:", e)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name).to(self.device)

        self.model.eval()
        print("✅ Model loaded successfully!")

    def _load_index(self):
        idx_file = os.path.join(self.index_path, 'index.faiss')
        meta_file = os.path.join(self.index_path, 'meta.pkl')

        # Automatically rebuild index if missing
        if not os.path.exists(idx_file) or not os.path.exists(meta_file):
            print("⚠️ FAISS index not found. Running ingest.py to build it...")
            try:
                subprocess.run(["python", "src/ingest.py"], check=True)
            except Exception as e:
                print(f"❌ Failed to run ingest.py automatically: {e}")
                self.index = None
                self.meta, self.texts = [], []
                return

        print("📚 Loading FAISS index...")
        self.index = faiss.read_index(idx_file)
        with open(meta_file, 'rb') as f:
            data = pickle.load(f)

        self.meta = data['meta']
        self.texts = data['texts']
        print(f"✅ Loaded FAISS index with {len(self.texts)} entries.")

    def _retrieve(self, query, top_k=5):
        if not self.index:
            return []

        q_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        q_emb = normalize(q_emb).astype('float32')

        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= 0:
                results.append({
                    'question': self.meta[int(idx)]['question'],
                    'answer': self.meta[int(idx)]['answer'],
                    'meta': self.meta[int(idx)],
                    'score': float(score)
                })
        return results

    def _generate(self, prompt, max_new_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def ask(self, query, top_k=5, show_sources=False, min_score=0.25):
        chunks = self._retrieve(query, top_k=top_k)
        if not chunks:
            return "Sorry, I couldn't find anything relevant in the knowledge base."

        best = chunks[0]
        if best['score'] >= 0.55:
            answer = best['answer']
        else:
            combined = "\n\n".join([c['answer'] for c in chunks])
            prompt = (
                "You are a helpful Indian legal assistant. "
                "Answer based ONLY on the provided context:\n\n"
                f"{combined}\n\n"
                f"Question: {query}\nAnswer:"
            )
            answer = self._generate(prompt)

        if show_sources:
            sources_text = "\n\nSources:\n" + "\n".join(
                f"- {c['meta'].get('source', 'unknown')} (score={c['score']:.2f})"
                for c in chunks
            )
            answer = answer.strip() + "\n\n" + sources_text

        return answer.strip()
