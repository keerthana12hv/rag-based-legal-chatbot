# src/ingest.py
import os
import pickle
import faiss
import hashlib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

DATA_DIR = './data'
INDEX_DIR = './faiss_index'
os.makedirs(INDEX_DIR, exist_ok=True)

device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
print("Using device for embeddings:", device)

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

index_path = os.path.join(INDEX_DIR, 'index.faiss')
meta_path = os.path.join(INDEX_DIR, 'meta.pkl')


def file_hash(filepath):
    """Return MD5 hash of file contents"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def split_into_QA_pairs(text):
    """Split raw file into (Q, A) pairs"""
    pairs = []
    q, a = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Q:"):
            if q and a:
                pairs.append((q, " ".join(a)))
                a = []
            q = line[2:].strip()
        elif line.startswith("A:"):
            a.append(line[2:].strip())
        else:
            a.append(line)
    if q and a:
        pairs.append((q, " ".join(a)))
    return pairs


# Always rebuild index fresh
texts, meta = [], []
print("ℹ️ Rebuilding FAISS index from scratch...")

for file in sorted(os.listdir(DATA_DIR)):
    if not file.endswith('.txt'):
        continue
    path = os.path.join(DATA_DIR, file)
    current_hash = file_hash(path)

    print(f"Indexing: {file}")
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
        qa_pairs = split_into_QA_pairs(raw)
        for i, (q, a) in enumerate(qa_pairs):
            if q and a:
                # Store Q + A for embedding → better recall
                texts.append(q + " " + a)
                meta.append({
                    'source': file,
                    'chunk': i,
                    'question': q,
                    'answer': a,
                    'hash': current_hash
                })

if texts:
    print(f"Creating embeddings for {len(texts)} Q/A pairs...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize(embeddings, axis=1).astype('float32')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump({'texts': texts, 'meta': meta}, f)

    print(f"✅ Indexed {len(texts)} Q/A pairs from {len(os.listdir(DATA_DIR))} files.")
else:
    print("⚠️ No .txt files found in data/")
