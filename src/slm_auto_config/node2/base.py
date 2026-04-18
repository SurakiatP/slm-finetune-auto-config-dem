import logging
import re
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

class BaseSDGEngine:
    """
    Core engine providing Global Deduplication (LSH + FAISS) for any SDG task.
    """
    def __init__(self, task_description: str, target_count: int, max_loops: int = 20):
        self.task_description = task_description
        self.target_count = target_count
        self.max_loops = max_loops
        self.similarity_threshold = 0.85
        self.threshold = 0.70  # Minimum judge score
        
        # MinHash LSH setup
        self.num_perm = 128
        self.global_lsh = MinHashLSH(threshold=0.90, num_perm=self.num_perm)
        self.lsh_counter = 0

        # FAISS setup
        logger.info("Initializing SDGEngine: Loading Embedding Model (paraphrase-multilingual-MiniLM-L12-v2)...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

    def compute_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        clean_text = re.sub(r'[^a-zA-Z0-9ก-๙]', '', str(text)).lower()
        if len(clean_text) < 5:
            m.update(clean_text.encode('utf-8'))
        else:
            for i in range(len(clean_text) - 4):
                ngram = clean_text[i:i+5]
                m.update(ngram.encode('utf-8'))
        return m

    def inject_seed_into_index(self, texts: List[str]):
        """Hash and embed seed texts to ensure generator does not copy them exactly."""
        if not texts:
            return
            
        logger.info(f"Indexing {len(texts)} seed items into FAISS/LSH...")
        for text in texts:
            m = self.compute_minhash(text)
            self.global_lsh.insert(f"seed_{self.lsh_counter}", m)
            self.lsh_counter += 1
            
        seed_embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        self.faiss_index.add(seed_embeddings)

    def is_semantically_redundant(self, text_embedding) -> Tuple[bool, float]:
        """Checks if a vector is too similar to existing generated/seed content."""
        D, I = self.faiss_index.search(text_embedding.reshape(1, -1), 1)
        max_sim = float(D[0][0])
        return max_sim >= self.similarity_threshold, max_sim

    def add_generated_to_index(self, text: str, text_embedding):
        """Adds verified text to the global LSH and FAISS index."""
        m = self.compute_minhash(text)
        self.global_lsh.insert(f"gen_{self.lsh_counter}", m)
        self.lsh_counter += 1
        self.faiss_index.add(text_embedding.reshape(1, -1))

    def run(self, seed_data_path: str, output_path: str, api_kwargs: Dict[str, str]):
        raise NotImplementedError("Child classes must implement the run method for specific SDG pipelines.")
