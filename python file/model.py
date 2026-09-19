"""
model.py
N-gram Language Model Implementation (Unigram, Bigram, Trigram)
with Laplace (Add-k) Smoothing, Conditional Probabilities, and Dynamic Prediction.
"""

from collections import defaultdict, Counter
import math
from preprocessor import BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, tokenize_sentence

class NGramLanguageModel:
    """
    N-gram Language Model supporting n = 1 (Unigram), n = 2 (Bigram), n = 3 (Trigram)
    Features Laplace (Add-k) smoothing and backoff prediction.
    """
    def __init__(self, n: int = 2, k_smoothing: float = 1.0):
        """
        Args:
            n: Order of N-gram (1: Unigram, 2: Bigram, 3: Trigram)
            k_smoothing: Laplace smoothing parameter (default: 1.0)
        """
        assert n in (1, 2, 3), "Supported n-gram orders are 1 (Unigram), 2 (Bigram), and 3 (Trigram)"
        self.n = n
        self.k = float(k_smoothing)
        
        # Vocabulary
        self.vocab = set()
        self.vocab_size = 0
        
        # Frequency counts
        self.ngram_counts = Counter()      # Count of (w1, ..., wn)
        self.context_counts = Counter()    # Count of (w1, ..., wn-1)
        self.unigram_counts = Counter()    # Count of w
        self.total_tokens = 0
        
        # Additional lower-order models for backoff
        self.bigram_counts = Counter()
        self.bigram_context_counts = Counter()
        
        self.is_trained = False

    def train(self, tokenized_sentences: list[list[str]]):
        """
        Train the N-gram model on padded sentences.
        """
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.bigram_context_counts.clear()
        self.vocab.clear()
        self.total_tokens = 0

        for sentence in tokenized_sentences:
            for token in sentence:
                if token not in (BOS_TOKEN, EOS_TOKEN):
                    self.vocab.add(token)
                    self.unigram_counts[token] += 1
                    self.total_tokens += 1
                    
            # Extract Bigrams for backoff and stats
            for i in range(len(sentence) - 1):
                ctx = (sentence[i],)
                target = sentence[i + 1]
                self.bigram_counts[(sentence[i], target)] += 1
                self.bigram_context_counts[ctx] += 1

            # Extract target n-grams
            if self.n == 1:
                # Unigram
                for token in sentence:
                    if token != BOS_TOKEN:
                        self.ngram_counts[(token,)] += 1
            elif self.n == 2:
                # Bigram
                for i in range(len(sentence) - 1):
                    ctx = (sentence[i],)
                    target = sentence[i + 1]
                    self.ngram_counts[(sentence[i], target)] += 1
                    self.context_counts[ctx] += 1
            elif self.n == 3:
                # Trigram
                for i in range(len(sentence) - 2):
                    ctx = (sentence[i], sentence[i + 1])
                    target = sentence[i + 2]
                    self.ngram_counts[(sentence[i], sentence[i + 1], target)] += 1
                    self.context_counts[ctx] += 1

        self.vocab_size = len(self.vocab)
        self.is_trained = True

    def get_probability(self, word: str, context: tuple = ()) -> float:
        """
        Calculate smoothed conditional probability P(word | context).
        Applies Laplace (Add-k) smoothing:
        P(w | context) = (C(context, w) + k) / (C(context) + k * |V|)
        """
        if not self.is_trained:
            return 0.0
            
        V = max(1, self.vocab_size)
        
        if self.n == 1:
            # Unigram: P(w) = (C(w) + k) / (N + k * |V|)
            count_w = self.unigram_counts.get(word, 0)
            prob = (count_w + self.k) / (self.total_tokens + self.k * V)
            return prob
            
        elif self.n == 2:
            # Bigram: P(w | c1)
            ctx = (context[-1],) if len(context) >= 1 else (BOS_TOKEN,)
            count_ctx_w = self.ngram_counts.get((ctx[0], word), 0)
            count_ctx = self.context_counts.get(ctx, 0)
            
            # Laplace Smoothed formula
            prob = (count_ctx_w + self.k) / (count_ctx + self.k * V)
            return prob
            
        elif self.n == 3:
            # Trigram: P(w | c1, c2)
            if len(context) >= 2:
                ctx = (context[-2], context[-1])
            elif len(context) == 1:
                ctx = (BOS_TOKEN, context[-1])
            else:
                ctx = (BOS_TOKEN, BOS_TOKEN)
                
            count_ctx_w = self.ngram_counts.get((ctx[0], ctx[1], word), 0)
            count_ctx = self.context_counts.get(ctx, 0)
            
            prob = (count_ctx_w + self.k) / (count_ctx + self.k * V)
            return prob
            
        return 0.0

    def predict_next_words(self, input_text: str, top_k: int = 5, include_eos: bool = False) -> list[dict]:
        """
        Predict top-k candidate next words for a given input sentence or prefix.
        Returns a list of dictionaries with rank, word, count, probability, and percentage.
        """
        if not self.is_trained:
            return []

        tokens = tokenize_sentence(input_text, lowercase=True, remove_punctuation=True)
        
        # Prepare context tuple based on model order
        if self.n == 1:
            context = ()
        elif self.n == 2:
            context = (tokens[-1],) if len(tokens) >= 1 else (BOS_TOKEN,)
        elif self.n == 3:
            if len(tokens) >= 2:
                context = (tokens[-2], tokens[-1])
            elif len(tokens) == 1:
                context = (BOS_TOKEN, tokens[-1])
            else:
                context = (BOS_TOKEN, BOS_TOKEN)

        candidates = list(self.vocab)
        if include_eos:
            candidates.append(EOS_TOKEN)

        # Check if context exists in training corpus
        context_seen = True
        if self.n > 1:
            context_seen = self.context_counts.get(context, 0) > 0

        # Calculate probabilities for all words in vocabulary
        scored_words = []
        for word in candidates:
            prob = self.get_probability(word, context)
            
            # Extract raw count for reporting
            if self.n == 1:
                raw_count = self.unigram_counts.get(word, 0)
            elif self.n == 2:
                raw_count = self.ngram_counts.get((context[0], word), 0)
            elif self.n == 3:
                raw_count = self.ngram_counts.get((context[0], context[1], word), 0)
                
            scored_words.append({
                "word": word,
                "raw_count": raw_count,
                "probability": prob,
                "context_seen": context_seen
            })

        # If higher order context has no observed n-grams, optionally apply smart backoff scoring
        if self.n > 1 and not context_seen:
            # Backoff to unigram frequency to provide sensible fallback ranking
            for item in scored_words:
                item["backoff_unigram_count"] = self.unigram_counts.get(item["word"], 0)
            # Sort by raw probability with unigram tie-breaker
            scored_words.sort(key=lambda x: (x["probability"], x.get("backoff_unigram_count", 0)), reverse=True)
        else:
            # Sort by probability descending, then raw count
            scored_words.sort(key=lambda x: (x["probability"], x["raw_count"]), reverse=True)

        # Calculate normalized percentages among top candidates or total
        top_candidates = scored_words[:top_k]
        total_top_prob = sum(item["probability"] for item in top_candidates) or 1.0

        results = []
        for rank, item in enumerate(top_candidates, 1):
            results.append({
                "rank": rank,
                "word": item["word"],
                "raw_count": item["raw_count"],
                "probability": item["probability"],
                "percentage": f"{(item['probability'] / total_top_prob) * 100:.2f}%",
                "abs_percentage": f"{item['probability'] * 100:.2f}%",
                "context_seen": item["context_seen"]
            })

        return results

    def generate_continuation(self, prompt: str, num_words: int = 5) -> str:
        """
        Autonomously continue a sentence by greedily picking the most probable next word.
        """
        current_text = prompt
        generated = []
        
        for _ in range(num_words):
            preds = self.predict_next_words(current_text, top_k=1, include_eos=False)
            if not preds:
                break
            next_word = preds[0]["word"]
            if next_word in (EOS_TOKEN, BOS_TOKEN, UNK_TOKEN):
                break
            generated.append(next_word)
            current_text += " " + next_word

        return " ".join(generated)

    def get_summary_stats(self) -> dict:
        """Return model metadata and training parameters."""
        return {
            "model_type": {1: "Unigram (1-gram)", 2: "Bigram (2-gram)", 3: "Trigram (3-gram)"}.get(self.n),
            "n_order": self.n,
            "smoothing_k": self.k,
            "vocab_size": self.vocab_size,
            "total_ngrams_learned": len(self.ngram_counts),
            "total_tokens": self.total_tokens
        }
