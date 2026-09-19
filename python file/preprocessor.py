"""
preprocessor.py
NLP Text Preprocessing & Tokenization Module using NLTK and Regex fallbacks.
"""

import re
import nltk

# Ensure necessary NLTK resources are available
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

def clean_text(text: str) -> str:
    """Normalize whitespace and remove unwanted non-standard characters."""
    if not text:
        return ""
    # Normalize multiple whitespace characters
    cleaned = re.sub(r'\s+', ' ', text.strip())
    return cleaned

def split_sentences(text: str) -> list[str]:
    """Split corpus into sentences using NLTK sent_tokenize with regex fallback."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    try:
        sentences = nltk.sent_tokenize(cleaned)
    except Exception:
        # Fallback regex sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    return [s.strip() for s in sentences if s.strip()]

def tokenize_sentence(sentence: str, lowercase: bool = True, remove_punctuation: bool = True) -> list[str]:
    """
    Tokenize a single sentence into a list of word tokens.
    Handles lowercasing and optional punctuation removal.
    """
    if lowercase:
        sentence = sentence.lower()
    
    if remove_punctuation:
        # Replace punctuation marks with spaces, keeping alphanumeric tokens
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        tokens = sentence.split()
    else:
        try:
            tokens = nltk.word_tokenize(sentence)
        except Exception:
            tokens = re.findall(r'\w+|[^\w\s]', sentence)
            
    return [t.strip() for t in tokens if t.strip()]

def preprocess_corpus(text: str, n_order: int = 2, lowercase: bool = True, remove_punctuation: bool = True) -> tuple[list[list[str]], dict]:
    """
    Preprocess entire text corpus:
    - Splits text into sentences
    - Tokenizes each sentence
    - Adds boundary padding tokens (<s>, </s>) based on n-gram order
    
    Returns:
        padded_sentences: List of token lists with padding
        stats: Dictionary containing raw statistics (sentences, raw_tokens, vocab_size, etc.)
    """
    sentences = split_sentences(text)
    all_raw_tokens = []
    padded_sentences = []
    
    # Boundary tokens needed: (n_order - 1) BOS tokens and 1 EOS token
    bos_count = max(1, n_order - 1)
    
    for sent in sentences:
        tokens = tokenize_sentence(sent, lowercase=lowercase, remove_punctuation=remove_punctuation)
        if not tokens:
            continue
        all_raw_tokens.extend(tokens)
        
        # Add boundary tags
        padded = [BOS_TOKEN] * bos_count + tokens + [EOS_TOKEN]
        padded_sentences.append(padded)
        
    vocab = set(all_raw_tokens)
    
    stats = {
        "num_sentences": len(sentences),
        "num_tokens": len(all_raw_tokens),
        "vocab_size": len(vocab),
        "vocabulary": sorted(list(vocab)),
        "char_count": len(text)
    }
    
    return padded_sentences, stats
