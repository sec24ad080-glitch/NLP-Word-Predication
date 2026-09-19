"""
evaluator.py
Model Evaluation & Benchmarking Module: Perplexity, Top-1/Top-K Accuracy, and Model Comparison.
"""

import math
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessor import BOS_TOKEN, EOS_TOKEN, preprocess_corpus, split_sentences
from model import NGramLanguageModel

def calculate_perplexity(model: NGramLanguageModel, test_sentences: list[list[str]]) -> float:
    """
    Calculate Perplexity (PP) of the trained N-gram model on test sentences.
    Perplexity = exp( - 1/N * sum( log P(w_i | context) ) )
    Lower perplexity indicates a better language model.
    """
    if not model.is_trained or not test_sentences:
        return float('inf')

    log_prob_sum = 0.0
    total_words = 0

    for sentence in test_sentences:
        # Padded sentence tokens
        for i in range(len(sentence)):
            target_word = sentence[i]
            if target_word in (BOS_TOKEN, EOS_TOKEN):
                continue
                
            if model.n == 1:
                context = ()
            elif model.n == 2:
                context = (sentence[i - 1],) if i >= 1 else (BOS_TOKEN,)
            elif model.n == 3:
                c1 = sentence[i - 2] if i >= 2 else BOS_TOKEN
                c2 = sentence[i - 1] if i >= 1 else BOS_TOKEN
                context = (c1, c2)

            prob = model.get_probability(target_word, context)
            
            # Avoid math domain error if probability is extremely low or zero
            prob = max(prob, 1e-12)
            log_prob_sum += math.log(prob)
            total_words += 1

    if total_words == 0:
        return float('inf')

    cross_entropy = - (log_prob_sum / total_words)
    try:
        perplexity = math.exp(cross_entropy)
    except OverflowError:
        perplexity = float('inf')

    return round(perplexity, 2)

def calculate_accuracy(model: NGramLanguageModel, test_sentences: list[list[str]], top_k_list: list[int] = [1, 3, 5]) -> dict:
    """
    Evaluate Top-k next word prediction accuracy on test sentences.
    For each word w_i in the test corpus, check if w_i is in model.predict_next_words(context, top_k).
    """
    if not model.is_trained or not test_sentences:
        return {f"top_{k}_acc": 0.0 for k in top_k_list}

    correct_counts = {k: 0 for k in top_k_list}
    total_eval_tokens = 0
    max_k = max(top_k_list)

    for sentence in test_sentences:
        for i in range(len(sentence)):
            target_word = sentence[i]
            if target_word in (BOS_TOKEN, EOS_TOKEN):
                continue

            # Build context string
            if model.n == 1:
                context_str = ""
            elif model.n == 2:
                context_str = sentence[i - 1] if i >= 1 else ""
            elif model.n == 3:
                c1 = sentence[i - 2] if i >= 2 else ""
                c2 = sentence[i - 1] if i >= 1 else ""
                context_str = f"{c1} {c2}".strip()

            predictions = model.predict_next_words(context_str, top_k=max_k, include_eos=False)
            predicted_words = [p["word"] for p in predictions]

            for k in top_k_list:
                if target_word in predicted_words[:k]:
                    correct_counts[k] += 1

            total_eval_tokens += 1

    results = {}
    for k in top_k_list:
        acc = (correct_counts[k] / total_eval_tokens * 100) if total_eval_tokens > 0 else 0.0
        results[f"top_{k}_acc"] = round(acc, 2)

    results["total_tokens_evaluated"] = total_eval_tokens
    return results

def evaluate_models_comparison(raw_text: str, test_size: float = 0.2, k_smoothing: float = 1.0, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Train and compare Unigram, Bigram, and Trigram models on the given corpus.
    Splits sentences into Train and Test sets.
    Returns:
        comparison_df: Pandas DataFrame comparing models across metrics
        models_dict: Dict with trained model instances {1: unigram, 2: bigram, 3: trigram}
    """
    sentences = split_sentences(raw_text)
    
    if len(sentences) > 3:
        train_sents, test_sents = train_test_split(sentences, test_size=test_size, random_state=random_state)
    else:
        train_sents, test_sents = sentences, sentences

    train_text = " ".join(train_sents)
    test_text = " ".join(test_sents)

    records = []
    models_dict = {}

    for order, name in [(1, "Unigram (1-gram)"), (2, "Bigram (2-gram)"), (3, "Trigram (3-gram)")]:
        train_padded, train_stats = preprocess_corpus(train_text, n_order=order)
        test_padded, _ = preprocess_corpus(test_text, n_order=order)

        model = NGramLanguageModel(n=order, k_smoothing=k_smoothing)
        model.train(train_padded)
        models_dict[order] = model

        # Metrics
        pp = calculate_perplexity(model, test_padded)
        acc_dict = calculate_accuracy(model, test_padded, top_k_list=[1, 3, 5])

        records.append({
            "Model": name,
            "Order (N)": order,
            "Perplexity (Lower is better)": pp,
            "Top-1 Accuracy (%)": f"{acc_dict['top_1_acc']}%",
            "Top-3 Accuracy (%)": f"{acc_dict['top_3_acc']}%",
            "Top-5 Accuracy (%)": f"{acc_dict['top_5_acc']}%",
            "Unique N-grams": len(model.ngram_counts),
            "Vocab Size": model.vocab_size
        })

    df = pd.DataFrame(records)
    return df, models_dict
