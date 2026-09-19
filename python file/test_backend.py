"""
test_backend.py
Unit tests for Next-Word Prediction backend modules.
"""

import unittest
from sample_data import get_sample_corpus
from preprocessor import preprocess_corpus, clean_text, split_sentences, tokenize_sentence
from model import NGramLanguageModel
from evaluator import calculate_perplexity, calculate_accuracy, evaluate_models_comparison

class TestNLPNextWordModel(unittest.TestCase):
    def setUp(self):
        self.sample_text = get_sample_corpus("AI & Natural Language Processing")

    def test_preprocessor(self):
        sents = split_sentences(self.sample_text)
        self.assertGreater(len(sents), 5)
        tokens = tokenize_sentence("Natural Language Processing is powerful!", lowercase=True, remove_punctuation=True)
        self.assertEqual(tokens, ["natural", "language", "processing", "is", "powerful"])

        padded, stats = preprocess_corpus(self.sample_text, n_order=2)
        self.assertGreater(stats["num_tokens"], 50)
        self.assertGreater(stats["vocab_size"], 20)

    def test_models_training_and_prediction(self):
        for n in [1, 2, 3]:
            padded, stats = preprocess_corpus(self.sample_text, n_order=n)
            model = NGramLanguageModel(n=n, k_smoothing=1.0)
            model.train(padded)
            self.assertTrue(model.is_trained)
            
            # Predict for a typical prompt
            preds = model.predict_next_words("Natural language processing is", top_k=3)
            self.assertEqual(len(preds), 3)
            for p in preds:
                self.assertIn("rank", p)
                self.assertIn("word", p)
                self.assertIn("probability", p)
                self.assertGreater(p["probability"], 0.0)

    def test_perplexity_and_accuracy(self):
        df, models = evaluate_models_comparison(self.sample_text, test_size=0.25, k_smoothing=1.0)
        self.assertEqual(len(df), 3)
        self.assertIn("Perplexity (Lower is better)", df.columns)
        self.assertIn("Top-1 Accuracy (%)", df.columns)
        print("\nModel Comparison Table:")
        print(df.to_string(index=False))

    def test_out_of_vocab_and_unseen_context(self):
        padded, _ = preprocess_corpus(self.sample_text, n_order=3)
        model = NGramLanguageModel(n=3, k_smoothing=0.5)
        model.train(padded)
        
        # Test unseen prefix
        preds = model.predict_next_words("Completely unknown words xyz abc", top_k=3)
        self.assertEqual(len(preds), 3)
        # Should gracefully return valid vocabulary words using smoothing/backoff
        for p in preds:
            self.assertIn(p["word"], model.vocab)

if __name__ == "__main__":
    unittest.main()
