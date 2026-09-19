"""
sample_data.py
Curated sample corpora for demonstrating Next-Word Prediction models in NLP.
"""

SAMPLE_CORPORA = {
    "AI & Natural Language Processing": (
        "Natural language processing is a subfield of artificial intelligence and linguistics. "
        "Natural language processing is used to build intelligent computational systems that understand human language. "
        "Natural language processing is powerful and enables machines to read, analyze, and interpret spoken and written text. "
        "Natural language processing is useful for speech recognition, sentiment analysis, and machine translation. "
        "Machine learning algorithms are essential for training robust next-word prediction models. "
        "Deep learning and recurrent neural networks are widely used in modern language modeling. "
        "Language models calculate the probability of a sequence of words occurring in a sentence. "
        "Language models predict the next word based on previous context words. "
        "An n-gram model is a probabilistic language model based on Markov assumption. "
        "Unigram models assume each word occurs independently of all other words. "
        "Bigram models predict the next word using only the single previous word. "
        "Trigram models predict the next word using the two preceding words. "
        "Laplace smoothing is applied to handle zero probabilities for unseen n-grams. "
        "Artificial intelligence is transforming industries worldwide with smart automated solutions. "
        "Data science and statistical analysis help engineers understand text data patterns. "
        "Text preprocessing includes lowercasing, tokenization, removing special characters, and sentence segmentation. "
        "Perplexity is an evaluation metric that measures the quality of a language model. "
        "A lower perplexity score indicates a better language model with higher predictive confidence. "
        "Next-word prediction is widely used in smartphone keyboards, search engines, and smart autocomplete systems."
    ),
    "Science, Technology & Computing": (
        "Computer science is the study of computation, information, and automation. "
        "Computer science is fundamental to developing modern software applications. "
        "Software engineering is the systematic approach to designing, developing, and maintaining software systems. "
        "Python is a popular programming language for artificial intelligence, web development, and data science. "
        "Python is simple to learn and provides powerful libraries for numerical computing and machine learning. "
        "Algorithms are step-by-step procedures used for solving complex computational problems efficiently. "
        "Data structures like arrays, hash maps, and trees are essential for fast data retrieval. "
        "Cloud computing provides scalable infrastructure and reliable remote storage services. "
        "Cybersecurity is crucial for protecting sensitive digital data and network communications from attacks. "
        "Modern technology empowers students and engineers to create innovative software projects. "
        "The internet is a global network of interconnected computers communicating via standardized protocols. "
        "Open-source software promotes collaboration, transparency, and continuous learning among developers."
    ),
    "Classic Literature & General English": (
        "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife. "
        "To be or not to be that is the question that has puzzled philosophers for centuries. "
        "All that glitters is not gold, often have you heard that told by wise men. "
        "Knowledge is power and wisdom comes with curiosity, dedication, and patient practice. "
        "The sun rises in the east and sets in the west every single day without fail. "
        "A journey of a thousand miles begins with a single confident step forward. "
        "Curiosity is the engine of intellectual growth and scientific discovery. "
        "Reading books expands the mind, fosters creativity, and improves communication skills. "
        "Every great accomplishment begins with the decision to try and learn continuously."
    )
}

def get_sample_corpus_names():
    return list(SAMPLE_CORPORA.keys())

def get_sample_corpus(name: str) -> str:
    return SAMPLE_CORPORA.get(name, SAMPLE_CORPORA["AI & Natural Language Processing"])
