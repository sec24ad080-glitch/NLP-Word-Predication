"""
app.py
Interactive Next-Word Prediction NLP Prototype
Built with Streamlit, NLTK, Pandas, NumPy, and Scikit-learn.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os

from sample_data import SAMPLE_CORPORA, get_sample_corpus_names, get_sample_corpus
from preprocessor import preprocess_corpus, clean_text, split_sentences, tokenize_sentence
from model import NGramLanguageModel
from evaluator import calculate_perplexity, calculate_accuracy, evaluate_models_comparison

# Set Page Config
st.set_page_config(
    page_title="Next-Word Prediction Model | NLP Prototype",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 50%, #DB2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        color: #F8FAFC;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }
    
    .rank-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .rank-1 { background-color: #FEF3C7; color: #92400E; }
    .rank-2 { background-color: #E0E7FF; color: #3730A3; }
    .rank-3 { background-color: #FCE7F3; color: #9D174D; }
    .rank-other { background-color: #F1F5F9; color: #475569; }
    
    .highlight-box {
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .flow-step {
        background: #1E293B;
        color: #E2E8F0;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .formula-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "corpus_text" not in st.session_state:
    st.session_state.corpus_text = get_sample_corpus("AI & Natural Language Processing")
if "k_smoothing" not in st.session_state:
    st.session_state.k_smoothing = 1.0
if "selected_model_order" not in st.session_state:
    st.session_state.selected_model_order = 2  # Bigram default
if "active_model" not in st.session_state:
    st.session_state.active_model = None
if "corpus_stats" not in st.session_state:
    st.session_state.corpus_stats = None
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = "Natural language processing is"

# Helper to train/refresh model
def train_current_model(text: str, n_order: int, k_smooth: float):
    padded_sents, stats = preprocess_corpus(text, n_order=n_order)
    model = NGramLanguageModel(n=n_order, k_smoothing=k_smooth)
    model.train(padded_sents)
    st.session_state.active_model = model
    st.session_state.corpus_stats = stats
    return model, stats

# Initial training if model not initialized
if st.session_state.active_model is None:
    train_current_model(
        st.session_state.corpus_text,
        st.session_state.selected_model_order,
        st.session_state.k_smoothing
    )

# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("NLP Project Suite")
    st.markdown("**Next-Word Prediction** using Statistical N-gram Language Modeling.")
    st.divider()

    menu = st.radio(
        "Navigate Project Sections:",
        [
            "🏠 1. Project Home & Overview",
            "📚 2. Corpus & Training",
            "🔮 3. Prediction Playground",
            "📊 4. Model Evaluation & Comparison"
        ],
        index=0
    )

    st.divider()
    st.subheader("⚙️ Quick Model Controls")
    
    model_name_map = {1: "Unigram (1-gram)", 2: "Bigram (2-gram)", 3: "Trigram (3-gram)"}
    selected_n = st.selectbox(
        "Active Prediction Model:",
        options=[1, 2, 3],
        index=st.session_state.selected_model_order - 1,
        format_func=lambda x: model_name_map[x],
        help="Select which N-gram model to query for next-word predictions."
    )
    
    k_val = st.slider(
        "Laplace Smoothing ($k$):",
        min_value=0.01,
        max_value=2.0,
        value=float(st.session_state.k_smoothing),
        step=0.05,
        help="Add-k smoothing factor to prevent zero probabilities for unseen N-grams."
    )

    # Trigger re-train if parameters changed
    if (selected_n != st.session_state.selected_model_order or k_val != st.session_state.k_smoothing):
        st.session_state.selected_model_order = selected_n
        st.session_state.k_smoothing = k_val
        train_current_model(st.session_state.corpus_text, selected_n, k_val)
        st.success("Model updated successfully!")

    st.caption("College Mini Project Demo | Antigravity AI Engine")

# ==========================================
# SECTION 1: HOME & OVERVIEW
# ==========================================
if "1. Project Home" in menu:
    st.markdown('<div class="main-title">Next-Word Prediction Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Statistical Natural Language Processing Project using N-gram Language Models with Laplace Smoothing</div>', unsafe_allow_html=True)
    
    # Project Objective Card
    st.markdown("""
    <div class="highlight-box">
        <h4>🎯 Project Objective</h4>
        <p>Design, build, and evaluate an interactive statistical <b>Next-Word Prediction NLP Prototype</b> that dynamically learns language patterns from any arbitrary text corpus using <b>Unigram, Bigram, and Trigram</b> models, computes conditional probabilities with <b>Laplace Smoothing</b>, and outputs ranked next-word candidates alongside empirical <b>Perplexity</b> and <b>Accuracy</b> metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline Flow
    st.subheader("🔄 End-to-End Functional Architecture")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown('<div class="flow-step">1. Corpus Upload / Text Input</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="flow-step">2. NLTK Sentence & Word Tokenize</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="flow-step">3. N-gram Frequency Generation</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="flow-step">4. Laplace Smoothing (Add-k)</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="flow-step">5. Conditional Probability Ranking</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="flow-step">6. Perplexity & Accuracy Evaluation</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Concepts & Mathematical Formulation
    st.subheader("📐 Mathematical Formulations")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1. Unigram Model ($n=1$)")
        st.caption("Assumes words occur independently (Bag-of-Words assumption):")
        st.latex(r"P(w_i) = \frac{C(w_i) + k}{N + k \cdot |V|}")
        st.caption("**Pros:** Fast, lightweight. | **Cons:** No context memory.")

    with col2:
        st.markdown("#### 2. Bigram Model ($n=2$)")
        st.caption("Predicts next word using 1 previous context word (1st-order Markov):")
        st.latex(r"P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k \cdot |V|}")
        st.caption("**Pros:** Captures word pairs. | **Cons:** 1-word horizon.")

    with col3:
        st.markdown("#### 3. Trigram Model ($n=3$)")
        st.caption("Predicts next word using 2 previous context words (2nd-order Markov):")
        st.latex(r"P(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i) + k}{C(w_{i-2}, w_{i-1}) + k \cdot |V|}")
        st.caption("**Pros:** High contextual precision. | **Cons:** Sparsity without smoothing.")

    # Call to action
    st.info("💡 **Ready to explore?** Head to the **Corpus & Training** section to inspect or upload training data, or test predictions directly in the **Prediction Playground**.")

# ==========================================
# SECTION 2: CORPUS & TRAINING
# ==========================================
elif "2. Corpus & Training" in menu:
    st.markdown('<div class="main-title">Corpus Management & Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect current training data, upload custom text files, and explore vocabulary statistics</div>', unsafe_allow_html=True)

    tab_preset, tab_upload, tab_custom = st.tabs(["📦 Built-in Sample Corpora", "📤 Upload Custom .txt Corpus", "✍️ Manual Text Editor"])

    with tab_preset:
        sample_names = get_sample_corpus_names()
        selected_sample = st.selectbox("Select Built-in Sample Domain:", sample_names, index=0)
        
        if st.button("Load & Train with Selected Sample Corpus", type="primary"):
            new_text = get_sample_corpus(selected_sample)
            st.session_state.corpus_text = new_text
            train_current_model(new_text, st.session_state.selected_model_order, st.session_state.k_smoothing)
            st.success(f"Successfully loaded and trained model on '{selected_sample}' corpus!")

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a plain text corpus (.txt)", type=["txt"])
        if uploaded_file is not None:
            raw_content = uploaded_file.read().decode("utf-8", errors="ignore")
            st.write(f"**Uploaded File:** `{uploaded_file.name}` ({len(raw_content)} characters)")
            if st.button("Train Model on Uploaded File", type="primary"):
                st.session_state.corpus_text = raw_content
                train_current_model(raw_content, st.session_state.selected_model_order, st.session_state.k_smoothing)
                st.success("Custom corpus trained successfully!")

    with tab_custom:
        edited_text = st.text_area("Live Corpus Text:", value=st.session_state.corpus_text, height=220)
        if st.button("Retrain on Edited Text", type="secondary"):
            st.session_state.corpus_text = edited_text
            train_current_model(edited_text, st.session_state.selected_model_order, st.session_state.k_smoothing)
            st.success("Model retrained on updated text!")

    st.divider()

    # Live Corpus Dashboard Statistics
    st.subheader("📊 Live Corpus & Model Statistics")
    stats = st.session_state.corpus_stats or {}
    model = st.session_state.active_model

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Sentences</div>
            <div class="metric-num">{stats.get('num_sentences', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Word Tokens (N)</div>
            <div class="metric-num">{stats.get('num_tokens', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Vocabulary Size (|V|)</div>
            <div class="metric-num">{stats.get('vocab_size', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">N-grams Learned</div>
            <div class="metric-num">{len(model.ngram_counts) if model else 0}</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Smoothing ($k$)</div>
            <div class="metric-num">{st.session_state.k_smoothing}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Vocabulary Explorer
    with st.expander("🔍 Explore Vocabulary & Frequency Distribution", expanded=False):
        if model and model.unigram_counts:
            vocab_df = pd.DataFrame([
                {"Token": word, "Frequency Count": count, "Unigram Probability (%)": f"{(count / model.total_tokens)*100:.2f}%"}
                for word, count in model.unigram_counts.most_common(50)
            ])
            st.dataframe(vocab_df, use_container_width=True)

# ==========================================
# SECTION 3: PREDICTION PLAYGROUND
# ==========================================
elif "3. Prediction Playground" in menu:
    st.markdown('<div class="main-title">Next-Word Prediction Playground</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter a sentence or prefix and get dynamic real-time candidate predictions</div>', unsafe_allow_html=True)

    col_input, col_settings = st.columns([3, 1])

    with col_settings:
        top_k_options = st.selectbox("Number of Predictions (Top-K):", [3, 5, 10], index=1)
        show_chart = st.checkbox("Show Probability Bar Chart", value=True)
        include_eos = st.checkbox("Include End-of-Sentence Token (</s>)", value=False)

    with col_input:
        # Quick-fill buttons
        st.write("💡 **Sample Quick-Fill Prompts:**")
        quick_cols = st.columns(4)
        if quick_cols[0].button("Natural language processing is"):
            st.session_state.prompt_input = "Natural language processing is"
        if quick_cols[1].button("Machine learning"):
            st.session_state.prompt_input = "Machine learning"
        if quick_cols[2].button("Artificial intelligence is"):
            st.session_state.prompt_input = "Artificial intelligence is"
        if quick_cols[3].button("Python is a"):
            st.session_state.prompt_input = "Python is a"

        user_prompt = st.text_input(
            "Enter sentence context for prediction:",
            value=st.session_state.prompt_input,
            placeholder="Type your phrase here... e.g. Natural language processing is"
        )
        st.session_state.prompt_input = user_prompt

    predict_btn = st.button("🔮 Predict Next Word", type="primary", use_container_width=True)

    if predict_btn or user_prompt:
        model = st.session_state.active_model
        if not model or not model.is_trained:
            st.error("Model is not trained yet. Please visit Corpus & Training to train.")
        else:
            predictions = model.predict_next_words(user_prompt, top_k=top_k_options, include_eos=include_eos)

            st.markdown("### 🎯 Predicted Next Words")

            if not predictions:
                st.warning("No predictions could be generated for this context.")
            else:
                # Check if context was seen
                context_seen = predictions[0]["context_seen"]
                if not context_seen and model.n > 1:
                    st.info(f"ℹ️ **Note on Out-of-Context / Zero Count:** The context phrase was not directly observed in the {model_name_map[model.n]} training data. **Laplace (Add-{model.k}) smoothing** and backoff ranking were applied to generate reliable predictions.")

                # Table Display
                df_preds = pd.DataFrame([
                    {
                        "Rank": p["rank"],
                        "Predicted Word": p["word"],
                        "Corpus Co-occurrences": p["raw_count"],
                        "Conditional Probability": f"{p['probability']:.6f}",
                        "Relative Top Probability": p["percentage"],
                        "Absolute Probability": p["abs_percentage"]
                    }
                    for p in predictions
                ])

                # Visual Cards for Top 3
                card_cols = st.columns(min(len(predictions), 3))
                for i, c in enumerate(card_cols):
                    if i < len(predictions):
                        p = predictions[i]
                        badge_class = f"rank-{p['rank']}" if p['rank'] <= 3 else "rank-other"
                        with c:
                            st.markdown(f"""
                            <div class="metric-card" style="border-top: 4px solid {'#10B981' if i==0 else ('#3B82F6' if i==1 else '#8B5CF6')};">
                                <span class="rank-badge {badge_class}">Rank #{p['rank']}</span>
                                <div class="metric-num" style="color: {'#34D399' if i==0 else '#60A5FA'}; font-size: 1.6rem; margin-top: 0.5rem;">
                                    {p['word']}
                                </div>
                                <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 0.3rem;">
                                    Relative Prob: <b>{p['percentage']}</b><br>
                                    Raw Count: <code>{p['raw_count']}</code>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df_preds, use_container_width=True, hide_index=True)

                # Probability Chart
                if show_chart and len(predictions) > 0:
                    st.markdown("#### 📊 Candidate Probability Distribution")
                    chart_data = pd.DataFrame({
                        "Word": [p["word"] for p in predictions],
                        "Relative Probability (%)": [float(p["percentage"].replace("%", "")) for p in predictions]
                    }).set_index("Word")
                    st.bar_chart(chart_data)

            # Autonomous Multi-Word Continuation
            st.divider()
            st.subheader("⚡ Multi-Word Autocomplete / Sentence Continuation")
            st.write("Generate a continuous sequence of the most probable next words from your prompt:")
            gen_col1, gen_col2 = st.columns([3, 1])
            with gen_col2:
                num_continue_words = st.slider("Words to generate:", min_value=1, max_value=10, value=5)
            with gen_col1:
                if st.button("🚀 Generate Continuation"):
                    continuation = model.generate_continuation(user_prompt, num_words=num_continue_words)
                    st.success(f"**Generated Full Sentence:**\n\n> *\"{user_prompt} **{continuation}**\"*")

# ==========================================
# SECTION 4: MODEL EVALUATION & COMPARISON
# ==========================================
elif "4. Model Evaluation" in menu:
    st.markdown('<div class="main-title">Model Evaluation & Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empirical evaluation comparing Unigram, Bigram, and Trigram language models using real Perplexity and Accuracy metrics</div>', unsafe_allow_html=True)

    eval_col1, eval_col2 = st.columns([2, 1])
    with eval_col1:
        test_ratio = st.slider("Evaluation Test Split Ratio:", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
    with eval_col2:
        eval_k = st.slider("Evaluation Smoothing ($k$):", min_value=0.1, max_value=2.0, value=float(st.session_state.k_smoothing), step=0.1)

    if st.button("🚀 Run Full Model Benchmark Comparison", type="primary"):
        with st.spinner("Training and evaluating Unigram, Bigram, and Trigram models..."):
            comparison_df, models_dict = evaluate_models_comparison(
                st.session_state.corpus_text,
                test_size=test_ratio,
                k_smoothing=eval_k
            )

        st.markdown("### 🏆 Comprehensive Model Comparison Table")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### 📉 Perplexity Comparison (Lower is Better)")
            st.caption("Perplexity measures model surprise on unseen text. A lower score indicates higher confidence and predictive accuracy.")
            chart_pp = comparison_df[["Model", "Perplexity (Lower is better)"]].set_index("Model")
            st.bar_chart(chart_pp)

        with col_c2:
            st.markdown("#### 🎯 Top-K Accuracy Comparison (%)")
            st.caption("Percentage of times the true next word is included in the model's top candidates.")
            acc_data = []
            for _, row in comparison_df.iterrows():
                acc_data.append({
                    "Model": row["Model"],
                    "Top-1 Acc": float(row["Top-1 Accuracy (%)"].replace("%", "")),
                    "Top-3 Acc": float(row["Top-3 Accuracy (%)"].replace("%", "")),
                    "Top-5 Acc": float(row["Top-5 Accuracy (%)"].replace("%", ""))
                })
            df_acc = pd.DataFrame(acc_data).set_index("Model")
            st.bar_chart(df_acc)




