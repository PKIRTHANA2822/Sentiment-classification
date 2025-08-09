# Project title
- Sentiment Classification — End-to-End (Count / TF-IDF → Stemming → NB / Logistic / Tree models)
<img width="1030" height="463" alt="Screenshot 2025-08-09 161002" src="https://github.com/user-attachments/assets/c6eaaaf9-db34-4134-9ffa-21ebece80639" />
# Objective
- Build a reliable binary sentiment classifier that maps raw review text to positive (1) or negative (0) labels. Deliver a reproducible NLP pipeline that includes text cleaning, vectorization, model comparison, - evaluation, and interpretable outputs for downstream use.
# Why we do this project
- Business value: automate review moderation, customer feedback routing, or product sentiment monitoring.
- Technical value: learn practical NLP preprocessing (stop-words, stemming/lemmatization, n-grams), sparse vector models (Count / TF-IDF), and basic supervised modeling and evaluation.
- Operational value: produce a fast, lightweight classifier suitable for production where deep models may be overkill.
# Step-by-step approach (high level)
- Data load & quick checks — confirm shape, missing values, and label distribution.
- Exploratory data analysis (EDA) — class balance, common words, rare words, word clouds.
- Text cleaning & preprocessing — lowercasing, punctuation handling, stop-words, optional lemmatization/stemming.
- Vectorization — CountVectorizer and TF-IDF with configurable max_features and ngram_range.
- Feature selection / pruning — remove extremely rare words, keep top-k features.
- Modeling — train multiple models (BernoulliNB, LogisticRegression, RandomForest, DecisionTree, KNN) and tune hyperparameters.
- Evaluation — cross-validation and hold-out test set using accuracy, precision, recall, F1, confusion matrix; report per-class metrics.
- Interpretation & reporting — top predictive words per class, error analysis, and visualizations.
- Deliverables — save vectorizers, trained models, predictions, and a short report.
- Exploratory Data Analysis (what to inspect)
- Dataset size & format: number of documents, fields (text + label).
- Class distribution: check % of positive vs negative to identify imbalance (the notebook used ~6.9k rows).
- Document length distribution: mean/median words per review.
- Most frequent words and rare words: how many features appear only once (candidates for removal).
- Word cloud: quick visual of dominant tokens.
- Top words by sentiment: bar charts showing counts of top tokens separated by label to validate signal.
# Feature selection (what to keep & why)
- Vocabulary pruning: drop tokens that occur once or extremely rarely; keep top max_features (e.g., 1000) to reduce noise and dimensionality.
- Stop-words removal: remove generic English stop-words and domain artifacts (e.g., dataset-specific tokens like author names).
- Token selection criterion: choose tokens based on document frequency and usefulness in separating classes (inspect per-token sentiment counts).
# Feature engineering (common steps used)
- Stemming / lemmatization: reduce word forms to roots (Porter stemmer in the notebook; lemmatizer optional if WordNet is available).
- n-grams: include bigrams (1,2) to capture short phrases (e.g., “not good”).
- Binary vs frequency features: BernoulliNB benefits from binary presence; TF-IDF or counts for other models.
- Custom tokenizer / analyzer: combine tokenization + cleaning + stemming inside vectorizer for reproducibility.
- Stopword augmentation: add domain-specific noise words (high-frequency but uninformative terms) to the stop list.
# Model training (what was tried & rationale)
- Naive Bayes (BernoulliNB): very fast baseline for binary text features; performed well in bag-of-words.
### Logistic Regression: strong linear baseline for TF-IDF; good calibration and interpretability.
- Tree-based / others (RandomForest, DecisionTree, KNN): explored for comparison — trees tolerate non-linearities and mixed features.
### Cross-validation: use stratified folds to estimate generalization; measure mean accuracy and variance.
### Model selection: pick model balancing accuracy, complexity, and inference latency (Logistic often best tradeoff for TF-IDF).
# Model testing & evaluation (metrics & findings)
- Metrics to report: Accuracy, Precision, Recall, F1-score (per class), Confusion Matrix. ROC-AUC optional for probability-based models.
- Typical outcomes (from notebook runs):
- BernoulliNB: ~98% accuracy on hold-out in this run.
- Logistic Regression: ~99% accuracy (strong performer when using TF-IDF + n-grams).
- Error analysis: examine false positives and false negatives to find systematic issues (negation, sarcasm, unseen vocabulary).
- Cross-validation stability: check CV mean ± std; guard against overly optimistic single-split results.
# Output (deliverables & artifacts)
<img width="602" height="149" alt="Screenshot 2025-08-09 161042" src="https://github.com/user-attachments/assets/6eb57450-eb23-4b1a-8fdb-45b6b15e7bf2" />
