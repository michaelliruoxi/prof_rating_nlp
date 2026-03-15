# Professor Rating Prediction Using NLP

A multi-model NLP pipeline that predicts professor ratings (1.0 - 5.0) from student review comments. The project compares six different classification approaches — from traditional linear classifiers to deep learning — while addressing class imbalance through oversampling and text augmentation.

## Dataset

**Source:** RateMyProfessor reviews (`RateMyProfessor-9c.csv`)

- **19,994 student reviews** across multiple universities, departments, and states
- **29 columns** including professor metadata, student comments, and binary teaching tags
- **Target variable:** `student_star` — rating on a 1.0 to 5.0 scale (0.5 increments, 9 classes)
- **Text feature:** `comments` — free-text student review
- **Binary tags (22):** `tough_grader`, `amazing_lectures`, `caring`, `hilarious`, `inspirational`, `gives_good_feedback`, `extra_credit`, and more

## Models

| # | Model | Data Strategy |
|---|-------|---------------|
| 1 | RidgeClassifier (Baseline) | No balancing |
| 2 | RidgeClassifier | Random oversampling (imbalanced-learn) |
| 3 | RidgeClassifier | Text augmentation |
| 4 | XGBoost | Text augmentation |
| 5 | Random Forest | Text augmentation |
| 6 | Bidirectional LSTM | Text augmentation |

### LSTM Architecture (Model 6)

```
Embedding → BiLSTM(128) → BiLSTM(64) → Dense(128, ReLU) → Dropout(0.5) → Dense(64, ReLU) → Dropout(0.5) → Softmax
```

- Bidirectional layers capture context from both directions
- Categorical crossentropy loss with softmax output
- Trained on padded token sequences from augmented data

## Pipeline

```
Raw CSV Data
  ├── Data Exploration & Statistics
  ├── NLTK Tokenization (word_tokenize)
  ├── TF-IDF Vectorization (max_features=5000, min_df=2)
  ├── Stratified Train/Test Split (80/20)
  │
  ├── Model 1: Baseline (no balancing)
  ├── Model 2: Random Oversampling
  ├── Text Augmentation (synonym replacement + random deletion)
  │     ├── Model 3: Ridge + Augmentation
  │     ├── Model 4: XGBoost + Augmentation
  │     ├── Model 5: Random Forest + Augmentation
  │     └── Model 6: BiLSTM + Augmentation
  │
  ├── Model Comparison (accuracy, precision, recall, F1)
  └── Word Frequency Analysis by Rating Category
```

## Text Augmentation

To address class imbalance, minority classes are augmented using:

- **Synonym replacement** — randomly swaps words with WordNet synonyms (p=0.3)
- **Random word deletion** — drops words with probability p=0.15
- Augmentation factor computed per class, capped at 3 augmentations per sample

## Visualizations

- Class distribution histograms (before/after balancing)
- Confusion matrices for each model
- Classification reports (precision, recall, F1 per class)
- Model comparison table across all 6 models
- Top-10 most frequent words per rating category (1.0 through 5.0)

## Setup

### Requirements

Python 3.x with dependencies listed in `requirements_container.txt`.

```bash
pip install -r requirements_container.txt
```

**Key dependencies:** pandas, numpy, scikit-learn, xgboost, tensorflow, nltk, imbalanced-learn, matplotlib, seaborn

### NLTK Data

The notebook automatically downloads required NLTK resources:

- `punkt`, `punkt_tab` (tokenization)
- `wordnet`, `omw-1.4` (synonym replacement)
- `stopwords` (word frequency analysis)

## Usage

Run the Jupyter notebook end-to-end:

```bash
jupyter notebook model1.ipynb
```

The notebook executes the full pipeline — data loading, preprocessing, training all 6 models, evaluation, and visualization.

## Project Structure

```
prof_rating_nlp/
├── model1.ipynb                 # Full pipeline notebook (all models)
├── RateMyProfessor-9c.csv       # Dataset (19,994 reviews)
├── requirements_container.txt   # Python dependencies
└── README.md
```
