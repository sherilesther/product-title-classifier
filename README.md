# Product Title Classifier

A machine learning pipeline to classify e-commerce product titles into specific categories using both traditional ML and deep learning approaches.

## 📋 Overview

This project implements two distinct modeling approaches for classifying fashion product titles:

1. **Traditional ML**: TF-IDF Vectorization with Logistic Regression
2. **Deep Learning**: Fine-tuned DistilBERT transformer model

## 🎯 Capabilities

### Current Features
- **Text Preprocessing**: Clean and normalize product titles (lowercasing, stopword removal, tokenization)
- **Dual Model Training**: Support for both classical ML and modern deep learning approaches
- **Model Persistence**: Save and load trained models for inference
- **Flexible Architecture**: Easily extendable to other product categories
- **Category Classification**: Classify product titles into categories like:
  - Women's Dresses
  - Men's Jeans
  - Women's Shoes
  - Men's Shoes
  - Men's Tops
  - Women's Jeans
  - Kids Tops
  - Men's Shorts

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd product_title_classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```bash
python setup.py
```

### Usage

#### Train TF-IDF Model (Traditional ML)
```bash
python src/train_tfidf.py
```

**Output:**
- Trained model: `models/tfidf_logreg.pkl`
- Vectorizer: `models/tfidf_logreg_vectorizer.pkl`
- Console: Classification report with precision, recall, F1-score

#### Train BERT Model (Deep Learning)
```bash
python src/train_bert.py
```

**Output:**
- Model checkpoint: `models/bert/`
- Tokenizer: `models/bert/`
- Label encoder: `models/bert/label_encoder.pkl`

**Note:** BERT training requires GPU for reasonable performance. On CPU, expect longer training times.

## 📁 Project Structure

```
product_title_classifier/
├── data/
│   └── sample_fashion.csv       # Sample dataset (10 fashion items)
├── models/                       # Trained models (auto-generated)
│   ├── tfidf_logreg.pkl
│   ├── tfidf_logreg_vectorizer.pkl
│   └── bert/                    # BERT model directory
├── src/
│   ├── preprocess.py            # Text cleaning and data loading
│   ├── train_bert.py            # BERT model training script
│   ├── train_tfidf.py           # TF-IDF model training script
│   └── utils.py                 # Helper functions
├── requirements.txt             # Python dependencies
├── setup.py                     # NLTK data downloader
└── README.md                    # This file
```

## 🔧 Technical Details

### Data Format
The input CSV file must contain these columns:
- `title`: Product title text
- `category`: Product category label

Example:
```csv
title,category
"Floral chiffon midi dress","Women's Dresses"
"Men's slim fit blue jeans","Men's Jeans"
```

### Preprocessing Pipeline
1. **Lowercase conversion**: Standardize text case
2. **Special character removal**: Keep only alphanumeric characters
3. **Tokenization**: Split text into words using NLTK
4. **Stopword removal**: Remove common English words (a, the, is, etc.)
5. **Rejoin**: Create cleaned text string

### Model Architectures

#### TF-IDF + Logistic Regression
- **Vectorization**: TF-IDF with unigrams and bigrams (1,2)
- **Classifier**: Logistic Regression (max_iter=2000)
- **Train/Test Split**: 80/20
- **Advantages**: Fast training, interpretable, low resource requirements

#### DistilBERT
- **Base Model**: `distilbert-base-uncased` (pre-trained)
- **Task**: Sequence classification
- **Training**: 2 epochs, batch size 4
- **Advantages**: State-of-the-art accuracy, context-aware embeddings

## 📊 Sample Dataset

The project includes a sample dataset with 10 fashion product examples:
- 2 Women's Dresses
- 3 Women's Shoes
- 2 Men's Jeans
- 2 Men's Shoes
- And more...

**Note:** For production use, train on a larger, more diverse dataset (1000+ samples recommended).

## 🛠️ Extending the Project

### Add More Categories
1. Update your CSV with new categories
2. Re-run training scripts
3. Models automatically adapt to new label counts

### Use Your Own Data
Replace `data/sample_fashion.csv` with your dataset maintaining the same format (title, category columns).

### Inference Example
```python
from utils import load_model
import joblib

# Load TF-IDF model
model, vectorizer = load_model("models/tfidf_logreg.pkl", vectorizer=True)

# Predict
from preprocess import clean_text
title = "Blue denim skinny jeans"
clean = clean_text(title)
vec = vectorizer.transform([clean])
prediction = model.predict(vec)
print(f"Predicted category: {prediction[0]}")
```

## 📦 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: ML algorithms and metrics
- **nltk**: Natural language processing
- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers
- **joblib**: Model serialization
- **tqdm**: Progress bars

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add inference scripts
- Implement cross-validation
- Support for multilingual titles
- API endpoint for real-time classification
- Docker containerization

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Ester's Project

## 🙏 Acknowledgments

- NLTK for text preprocessing tools
- Hugging Face for transformer models
- scikit-learn for classical ML algorithms
