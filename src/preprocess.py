import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")

stop = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop]
    return " ".join(tokens)

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(subset=["title", "category"], inplace=True)
    df["clean_title"] = df["title"].apply(clean_text)
    return df
