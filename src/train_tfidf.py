from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocess import load_data
from utils import save_model, print_box

def main():
    print_box("Loading dataset...")
    df = load_data("data/sample_fashion.csv")

    X = df["clean_title"]
    y = df["category"]

    print_box("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print_box("Vectorizing text...")
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    print_box("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    print_box("Model Evaluation")
    print(classification_report(y_test, preds))

    print_box("Saving model...")
    save_model(clf, tfidf, path="models", name="tfidf_logreg")

if __name__ == "__main__":
    main()
