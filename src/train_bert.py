import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from preprocess import load_data
from utils import print_box
import joblib

class TitleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    print_box("Loading dataset...")
    df = load_data("data/sample_fashion.csv")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["category"])

    dataset = TitleDataset(
        df["title"].tolist(),
        df["label"].values,
        tokenizer
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_encoder.classes_)
    )

    args = TrainingArguments(
        output_dir="./models/bert",
        per_device_train_batch_size=4,
        num_train_epochs=2,
        save_strategy="epoch",
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)

    print_box("Training BERT model...")
    trainer.train()

    print_box("Saving model...")
    model.save_pretrained("models/bert")
    tokenizer.save_pretrained("models/bert")
    joblib.dump(label_encoder, "models/bert/label_encoder.pkl")

if __name__ == "__main__":
    main()
