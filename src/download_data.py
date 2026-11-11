from datasets import load_dataset
import os

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

ds = load_dataset("stanfordnlp/imdb")

train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()

train_df.to_csv(os.path.join(RAW_DIR, "imdb_train.csv"), index=False)
test_df.to_csv(os.path.join(RAW_DIR, "imdb_test.csv"), index=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df["label"].value_counts())
print(train_df.head())