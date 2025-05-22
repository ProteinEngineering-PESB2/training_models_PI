import sys

sys.path.insert(0, "../src/")

import pandas as pd

from embedding_extraction.prot5_based import Prot5Based

df_data = pd.read_csv(
    "../raw_data/Antimicrobial/train_data.csv"
)
df_data = df_data[:100]

df_data["sequence"] = df_data["sequence"].apply(lambda x: " ".join(x))

name_model = "Rostlab/ProstT5"

prot5_based = Prot5Based(
    name_device="cuda",
    dataset=df_data,
    name_model=name_model,
    name_tokenizer=name_model,
    column_seq="sequence",
    columns_ignore=["label"],
)

print("Loading model/tokenizer")
prot5_based.load_model_tokenizer()

print("Generating embedding")
df_embedding = prot5_based.embedding_process(batch_size=5)

print(df_embedding)

prot5_based.cleaning_memory()
print("Process finished")
