import sys

sys.path.insert(0, "../src/")

import pandas as pd

from embedding_extraction.mistral_based import MistralBasedEmbedding

df_data = pd.read_csv(
    "/home/dmedina/Desktop/tutorials/training_models_PI/raw_data/Antimicrobial/train_data.csv"
)
df_data = df_data[:100]

name_model = "RaphaelMourad/Mistral-Prot-v1-134M"

mistral_based = MistralBasedEmbedding(
    name_device="cuda",
    dataset=df_data,
    name_model=name_model,
    name_tokenizer=name_model,
    column_seq="sequence",
    columns_ignore=["label"],
)

print("Loading model/tokenizer")
mistral_based.load_model_tokenizer()

print("Generating embedding")
df_embedding = mistral_based.embedding_process(batch_size=50)

print(df_embedding)

mistral_based.cleaning_memory()
print("Process finished")
