import sys

sys.path.insert(0, "../src/")

import pandas as pd

from embedding_extraction.esm_based import ESMBasedEmbedding

df_data = pd.read_csv(
    "/home/dmedina/Desktop/tutorials/training_models_PI/raw_data/Antimicrobial/train_data.csv"
)

name_model = "facebook/esm2_t36_3B_UR50D"

esm_based = ESMBasedEmbedding(
    name_device="cuda",
    dataset=df_data,
    name_model=name_model,
    name_tokenizer=name_model,
    column_seq="sequence",
    columns_ignore=["label"],
)

print("Loading model/tokenizer")
esm_based.load_model_tokenizer()

print("Generating embedding")
df_embedding = esm_based.embedding_process(batch_size=50)

print(df_embedding)

esm_based.cleaning_memory()
print("Process finished")
