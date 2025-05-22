import sys

sys.path.insert(0, "../src/")

import pandas as pd

from embedding_extraction.ankh2_based import Ankh2BasedEmbedding

df_data = pd.read_csv(
    "../raw_data/Antimicrobial/train_data.csv"
)
df_data = df_data[:100]

name_model = "ElnaggarLab/ankh2-ext2"

ankh2_based = Ankh2BasedEmbedding(
    name_device="cuda",
    dataset=df_data,
    name_model=name_model,
    name_tokenizer=name_model,
    column_seq="sequence",
    columns_ignore=["label"],
)

print("Loading model/tokenizer")
ankh2_based.load_model_tokenizer()

print("Generating embedding")
df_embedding = ankh2_based.embedding_process(batch_size=50)

print(df_embedding)

ankh2_based.cleaning_memory()
print("Process finished")
