import sys
sys.path.insert(0, "../src/")

from embedding_extraction.Ankh2Based import Ankh2BasedEmbedding
import pandas as pd

df_data = pd.read_csv("/home/dmedina/Desktop/tutorials/training_models_PI/raw_data/Antimicrobial/train_data.csv")
df_data = df_data[:100]

name_model = "ElnaggarLab/ankh2-ext2"

ankh2_based = Ankh2BasedEmbedding(
    name_device="cuda", 
    dataset=df_data, 
    name_model=name_model,
    name_tokenizer=name_model,
    column_seq="sequence",
    columns_ignore=["label"]
)

print("Loading model/tokenizer")
ankh2_based.loadModelTokenizer()

print("Generating embedding")
df_embedding = ankh2_based.embeddingProcess(batch_size=50)

print(df_embedding)

ankh2_based.cleaning_memory()
print("Process finished")
