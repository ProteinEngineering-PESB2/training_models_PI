from transformers import AutoModel, AutoTokenizer

from embedding_extraction.embedding_based import EmbeddingBased


class ESMBasedEmbedding(EmbeddingBased):
    def __init__(
        self,
        name_device="cuda",
        name_model="facebook/esm2_t6_8M_UR50D",
        name_tokenizer="facebook/esm2_t6_8M_UR50D",
        dataset=None,
        column_seq=None,
        columns_ignore=[],
    ):
        super().__init__(
            name_device, name_model, name_tokenizer, dataset, column_seq, columns_ignore
        )

    def load_model_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_model, do_lower_case=False)

        self.model = AutoModel.from_pretrained(self.name_model).to(self.device)

        self.model.eval()
