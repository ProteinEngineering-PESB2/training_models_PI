from transformers import BertModel, BertTokenizer

from embedding_extraction.embedding_based import EmbeddingBased


class BertBasedMebedding(EmbeddingBased):
    def __init__(
        self,
        name_device="cuda",
        name_model="Rostlab/prot_bert",
        name_tokenizer="Rostlab/prot_bert",
        dataset=None,
        column_seq=None,
        columns_ignore=[],
    ):
        super().__init__(
            name_device, name_model, name_tokenizer, dataset, column_seq, columns_ignore
        )

    def load_model_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.name_model, do_lower_case=False)

        self.model = BertModel.from_pretrained(self.name_model).to(self.device)

        self.model.eval()
