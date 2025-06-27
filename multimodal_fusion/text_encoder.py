
import torch
import torch.nn as nn
from typing import List
from transformers import AutoTokenizer, AutoModel
from tokenizers.normalizers import BertNormalizer


class MatSciBERTEncoder(nn.Module):
    def __init__(
        self, 
        model_path: str, 
        output_dim: int, 
        dropout: float = 0.1,
        use_token_output: bool = False  ## 根据cross-attn判断是否token-level
    ):
        super().__init__()
        self.use_token_output = use_token_output

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        self.normalizer = BertNormalizer(
            lowercase=False,
            strip_accents=True,
            clean_text=True,
            handle_chinese_chars=True
        )

        self.projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def normalize(self, text: str) -> str:
        return self.normalizer.normalize_str(text)

    def forward(self, texts: List[str]) -> torch.Tensor:
        normed = [self.normalize(t) for t in texts]
        encodings = self.tokenizer(
            normed, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512)
        encodings = {k: v.to(next(self.model.parameters()).device) for k, v in encodings.items()}

        with torch.no_grad():
            output = self.model(**encodings).last_hidden_state  # [B, L, H]
            projected = self.projection(output)  # [B, L, D]

        ### 根据是否cross-attn决定输出维度
        if self.use_token_output:
            return projected  # [B, L, D]
        else:  # embedding
            return projected[:, 0, :]  # [B, D]


class GenericBERTEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        output_dim: int,
        dropout: float = 0.1,
        use_token_output: bool = False
    ):
        super().__init__()
        self.use_token_output = use_token_output

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        self.projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        encodings = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        )
        encodings = {k: v.to(next(self.model.parameters()).device) for k, v in encodings.items()}

        with torch.no_grad():
            output = self.model(**encodings).last_hidden_state  # [B, L, H]
            projected = self.projection(output)  # [B, L, D]

        if self.use_token_output:
            return projected  # [B, L, D]
        else:
            return projected[:, 0, :]  # [B, D]


def build_text_encoder(params) -> nn.Module:
    encoder_name = getattr(params, "text_encoder_name", "matscibert").lower()  # lowercase for robust matching
    model_path = getattr(params, "text_encoder_path", None)
    output_dim = getattr(params, "model_dim", 128)
    dropout = getattr(params, "text_dropout", 0.1)
    fusion_type = getattr(params, "fusion_type", "concat").lower()

    use_token_output = (fusion_type == "cross_attn")

    # Default model paths if not specified
    if model_path is None:
        if encoder_name == "matscibert":
            model_path = "m3rg-iitd/matscibert"
        elif encoder_name == "bert":
            model_path = "google-bert/bert-base-uncased"
        else:
            raise ValueError(f"Unsupported text_encoder_name with no model_path provided: {encoder_name}")

    if encoder_name == "matscibert":
        encoder = MatSciBERTEncoder(model_path, output_dim, dropout, use_token_output)
    elif encoder_name == "bert":
        encoder = GenericBERTEncoder(model_path, output_dim, dropout, use_token_output)
    else:
        raise ValueError(f"Unsupported text_encoder_name: {encoder_name}")

    if getattr(params, "freeze_text_encoder", True):
        for p in encoder.parameters():
            p.requires_grad = False

    print(f"[TextEncoder] Initialized '{encoder_name}' with token_output={use_token_output}, freeze={getattr(params, 'freeze_text_encoder', True)}")

    return encoder
