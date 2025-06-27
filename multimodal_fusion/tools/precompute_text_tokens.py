
import argparse
import json
import os
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


class MatSciBERTTokenEncoder(nn.Module):
    def __init__(self, model_name="m3rg-iitd/matscibert", output_dim=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.model.config.hidden_size, output_dim)

    def forward(self, text: str) -> torch.Tensor:
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            output = self.model(**encoded_input).last_hidden_state  # [1, L, H]
            projected = self.projection(output)  # [1, L, D]
        return projected  # no squeeze here, keep [1, L, D]


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_embedding_dict(embedding_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embedding_dict, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="Path to train/val/test.json")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save embedding .pt file")
    parser.add_argument("--model_name", type=str, default="m3rg-iitd/matscibert")
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model = MatSciBERTTokenEncoder(args.model_name, args.output_dim).to(args.device)
    model.eval()

    dataset = load_json(args.input_json)

    emb_dict = {}
    with torch.no_grad():
        for item in tqdm(dataset):
            mat_id = str(item["id"])
            text = item["text"]
            emb = model(text).squeeze(0).cpu()  # [L, D]
            emb_dict[mat_id] = emb

    save_embedding_dict(emb_dict, args.output_path)
    print(f"Saved {len(emb_dict)} token-level embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
