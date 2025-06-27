# 🔮 MultiCPF: A Multimodal Crystal Property Prediction Framework

MultiCPF is a modular, extensible framework for crystal property prediction via multimodal fusion of structural and textual representations. It is built upon **CrystalFormer** as the structure encoder and **MatSciBERT** for textual understanding, supporting various fusion mechanisms including sum, concat, gated, and cross-attention.

## ✨ Highlights

- ⚙️ **Modular Design**: Structure encoder, text encoder, and fusion block are independently pluggable.
- 🔀 **Fusion Flexibility**: Supports `concat`, `sum`, `gated`, and `cross-attn` fusion modes.
- 🧠 **Interpretability**: Token-level cross-modal attention visualization and ablation tools.
- 🧪 **Training Strategy**: Modality masking, data augmentation, and multitask loss supported.

## 📁 Project Structure

```bash
MultiCPF/
│
├── models/                  # CrystalFormer & fusion model
├── dataloaders/            # Dataset preparation
├── multimodal_fusion/      # Text encoder, fusion block, loss, tools
├── train_fusion.py         # Main training script for multimodal model
├── utils.py                # General utilities
├── README.md
├── LICENSE
└── download_models.sh      # 🆕 Script to download LLM weights
````

## 🚀 Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/Bi8bo001/MultimodalCPF.git
cd MultimodalCPF
pip install -r requirements.txt  # optional
```

### 2. Download pretrained language models

```bash
bash download_models.sh
```

### 3. Run multimodal training

```bash
bash train_fusion.sh
```

## 📜 License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.

## 🙋 Citation

Coming soon (under submission). If you use this work in your research, please consider citing us.
