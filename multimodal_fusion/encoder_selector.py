
# encoder_selector.py
### train_fusion.py: import get_dataset_and_encoder

def get_dataset_and_encoder(params):
    encoder_name = getattr(params, "encoder_name", None)
    fusion_type = getattr(params, "fusion_type", "none")

    assert encoder_name is not None, "Missing required param: encoder_name"
    assert fusion_type in ["none", "sum", "concat", "gated", "cross_attn"], f"Invalid fusion_type: {fusion_type}"

    ## registry-based clean design
    encoder_registry = {
        "latticeformer": {
            "structure_encoder": "models.latticeformer.Latticeformer",
            "dataset_unimodal": "dataloaders.dataset_latticeformer.RegressionDatasetMP_Latticeformer",
            "dataset_multimodal": "multimodal_fusion.dataset_fusion.FusionDatasetMP_Latticeformer",
        },
        # "potnet": { ... },
        # "megnet": { ... },
    }

    if encoder_name not in encoder_registry:
        raise NotImplementedError(f"Unsupported encoder_name: {encoder_name}")

    import importlib

    # import structure encoder class
    encoder_path = encoder_registry[encoder_name]["structure_encoder"]
    encoder_module, encoder_classname = encoder_path.rsplit(".", 1)
    StructureEncoder = getattr(importlib.import_module(encoder_module), encoder_classname)

    # import dataset class
    dataset_key = "dataset_multimodal" if fusion_type != "none" else "dataset_unimodal"
    dataset_path = encoder_registry[encoder_name][dataset_key]
    dataset_module, dataset_classname = dataset_path.rsplit(".", 1)
    Dataset = getattr(importlib.import_module(dataset_module), dataset_classname)

    return Dataset, StructureEncoder
