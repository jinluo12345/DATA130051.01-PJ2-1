import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageClassification

def get_model(model_name: str, pretrained: bool, num_classes: int):
    """
    Load a Huggingface model and replace its head.
    model_name: your HF repo path (e.g. 'microsoft/resnet-18')
    pretrained: whether to load pretrained weights
    num_classes: 101 for Caltech-101
    """
    config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
    if pretrained:
        model = AutoModelForImageClassification.from_pretrained(model_name, config=config,ignore_mismatched_sizes=True)
    else:
        model = AutoModelForImageClassification.from_config(config)
    return model
