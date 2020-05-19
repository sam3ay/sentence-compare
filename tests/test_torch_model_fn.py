import pytest
from model.code.torch_constructor import model_fn
import torch

model_dir = "/workspaces/sentence-compare/model/roberta-large-nli-stsb-mean-tokens"


def test_model_load():
    model = model_fn(model_dir)
    assert isinstance(model, torch.nn.Module)
