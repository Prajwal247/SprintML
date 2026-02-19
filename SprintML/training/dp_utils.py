import torch
import torch.nn as nn


class DPLayerNorm(nn.Module):
    """LayerNorm without internal buffers so Opacus can handle it."""

    def __init__(self, normalized_shape, eps: float = 1e-12):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        return nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )


def fix_model_for_opacus(model):
    """Replace LayerNorm with DPLayerNorm and strip padding_idx from Embeddings."""
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            dp_ln = DPLayerNorm(module.normalized_shape, eps=module.eps)
            dp_ln.weight = nn.Parameter(module.weight.data.clone())
            dp_ln.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, dp_ln)
        elif isinstance(module, nn.Embedding):
            new_emb = nn.Embedding(module.num_embeddings, module.embedding_dim)
            new_emb.weight = nn.Parameter(module.weight.data.clone())
            setattr(model, name, new_emb)
        else:
            fix_model_for_opacus(module)
