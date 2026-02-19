import torch
import torch.nn as nn


class SingleLayerModel(nn.Module):
    """A single trainable classification head on top of a fully frozen BERT."""

    def __init__(self, base_model, num_labels: int, hidden_dim: int = 128):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        with torch.no_grad():
            outputs = self.base_model.bert(
                input_ids=input_ids, attention_mask=attention_mask
            )

        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
