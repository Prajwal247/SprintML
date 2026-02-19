import torch
import torch.nn as nn


class PrefixTuningModel(nn.Module):
    """Learnable prefix embeddings prepended to each input while the base model stays frozen."""

    def __init__(self, base_model, num_labels: int, prefix_length: int = 10):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.prefix_length = prefix_length

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.hidden_size = self.base_model.config.hidden_size
        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, self.hidden_size) * 0.01
        )

        if hasattr(self.base_model, "classifier"):
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True

    def forward(
        self, input_ids, attention_mask=None, labels=None, token_type_ids=None, **kwargs
    ):
        batch_size = input_ids.size(0)
        inputs_embeds = self.base_model.bert.embeddings.word_embeddings(input_ids)

        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)

        if attention_mask is not None:
            if attention_mask.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                attention_mask = attention_mask.float()
            prefix_attention = torch.ones(
                batch_size,
                self.prefix_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        if token_type_ids is not None:
            prefix_token_types = torch.zeros(
                batch_size,
                self.prefix_length,
                dtype=token_type_ids.dtype,
                device=token_type_ids.device,
            )
            token_type_ids = torch.cat([prefix_token_types, token_type_ids], dim=1)

        outputs = self.base_model.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs.pooler_output
        logits = self.base_model.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
