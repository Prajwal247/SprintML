import torch
import torch.nn as nn

from models.lora import create_lora_model


class PrefixLoRAModel(nn.Module):
    """Prefix tuning combined with LoRA — trains both prefix embeddings and low-rank adapters."""

    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        prefix_length: int = 10,
        lora_r: int = 8,
    ):
        super().__init__()
        self.lora_model = create_lora_model(base_model_name, num_labels, lora_r)
        self.num_labels = num_labels
        self.prefix_length = prefix_length
        self.hidden_size = self.lora_model.config.hidden_size

        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, self.hidden_size) * 0.01
        )

    def _get_base_bert(self):
        if hasattr(self.lora_model, "base_model"):
            return self.lora_model.base_model.model.bert
        return self.lora_model.bert

    def _get_classifier(self):
        if hasattr(self.lora_model, "base_model"):
            return self.lora_model.base_model.model.classifier
        return self.lora_model.classifier

    def forward(
        self, input_ids, attention_mask=None, labels=None, token_type_ids=None, **kwargs
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device
        seq_length = input_ids.size(1)
        base_bert = self._get_base_bert()

        word_embeds = base_bert.embeddings.word_embeddings(input_ids)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = base_bert.embeddings.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = base_bert.embeddings.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = base_bert.embeddings.LayerNorm(embeddings)
        embeddings = base_bert.embeddings.dropout(embeddings)

        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prefix, embeddings], dim=1)

        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, self.prefix_length, device=device)
            attention_mask = attention_mask.float()
            extended_2d = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            total_length = self.prefix_length + seq_length
            extended_2d = torch.ones(batch_size, total_length, device=device)

        # BERT encoder expects [batch, 1, 1, seq_len] additive mask
        extended_4d = extended_2d[:, None, None, :]
        extended_4d = (1.0 - extended_4d) * -10000.0

        encoder_outputs = base_bert.encoder(
            inputs_embeds, attention_mask=extended_4d
        )

        sequence_output = encoder_outputs[0]
        pooled_output = base_bert.pooler(sequence_output)
        logits = self._get_classifier()(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
