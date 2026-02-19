import torch
import torch.nn as nn


class SoftPromptModel(nn.Module):
    """Learnable continuous embeddings prepended to input while the base model stays frozen."""

    def __init__(self, base_model, num_labels: int, prompt_length: int = 10):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.prompt_length = prompt_length

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.embedding_dim = self.base_model.config.hidden_size
        self.soft_prompt = nn.Parameter(
            torch.randn(prompt_length, self.embedding_dim) * 0.01
        )

        if hasattr(self.base_model, "classifier"):
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        embeddings = self.base_model.bert.embeddings.word_embeddings(input_ids)
        batch_size = embeddings.size(0)

        soft_prompts = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([soft_prompts, embeddings], dim=1)

        if attention_mask is not None:
            prompt_attention = torch.ones(
                batch_size,
                self.prompt_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        outputs = self.base_model.bert(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        logits = self.base_model.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
