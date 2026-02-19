import torch
import torch.nn as nn

from models.lora import create_lora_model


class SoftPromptLoRAModel(nn.Module):
    """Soft prompts combined with LoRA — trains both prompt embeddings and low-rank adapters."""

    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        prompt_length: int = 10,
        lora_r: int = 8,
    ):
        super().__init__()
        self.lora_model = create_lora_model(base_model_name, num_labels, lora_r)
        self.num_labels = num_labels
        self.prompt_length = prompt_length
        self.embedding_dim = self.lora_model.config.hidden_size

        self.soft_prompt = nn.Parameter(
            torch.randn(prompt_length, self.embedding_dim) * 0.01
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        embeddings = (
            self.lora_model.base_model.model.bert.embeddings.word_embeddings(input_ids)
        )
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

        outputs = self.lora_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
