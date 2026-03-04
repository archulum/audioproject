import torch
import torch.nn as nn
from transformers import ASTModel, ASTFeatureExtractor


class AudioClassifier(nn.Module):
    """
    Audio classifier using the Audio Spectrogram Transformer (AST) as a
    frozen feature extractor with a custom MLP classification head.

    The AST model (MIT/ast-finetuned-audioset-10-10-0.4593) converts
    mel-spectrogram inputs into rich audio embeddings. These embeddings
    are then passed through trainable MLP layers for classification.
    """

    def __init__(self, num_classes: int = 23, freeze_ast: bool = True, dropout: float = 0.3):
        """
        :param num_classes: Number of output classes (default 23 for BSD dataset).
        :param freeze_ast: Whether to freeze the AST backbone weights.
        :param dropout: Dropout probability for the MLP head.
        """
        super().__init__()


        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.hidden_size = self.ast.config.hidden_size  # 768

        # Freeze AST parameters so only the MLP head is trained
        if freeze_ast:
            for param in self.ast.parameters():
                param.requires_grad = False

        # Custom MLP classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param input_values: Mel-spectrogram tensor of shape
            (batch, freq_bins, time_frames) as produced by ASTFeatureExtractor.
        :returns: Logits tensor of shape (batch, num_classes).
        """

        outputs = self.ast(input_values=input_values)


        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        logits = self.classifier(cls_embedding)
        return logits


def get_feature_extractor() -> ASTFeatureExtractor:
    """
    Returns the AST feature extractor configured for the pre-trained model.
    Use this to preprocess raw audio waveforms before feeding them to the model.

    Usage:
        feature_extractor = get_feature_extractor()
        inputs = feature_extractor(
            raw_audio_list,          # list of 1-D numpy arrays (waveforms)
            sampling_rate=16000,
            padding="max_length",
            return_tensors="pt",
        )
        logits = model(inputs["input_values"])
    """
    return ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")


if __name__ == "__main__":

    model = AudioClassifier(num_classes=23, freeze_ast=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Dummy forward pass (AST default input: 1024 time frames x 128 mel bins)
    dummy_input = torch.randn(2, 1024, 128)
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}")  # Expected: (2, 23)
