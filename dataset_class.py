import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from model import get_feature_extractor

# Fixed time dimension for mel-spectrograms (pad or truncate to this length)
MAX_TIME_FRAMES = 1024


class MyDataset(Dataset):
    """
    Dataset that wraps raw audio waveforms and applies the AST feature
    extractor on-the-fly, returning input_values ready for the AST model.

    Can also accept pre-extracted mel-spectrograms (as numpy arrays) if
    ``use_ast_extractor=False``.
    """

    def __init__(self, features, labels, sampling_rate: int = 16000,
                 use_ast_extractor: bool = True,
                 max_time_frames: int = MAX_TIME_FRAMES):
        """
        :param features: list of 1-D numpy arrays (raw waveforms) or
            pre-extracted mel-spectrograms.
        :param labels: list of integer class labels.
        :param sampling_rate: audio sampling rate (default 16000).
        :param use_ast_extractor: if True, apply AST feature extractor to
            raw waveforms; if False, features are used as-is.
        :param max_time_frames: fixed time dimension to pad/truncate mel
            spectrograms to (only used when use_ast_extractor=False).
        """
        super().__init__()
        self.features = features
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.use_ast_extractor = use_ast_extractor
        self.max_time_frames = max_time_frames

        if self.use_ast_extractor:
            self.feature_extractor = get_feature_extractor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.use_ast_extractor:
            # feature is a raw waveform (1-D numpy array)
            inputs = self.feature_extractor(
                feature,
                sampling_rate=self.sampling_rate,
                padding="max_length",
                truncation=True,
                max_length=16000 * 10,  # 10 seconds max
                return_tensors="pt",
            )
            feature = inputs["input_values"].squeeze(0)  # (time_frames, freq_bins)
        else:
            # feature is a pre-extracted mel-spectrogram of shape (n_mels, time)
            feature = torch.tensor(np.array(feature), dtype=torch.float32)

            # Pad or truncate along the time axis to fixed length
            time_len = feature.shape[1]
            if time_len < self.max_time_frames:
                # Pad with zeros on the right
                feature = F.pad(feature, (0, self.max_time_frames - time_len))
            elif time_len > self.max_time_frames:
                # Truncate
                feature = feature[:, :self.max_time_frames]

        label = torch.tensor(int(label), dtype=torch.long)
        return feature, label

