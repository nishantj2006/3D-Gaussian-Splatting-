import torch


class FeatureMapCache:
    """Keep normalized teacher maps on a chosen device for repeated views."""

    def __init__(self, device="cuda", expected_channels=None):
        self.device = device
        self.expected_channels = expected_channels
        self._maps = {}

    def get(self, source):
        if not isinstance(source, str):
            return load_feature_map(source, self.device, self.expected_channels)
        feature = self._maps.get(source)
        if feature is None:
            feature = load_feature_map(source, self.device, self.expected_channels)
            self._maps[source] = feature
        return feature

    def __len__(self):
        return len(self._maps)


def feature_channels(shape):
    """Infer the channel axis of a 3D CHW or HWC feature map."""
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D feature map, got shape {tuple(shape)}")

    # Generated maps and *_fmap_CxHxW.pt files are channel-first. Prefer that
    # convention when spatial dimensions happen to look like channel counts.
    if shape[0] in (16, 32, 64, 128, 256, 512):
        return int(shape[0])

    # Fallback for older HWC maps from apply_pca.py/extract_features.py.
    first_is_channel = shape[0] <= 512 and shape[0] <= shape[1] and shape[0] <= shape[2]
    last_is_channel = shape[2] <= 512 and shape[2] <= shape[0] and shape[2] <= shape[1]
    if first_is_channel and not last_is_channel:
        return int(shape[0])
    if last_is_channel and not first_is_channel:
        return int(shape[2])
    if shape[2] in (16, 32, 64, 128, 256, 512):
        return int(shape[2])
    raise ValueError(f"Cannot infer channel axis for feature map shape {tuple(shape)}")


def load_feature_map(source, device="cuda", expected_channels=None):
    """Load a feature map and normalize it to contiguous float32 CHW."""
    feature = torch.load(source, map_location="cpu") if isinstance(source, str) else source
    if feature.ndim != 3:
        raise ValueError(f"Expected a 3D feature map, got shape {tuple(feature.shape)}")

    channels = expected_channels or feature_channels(feature.shape)
    if feature.shape[0] == channels:
        pass
    elif feature.shape[-1] == channels:
        feature = feature.permute(2, 0, 1)
    else:
        raise ValueError(
            f"Feature map shape {tuple(feature.shape)} does not contain {channels} channels"
        )
    return feature.contiguous().float().to(device)
