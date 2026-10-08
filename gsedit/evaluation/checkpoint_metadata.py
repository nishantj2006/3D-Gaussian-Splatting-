"""Load legacy LaMa tensor weights without executing trainer/config classes."""
from collections import defaultdict
from typing import Any


class MetadataRecord:
    """Inert container for unused legacy training metadata, not executable code."""
    pass


def load_lama_tensor_state(path):
    import torch
    metadata = {
        "omegaconf.dictconfig.DictConfig", "omegaconf.listconfig.ListConfig",
        "omegaconf.base.Metadata", "omegaconf.base.ContainerMetadata",
        "omegaconf.nodes.AnyNode", "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
    }
    containers = {"builtins.int", "builtins.dict", "builtins.list", "collections.defaultdict", "typing.Any"}
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if not unsafe <= metadata | containers:
        raise ValueError("Unsupported checkpoint metadata globals: " + repr(unsafe-metadata-containers))
    aliases = [(MetadataRecord, name) for name in sorted(metadata)]
    with torch.serialization.safe_globals(aliases+[int, dict, list, defaultdict, Any]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    result = {k.removeprefix("generator."): v for k, v in state.items() if k.startswith("generator.")}
    if not result or not all(isinstance(v, torch.Tensor) for v in result.values()):
        raise ValueError("Expected generator tensor weights only")
    return result
