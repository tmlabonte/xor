from dataclasses import fields, is_dataclass
from typing import Any, Sequence

import numpy as np
import yaml

class FlowSequence:
    """Helper class for dumping long sequences in flow style."""

    def __init__(self, data: Sequence[Any]) -> None:
        self.data = data

def _confirm_dataclass(obj: Any) -> None:
    """Confirms obj is a dataclass instance."""

    if not is_dataclass(obj):
        raise TypeError(
            "Please set obj to be a dataclass instance."
            f" Current value: {type(obj)}."
        )

def asdict_numpy(obj: Any, max_inline: int = 10) -> Any:
    """Converts obj to dict with numpy arrays cast to lists.

    If the length of a sequence is over max_inline, it will be dumped in flow
    style. Otherwise, it will be dumped in block style.
    """

    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = asdict_numpy(value)
        return result
    elif isinstance(obj, np.ndarray):
        return asdict_numpy(obj.flatten().tolist())
    elif isinstance(obj, (list, tuple)):
        if len(obj) >= max_inline:
            return FlowSequence(type(obj)(asdict_numpy(v) for v in obj))
        return [asdict_numpy(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: asdict_numpy(v) for k, v in obj.items()}
    else:
        return obj

def dump_dataclass_to_yaml(obj: Any, path: str) -> None:
    """Exports dataclass instance to yaml file."""

    _confirm_dataclass(obj)
    if not path.endswith(".yaml"):
        path += ".yaml"

    # Add representer so long sequences are dumped inline.
    def represent_flowsequence(
        dumper: yaml.SafeDumper,
        data: FlowSequence,
    ) -> yaml.nodes.SequenceNode:
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data.data, flow_style=True)
    yaml.add_representer(
        FlowSequence, represent_flowsequence, Dumper=yaml.SafeDumper)

    with open(path, "w") as f:
        yaml.safe_dump(asdict_numpy(obj), f, default_flow_style=False)
