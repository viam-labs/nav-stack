"""meta.json limits the Viam registry enforces on upload."""

import json
from pathlib import Path

META = json.loads((Path(__file__).resolve().parents[1] / "meta.json").read_text())


def test_model_descriptions_fit_the_registry_limit():
    # `viam module upload` rejects the whole upload when any is longer.
    too_long = {
        m["model"]: len(m.get("short_description", ""))
        for m in META["models"]
        if len(m.get("short_description", "")) > 100
    }
    assert too_long == {}
