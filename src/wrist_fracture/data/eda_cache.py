"""Small, independently validated JSON checkpoints for iterative measurements."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def signature(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def measure_batches(keys, measure, *, directory, state, valid, batch_size=128, progress=None):
    """Yield one outcome per key, resuming only complete and valid batches."""
    if directory is not None:
        directory = Path(directory) / state["signature"]
        directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(keys), batch_size):
        expected = keys[start:start + batch_size]
        checkpoint = directory / f"{start:06d}.json" if directory is not None else None
        outcomes = None
        if checkpoint is not None:
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                records = payload["records"]
                if (payload["state"] == state and isinstance(records, list)
                        and all(isinstance(row, dict) for row in records)
                        and payload["sha256"] == signature(records)
                        and [row["image_key"] for row in records] == expected
                        and all(valid(row) for row in records)):
                    outcomes = records
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
        if outcomes is None:
            outcomes = [measure(key) for key in expected]
            if checkpoint is not None:
                payload = {"state": state, "records": outcomes, "sha256": signature(outcomes)}
                temporary = None
                try:
                    with NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                            dir=directory, suffix=".tmp") as handle:
                        temporary = Path(handle.name)
                        json.dump(payload, handle, allow_nan=False)
                    temporary.replace(checkpoint)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        if progress is not None:
            progress(start + len(expected), len(keys))
        yield from outcomes
