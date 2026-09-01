"""
Compact on-disk representation for embeddings.

A 384-dimensional float32 vector written as JSON decimals costs about 13 KB
per node, and the whole graph is rewritten on every save. Packing it as
base64 float32 costs about 2 KB instead, and skips formatting several hundred
floats per node on the way out.

Both readers accept the old list form, so existing graphs load unchanged.
"""

import base64

import numpy as np


PREFIX = "f32:"


def encode(embedding) -> str | None:
    """
    Pack a vector into a base64 float32 string.
    """

    if embedding is None:
        return None

    array = np.asarray(embedding, dtype=np.float32)
    return PREFIX + base64.b64encode(array.tobytes()).decode("ascii")


def decode(value):
    """
    Read either the packed form or the original list of floats.
    """

    if value is None:
        return None

    if isinstance(value, str) and value.startswith(PREFIX):
        raw = base64.b64decode(value[len(PREFIX):])
        return np.frombuffer(raw, dtype=np.float32)

    return np.array(value, dtype=np.float32)


def is_encoded(value) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)
