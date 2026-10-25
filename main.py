import numpy as np

def smooth_softmax(logits, temperature=1.0, axis=-1):
    logits = np.asarray(logits, dtype=np.float64)

    if logits.size == 0:
        return np.array(logits, dtype=np.float64)

    if not np.all(np.isfinite(logits)):
        raise ValueError("Logits must contain only finite values")

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")

    scaled = logits / temperature

    # Numerical stability: prevent exponential overflow
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)

    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
