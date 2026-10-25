# Smoothed Logit Function Softmax
SLFMax is a lightweight AI Pipeline designed to smooth out softmaxes. The implementation smooths out
the softmax using a logit function. This is specifically designed for usage in AI Models, and Language models.

## Implementation
```python
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

    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)

    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)

a = smooth_softmax([1, 2, 3, 4])
print(a)
```
## Usage
SLFMax recommends usage in Large-Language models, Small-Language models, games needing probabilities, and randomized algorithims
