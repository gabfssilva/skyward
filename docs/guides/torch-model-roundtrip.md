# PyTorch Model Roundtrip

PyTorch's `nn.Module` is fully picklable — architecture and weights travel together through Python's serialization protocol. Skyward uses cloudpickle under the hood, which means you can send an untrained model to a remote worker, train it there, and get the trained model back. No `state_dict` files, no checkpoints, no manual save/load — the model object itself is the transport.

This guide walks through the full cycle: build locally, train remotely, evaluate locally.

## The Model

A standard `nn.Module` — nothing special required for serialization:

```python
--8<-- "examples/guides/15_torch_model_roundtrip.py:18:32"
```

When cloudpickle serializes this object, it captures both the class definition and the instance state (all parameter tensors). On the remote side, it reconstructs the exact same object with the exact same weights.

## Loading Data Locally

MNIST is loaded on the local machine and sent as tensors to the remote worker. The worker doesn't need `torchvision` — it only needs `torch` to work with the tensors it receives:

```python
--8<-- "examples/guides/15_torch_model_roundtrip.py:35:47"
```

The 60k training images (each 28x28) are flattened to 784-d vectors and normalized. Both `x_train` and `y_train` are regular tensors — cloudpickle handles them the same way it handles any Python object.

## The Remote Training Function

The `@sky.compute` function receives the model as an argument and returns it after training:

```python
--8<-- "examples/guides/15_torch_model_roundtrip.py:50:94"
```

The type signature tells the story: `nn.Module` goes in, `nn.Module` comes out. The optimizer modifies the model's parameters in-place during training. The final `model.cpu()` ensures all tensors are on CPU before serialization — this matters when training on GPU, since CUDA tensors can't deserialize on a machine without a GPU.

## Pinning the Torch Version

Torch tensors use pickle's `__reduce_ex__` protocol to serialize their raw storage. The binary format can change between torch versions — a tensor pickled with torch 2.10 may not deserialize correctly on torch 2.8 (or vice versa). Since the model travels both directions through cloudpickle, the local and remote torch versions must match:

```python
--8<-- "examples/guides/15_torch_model_roundtrip.py:15:15"
```

```python
--8<-- "examples/guides/15_torch_model_roundtrip.py:121:129"
```

`TORCH_VERSION` strips the build suffix (e.g. `+cpu`, `+cu128`) so the version pin works across wheel variants. The `PipIndex` scopes the PyTorch wheel index to the `torch` package only, preventing it from affecting other dependencies.

## The Full Cycle

Build the model locally, evaluate it (random accuracy), train remotely, evaluate again:

```python
--8<-- "examples/guides/15_torch_model_roundtrip.py:109:135"
```

The untrained model starts at ~10% accuracy (random chance for 10 classes). After `>> pool` dispatches the training to the remote worker and returns the trained model, local evaluation shows the learned accuracy — proving the weights survived the roundtrip.

## Local Evaluation

The `evaluate` function runs on your local machine using the model that came back from the cloud:

```python
--8<-- "examples/guides/15_torch_model_roundtrip.py:97:106"
```

No reconstruction, no `load_state_dict` — the returned object is a regular `MNISTClassifier` instance with trained weights, ready for inference.

## Run the Full Example

```bash
git clone https://github.com/gabfssilva/skyward.git
cd skyward
uv run python examples/guides/15_torch_model_roundtrip.py
```

---

**What you learned:**

- **`nn.Module` is picklable** — cloudpickle captures architecture + weights together, no manual serialization needed.
- **Models as arguments and return values** — send an untrained model in, get a trained model back via `>>`.
- **Pin torch versions** — the pickle format for tensors is version-sensitive; `TORCH_VERSION` keeps local and remote in sync.
- **`model.cpu()` before returning** — ensures CUDA tensors don't break deserialization on CPU-only machines.
- **Data stays local until dispatch** — load datasets on your machine, send as tensors; the worker only needs `torch`.
