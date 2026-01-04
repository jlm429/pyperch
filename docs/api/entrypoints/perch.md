# Perch Builder API

`Perch` is the user-facing entry point for configuring and running Pyperch. It provides a builder interface that constructs models, optimizers, data loaders, and training configuration.

---

## Typical Workflow

A complete experiment usually follows this pattern:

1. Define a model
2. Select optimizers (meta and/or gradient)
3. Configure metrics
4. Provide data
5. Train

```python
perch = (
    Perch()
    .model(SimpleMLP, input_dim=12, hidden=[32], output_dim=2)
    .optimizer("rhc", step_size=0.05)
    .metrics("accuracy")
    .data(X, y, batch_size=64, valid_split=0.2)
)

trainer, history = perch.train(max_epochs=250, seed=42)
```

---

## Responsibilities

### What Perch Handles

- Model construction
- Dataset splitting and loader creation
- Optimizer configuration (RHC, SA, GA, Adam, hybrids)
- Layer-level optimization modes (freeze / grad / meta)
- Metric instantiation
- Trainer creation and execution


## Public Builder Methods

### `model(...)`

Define the model class and its constructor arguments.

```python
.model(
    SimpleMLP,
    input_dim=12,
    hidden=[32],
    output_dim=2,
    activation="relu",
    loss_fn=nn.CrossEntropyLoss(),
)
```

Notes:

- The model class must be a `torch.nn.Module`
- `loss_fn` is optional; defaults are inferred when omitted

---

### `optimizer(name, **kwargs)`

Configure the meta-optimizer (RHC, SA, GA).

```python
.optimizer("sa", t=2.0, cooling=0.995, step_size=0.05)
```

Supported optimizers include:

- `"rhc"`
- `"sa"`
- `"ga"`

Only keyword arguments supported by `OptimizerConfig` are kept.

---

### `torch_optimizer(name="adam", **kwargs)`

Configure a PyTorch optimizer for gradient-based layers.

```python
.torch_optimizer("adam", lr=1e-3)
```

This is optional and only used when `.grad_opt(...)` is applied.

---

### `metrics(...)`

Specify metrics by name, class, or instance.

```python
.metrics("accuracy", F1())
```

Metrics are automatically instantiated for both training and validation.

---

### `data(X, y, **kwargs)`

Provide raw arrays for automatic loader construction.

```python
.data(
    X, y,
    batch_size=64,
    valid_split=0.2,
    stratify=True,
)
```

Common options include:

- `batch_size`
- `valid_split`
- `shuffle`
- `stratify`
- `normalize`

---

### `data_loaders(train_loader, valid_loader=None)`

Use pre-built PyTorch data loaders instead of raw arrays.

---

## Layer-Level Optimization Control

Layer behavior is controlled using parameter name patterns.

### Freeze Layers

```python
.freeze("net.0.weight", "net.0.bias")
```

Frozen layers:

- Are excluded from both gradient and meta optimization
- Remain unchanged throughout training

---

### Gradient-Trained Layers

```python
.grad_opt("net.2.weight", "net.2.bias")
```

These layers are optimized using the configured PyTorch optimizer.

---

### Meta-Optimized Layers

```python
.meta_opt("net.4.weight", "net.4.bias")
```

These layers are optimized using the configured meta-optimizer (RHC / SA / GA).

---

## Hybrid Optimization Example

```python
perch = (
    Perch()
    .model(SimpleMLP, input_dim=10, hidden=[32, 16], output_dim=2)
    .freeze("net.0.weight", "net.0.bias")
    .grad_opt("net.2.weight", "net.2.bias")
    .meta_opt("net.4.weight", "net.4.bias")
    .optimizer("rhc", step_size=0.5)
    .torch_optimizer("adam", lr=1e-3)
    .metrics("accuracy")
    .data(X, y, batch_size=32, valid_split=0.2)
)
```

---

## Training

```python
trainer, history = perch.train(
    max_epochs=200,
    seed=42,
    optimizer_mode="per_batch",
)
```

Returns:

- `Trainer` - fully configured trainer instance
- `history` - dictionary of losses and metrics over time

## Examples

- **[Perch Builder RHC Example Notebook](../../../examples/RHC_examples.ipynb)**
- **[Perch Builder SA Example Notebook](../../../examples/SA_examples.ipynb)**
- **[Perch Builder GA Example Notebook](../../../examples/GA_examples.ipynb)**
