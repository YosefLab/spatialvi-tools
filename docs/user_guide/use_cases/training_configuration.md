# Training Configuration

## Basic training

```python
model.train(max_epochs=300)
```

## Adjusting learning rate and batch size

```python
model.train(max_epochs=300, lr=1e-3, batch_size=256)
```

## KL annealing

KL divergence weight is annealed from 0 to 1 over the first `n_epochs_kl_warmup` epochs
(default varies by model). Increase this for more stable early training:

```python
model.train(max_epochs=400, n_epochs_kl_warmup=100)
```

## GPU training

Training automatically uses a GPU when one is available via PyTorch Lightning's
`accelerator="auto"` default. To force CPU:

```python
model.train(max_epochs=300, accelerator="cpu")
```

## Monitoring training

The training progress bar reports ELBO loss. For programmatic monitoring, pass a Lightning
callback:

```python
from lightning.pytorch.callbacks import EarlyStopping

model.train(
    max_epochs=500,
    callbacks=[EarlyStopping(monitor="elbo_train", patience=20)],
)
```
