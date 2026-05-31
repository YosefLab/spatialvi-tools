# SPARL Integration Design Spec

**Date:** 2026-05-31
**Status:** Approved
**Q&A log:** [2026-05-31-sparl-integration-qa.md](./2026-05-31-sparl-integration-qa.md)

---

## Summary

Integrate SPARL (Spatial Proteomics Analysis with Representation Learning — ViT/DINO-based)
into scviva-tools as the first model in a new `imaging/` submodule. v1 scope is
**inference-only**: load a pre-trained backbone, run forward pass on per-cell images,
write CLS-token embeddings to `adata.obsm["X_sparl"]`. No SSL pre-training, no fine-tuning.

---

## Scope (v1)

| In scope | Out of scope |
|----------|-------------|
| Load pre-trained ViT from local checkpoint or HuggingFace Hub | SSL pre-training (SSLMetaArch, DINO/iBOT losses) |
| Inference → CLS token embeddings → `obsm["X_sparl"]` | Fine-tuning (ClassificationMetaArch) |
| Two data entry points: disk paths (`setup_anndata`) + SpatialData (`from_spatialdata`) | Multi-GPU / distributed inference |
| High-level API only (`get_latent_representation`) | Backbone exposure (`model.backbone`) |
| `ImagingBaseModel` base class for future imaging models | Second imaging model |
| `sparl` as optional `[imaging]` dependency | HuggingFace model publishing |

---

## Architecture Decision

**Approach chosen:** New `scviva/imaging/` top-level submodule with `ImagingBaseModel`.

Rationale: more image-based models are planned beyond SPARL. `ImagingBaseModel` is the
shared foundation so future models require only a new `_module.py` + minimal `_model.py`.
`imaging/` is a peer of `model/` and `external/`, signalling a first-class model category.

---

## Directory Structure

```
src/scviva/
├── imaging/
│   ├── __init__.py              # lazy exports: ImagingBaseModel, SPARL
│   ├── base/
│   │   ├── __init__.py
│   │   └── _imaging_base.py     # ImagingBaseModel(SpatialBaseModel)
│   └── sparl/
│       ├── __init__.py
│       ├── _model.py            # SPARL(ImagingBaseModel)
│       └── _module.py           # SPARLModule(nn.Module) — wraps DinoVisionTransformer

tests/
└── imaging/
    ├── conftest.py              # shared fixtures
    ├── base/
    │   └── test_imaging_base.py
    └── sparl/
        └── test_sparl.py
```

---

## Component Design

### `ImagingBaseModel` (`imaging/base/_imaging_base.py`)

Inherits `SpatialBaseModel`. Provides everything every imaging model needs.
Gets `from_spatialdata()`, `setup_spatialdata()`, RAPIDS dispatch, and
`plot_spatial_embedding()` for free from `SpatialBaseModel`.

```python
class ImagingBaseModel(SpatialBaseModel):

    _default_obsm_key: str = "X_imaging"   # subclasses override

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **model_kwargs) -> "ImagingBaseModel":
        """Load pre-trained backbone from HuggingFace repo ID or local path."""
        # If model_name_or_path contains "/" and no os.sep → treat as HF repo ID
        # → hf_hub_download(repo_id, filename="model.pt")
        # Otherwise → local torch.load
        # Constructs cls._module_cls(backbone) and sets self.module

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        img_path_col: str,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> None:
        """Register obs column of image paths + spatial coordinates."""
        # Fields: ImagePathField(img_path_col) + SpatialCoordsField(spatial_key)

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        batch_size: int = 256,
        obsm_key: str | None = None,     # defaults to cls._default_obsm_key
        device: str = "cpu",
    ) -> np.ndarray:
        """Run inference; write embeddings to adata.obsm[obsm_key]. Return array."""
        # 1. _build_inference_dataloader(adata, batch_size)
        # 2. self.module.eval() + torch.no_grad()
        # 3. Accumulate CLS tokens
        # 4. adata.obsm[obsm_key] = result
        # 5. return result

    def _build_inference_dataloader(self, adata: AnnData, batch_size: int) -> DataLoader:
        """Minimal Dataset: load image path → PIL/tifffile → resize → normalize."""
        # No DataAugmentation, no MaskingGenerator, no EpochSampler

    def _load_backbone(self, path_or_id: str) -> nn.Module:
        """Resolve HuggingFace ID vs local path; return loaded nn.Module."""
```

### `SPARLModule` (`imaging/sparl/_module.py`)

Thin `nn.Module` wrapping SPARL's `DinoVisionTransformer`. Isolates the `sparl`
package import to a single file.

```python
class SPARLModule(nn.Module):
    def __init__(self, backbone: "DinoVisionTransformer"):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # returns: (B, embed_dim) — CLS token
        return self.backbone(x, is_training=False)["x_norm_clstoken"]
```

### `SPARL` (`imaging/sparl/_model.py`)

Inherits `ImagingBaseModel`. Adds only SPARL-specific concerns: channel metadata,
backbone config reading from checkpoint, and `"X_sparl"` obsm key.

```python
class SPARL(ImagingBaseModel):
    _module_cls = SPARLModule
    _default_obsm_key = "X_sparl"

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "SPARL":
        """Extends base: reads arch/patch_size/embed_dim from checkpoint metadata."""

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        img_path_col: str,
        channel_names: list[str] | None = None,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> None:
        """Extends base: also stores channel_names in adata.uns["sparl_channels"]."""

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        batch_size: int = 256,
        obsm_key: str = "X_sparl",
        device: str = "cpu",
    ) -> np.ndarray:
        return super().get_latent_representation(
            adata, batch_size=batch_size, obsm_key=obsm_key, device=device
        )
```

---

## Data Flow

### Path A — disk paths

```python
model = SPARL.from_pretrained("YosefLab/sparl-imc-vitb16")          # or local path
SPARL.setup_anndata(adata, img_path_col="crop_path",
                    channel_names=["CD3", "CD8", "DAPI"])
model.get_latent_representation(adata)
# → adata.obsm["X_sparl"]  shape: (n_cells, embed_dim)
```

### Path B — SpatialData

```python
model = SPARL.from_pretrained("/checkpoints/sparl_imc.pt")
SPARL.from_spatialdata(sdata, table_key="cells", region="imc_image")
model.get_latent_representation(adata)
# → adata.obsm["X_sparl"]  shape: (n_cells, embed_dim)
```

### Internal inference loop

```
adata.obs[img_path_col]  (or SpatialData crops)
  → _build_inference_dataloader()
      → load image (PIL / tifffile)
      → resize + channel normalize   ← SPARL transforms, no augmentation
      → batch tensor (B, C, H, W)
  → SPARLModule.forward(batch)
      → DinoVisionTransformer(x, is_training=False)["x_norm_clstoken"]
      → (B, embed_dim)
  → accumulate all batches → np.ndarray
  → adata.obsm["X_sparl"] = result
```

---

## Dependency Changes (`pyproject.toml`)

```toml
[project.optional-dependencies]
imaging = [
    "sparl>=0.1",           # SPARL package (install separately)
    "huggingface-hub>=0.20",
    "tifffile>=2023",
    "Pillow>=10",
]
```

`sparl` package not in `[project.dependencies]` — optional, like `rapids` and `spatial`.

---

## Tests

### Fixtures (`tests/imaging/conftest.py`)

```python
class _MinimalImagingModel(ImagingBaseModel):
    """Concrete subclass using random Linear backbone — no sparl dep."""
    _default_obsm_key = "X_test_imaging"

def make_imaging_adata(n=20, n_channels=4, img_size=32, tmp_path) -> AnnData:
    # AnnData with spatial coords + obs["img_path"] → tmp PNG files

@pytest.fixture(scope="module")
def imaging_adata(tmp_path_factory): ...

@pytest.fixture(scope="module")
def sparl_model(tmp_path):
    # Save tiny DinoVisionTransformer checkpoint locally
    # Return SPARL.from_pretrained(local_path)
```

### Test matrix

| Test | File | Requires sparl? |
|------|------|----------------|
| `test_from_pretrained_local` | `test_sparl.py` | yes |
| `test_setup_anndata_registers_fields` | `test_sparl.py` | yes |
| `test_get_latent_representation_shape` | `test_sparl.py` | yes |
| `test_get_latent_representation_no_nan` | `test_sparl.py` | yes |
| `test_obsm_key_override` | `test_sparl.py` | yes |
| `test_from_spatialdata` | `test_sparl.py` | yes |
| `test_imaging_base_no_sparl_dep` | `test_imaging_base.py` | no |
| `test_imaging_base_get_latent_shape` | `test_imaging_base.py` | no |

All `test_sparl.py` tests gated with `pytest.importorskip("sparl")`.
HuggingFace download mocked via `monkeypatch` on `hf_hub_download`.

> **TODO:** add `test_from_pretrained_hub` once a `YosefLab/sparl-*` model is published on HuggingFace.

---

## `scviva/__init__.py` changes

Add lazy import for `imaging` submodule, following existing `__getattr__` pattern:

```python
_SPARL_CLASSES = ["SPARL", "ImagingBaseModel"]
# added to the lazy import registry
```

---

## Future imaging models

To add a second imaging model, create:
```
src/scviva/imaging/<new_model>/
    __init__.py
    _model.py    # NewModel(ImagingBaseModel) — _default_obsm_key, _module_cls
    _module.py   # NewModelModule(nn.Module) — wraps external backbone
```

`ImagingBaseModel` requires no changes.

---

## Implementation notes

- `ImagingBaseModel.from_spatialdata()` will need a light override: the inherited
  `SpatialBaseModel.from_spatialdata()` calls `setup_anndata()`, which expects
  `img_path_col`. For the SpatialData path, image crops are extracted from the
  `sdata` image table rather than from disk. The override should skip `img_path_col`
  registration and instead register a `SpatialDataImageField` (new, to be defined).
  This is the only place where the two entry points diverge internally.

---

## Open items (post-v1)

- Fine-tuning support (linear probe + full fine-tune via scvi-tools TrainingPlan)
- Backbone exposure (`model.module` or `model.backbone`) for power users
- HuggingFace model card + `YosefLab/sparl-*` publication
- Multi-GPU inference
