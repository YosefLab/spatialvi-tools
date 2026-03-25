import numpy as np
import pandas as pd
from anndata import AnnData

from spatialvi.model.base._deconvolution_mixin import SpatialDeconvolutionMixin


class _MockDeconvModel(SpatialDeconvolutionMixin):
    """Mock model that returns fake proportions (shape: n_spots x n_cell_types)."""

    cell_type_mapping = np.array(["CellA", "CellB", "CellC"])

    def get_proportions(self, adata=None):
        n = adata.n_obs if adata is not None else 20
        props = np.random.dirichlet(np.ones(3), size=n)
        return props


def test_get_proportions_df_shape():
    adata = AnnData(X=np.random.rand(20, 10))
    model = _MockDeconvModel()
    df = model.get_proportions_df(adata)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20, 3)
    assert list(df.columns) == ["CellA", "CellB", "CellC"]


def test_get_proportions_df_sums_to_one():
    adata = AnnData(X=np.random.rand(20, 10))
    model = _MockDeconvModel()
    df = model.get_proportions_df(adata)
    np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-5)
