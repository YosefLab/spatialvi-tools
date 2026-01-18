"""Tests for external model wrappers."""


class TestDestVIWrapper:
    """Tests for DestVI wrapper."""

    def test_import(self):
        """Test that DestVI wrapper can be imported."""
        from spatialvi.external.destvi import DestVI

        assert DestVI is not None


class TestScVIVAWrapper:
    """Tests for scVIVA wrapper."""

    def test_import(self):
        """Test that scVIVA wrapper can be imported."""
        from spatialvi.external.scviva import scVIVA

        assert scVIVA is not None


class TestResolVIWrapper:
    """Tests for ResolVI wrapper."""

    def test_import(self):
        """Test that ResolVI wrapper can be imported."""
        from spatialvi.external.resolvi import ResolVI

        assert ResolVI is not None


class TestStarfyshWrapper:
    """Tests for Starfysh wrapper."""

    def test_import(self):
        """Test that Starfysh wrapper can be imported."""
        from spatialvi.external.starfysh import Starfysh

        assert Starfysh is not None


class TestHarremanWrapper:
    """Tests for Harreman wrapper."""

    def test_import(self):
        """Test that Harreman wrapper can be imported."""
        from spatialvi.external.harreman import Harreman

        assert Harreman is not None


class TestVIVSWrapper:
    """Tests for VIVS wrapper."""

    def test_import(self):
        """Test that VIVS wrapper can be imported."""
        from spatialvi.external.vivs import VIVS

        assert VIVS is not None


class TestAMICIModel:
    """Tests for AMICI model wrapper."""

    def test_import(self):
        """Test that AMICI model can be imported."""
        from spatialvi.external.amici import AMICI

        assert AMICI is not None


class TestNolanWrapper:
    """Tests for Nolan (NicheExplorer) wrapper."""

    def test_import(self):
        """Test that Nolan model can be imported."""
        from spatialvi.external.nolan import Nolan

        assert Nolan is not None

    def test_import_from_external(self):
        """Test that Nolan can be imported from external module."""
        from spatialvi.external import Nolan

        assert Nolan is not None


class TestLambdaWrapper:
    """Tests for Lambda (LLM annotation) wrapper."""

    def test_import(self):
        """Test that Lambda model can be imported."""
        from spatialvi.external.lambda_model import Lambda

        assert Lambda is not None

    def test_import_from_external(self):
        """Test that Lambda can be imported from external module."""
        from spatialvi.external import Lambda

        assert Lambda is not None


class TestPPIInferenceWrapper:
    """Tests for PPIInference wrapper."""

    def test_import(self):
        """Test that PPIInference can be imported."""
        from spatialvi.external.ppi import PPIInference

        assert PPIInference is not None

    def test_import_from_external(self):
        """Test that PPIInference can be imported from external module."""
        from spatialvi.external import PPIInference

        assert PPIInference is not None

    def test_classical_mean_ci(self):
        """Test classical mean CI computation (no external dependency)."""
        import numpy as np

        from spatialvi.external import PPIInference

        y = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        ci_low, ci_high = PPIInference.classical_mean_ci(y, alpha=0.1)

        assert ci_low < 1.0 < ci_high
        assert ci_low < ci_high


class TestSPARLWrapper:
    """Tests for SPARL wrapper."""

    def test_import(self):
        """Test that SPARL model can be imported."""
        from spatialvi.external.sparl import SPARL

        assert SPARL is not None

    def test_import_from_external(self):
        """Test that SPARL can be imported from external module."""
        from spatialvi.external import SPARL

        assert SPARL is not None
