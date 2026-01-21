"""Basic import tests for spatialvi_tools.

These tests verify that the top‑level API exports the expected symbols
and that the package can be imported without side effects.
"""

def test_top_level_imports() -> None:
    import spatialvi_tools as sv
    # check that version is defined
    assert isinstance(sv.__version__, str)
    # check that key classes are available
    assert hasattr(sv, "NolanModel")
    assert hasattr(sv, "LambdaModel")
    assert hasattr(sv, "PPIInference")
    assert hasattr(sv, "VIVSModel")
    assert hasattr(sv, "HarremanModel")
    assert hasattr(sv, "AmiciModel")
    assert hasattr(sv, "StarfyshModel")
    assert hasattr(sv, "SparlModel")