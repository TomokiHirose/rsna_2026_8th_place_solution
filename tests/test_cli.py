import importlib

import pytest

from .conftest import invoke

CLIS = ["tools.build_cache", "tools.kaggle_ops", "tools.fit_depth_prior", "tools.make_folds", "tools.select_weights", "tools.split_bilateral_knees", "tools.window_attention", "rsna_knee.training", "rsna_knee.predict"]


@pytest.mark.parametrize("module", CLIS)
def test_click_cli_prints_help(module: str) -> None:
    result = invoke(importlib.import_module(module).main, ["--help"])
    assert result.exit_code == 0 and "Usage:" in result.output
