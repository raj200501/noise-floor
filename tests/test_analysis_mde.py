import numpy as np
import pandas as pd
import pytest

from evalstats import design_from_logit, mde
from noisefloor.analysis import mde as m


def frame() -> pd.DataFrame:
    return pd.DataFrame({"day": ["d0", "d0", "d1", "d1"], "model": ["a", "b", "a", "b"]})


def test_reference_eta_and_pooled_design() -> None:
    coefs = {"(Intercept)": 0.5, "modelb": -1.0, "harness1": 0.3, "dayd1": 0.2}
    etas = m.reference_eta(coefs, frame(), ["a", "b"], {"harness": "1"}, ("day",))
    assert etas == {"a": pytest.approx(0.9), "b": pytest.approx(-0.1)}
    pooled = m.pooled_design(etas, 1.0, 0.2)
    single = design_from_logit(0.9, -0.1, 1.0, 0.2)
    assert pooled.between_task_var == pytest.approx(single.between_task_var)
    with pytest.raises(ValueError, match="two models"):
        m.pooled_design({"a": 0.0}, 1.0, 0.2)
    values = m.mde_values(pooled, ((40, 1), (89, 5)))
    assert values[(40, 1)] == pytest.approx(mde(pooled, 40, 1))
    assert values[(89, 5)] < values[(40, 1)]


def test_replicate_mdes() -> None:
    coefs = {"(Intercept)": [0.0, 0.1, 0.2], "modelb": [-0.5, -0.4, np.nan]}
    variances = {"task": [1.0, 1.1, 1.2], "model:task": [0.2, 0.3, 0.4]}
    out = m.replicate_mdes(
        coefs, variances, frame(), models=["a", "b"], reference={}, averaged=(), designs=((40, 1),)
    )
    values = out[(40, 1)]
    assert np.isfinite(values[:2]).all() and np.isnan(values[2])
    assert len(m.DESIGNS) == 15
