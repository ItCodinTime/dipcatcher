"""Lazy ``__getattr__`` re-export packages: every listed name resolves,
unknown names raise ``AttributeError``, and ``__dir__`` stays complete."""

from __future__ import annotations

import importlib
import types

import pytest

pytestmark = pytest.mark.synthetic

_LAZY_PACKAGES = (
    "quant_fund.diffbacktest",
    "quant_fund.hmm",
    "quant_fund.lightspeed",
    "quant_fund.models",
    "quant_fund.pipeline",
    "quant_fund.quant_models",
    "quant_fund.reality",
    "quant_fund.registry",
    "quant_fund.stress",
)


@pytest.mark.parametrize("package", _LAZY_PACKAGES)
def test_every_exported_name_resolves(package: str) -> None:
    mod = importlib.import_module(package)
    assert mod.__all__
    for name in mod.__all__:
        value = getattr(mod, name)
        assert value is not None, f"{package}.{name} failed to resolve"
        # Resolved names are cached on the module globals for repeat access.
        assert getattr(mod, name) is value


@pytest.mark.parametrize("package", _LAZY_PACKAGES)
def test_unknown_name_raises_attribute_error(package: str) -> None:
    mod = importlib.import_module(package)
    with pytest.raises(AttributeError, match="no attribute"):
        mod.__definitely_not_an_export__


_DIR_PACKAGES = tuple(
    pkg for pkg in _LAZY_PACKAGES if "__dir__" in vars(importlib.import_module(pkg))
)


@pytest.mark.parametrize("package", _DIR_PACKAGES)
def test_dir_lists_all_exports(package: str) -> None:
    mod = importlib.import_module(package)
    listed = set(dir(mod))
    assert set(mod.__all__) <= listed


def test_quant_fund_root_public_names_resolve() -> None:
    import quant_fund

    for name in quant_fund.__all__:
        assert getattr(quant_fund, name) is not None


def test_quant_fund_root_unknown_and_private_names() -> None:
    import quant_fund

    with pytest.raises(AttributeError):
        quant_fund.no_such_public_name
    # __firm__ / __version__ are package attrs already bound in globals; the
    # lazy hook refuses to re-resolve them if invoked directly.
    with pytest.raises(AttributeError):
        quant_fund.__getattr__("__firm__")
    assert quant_fund.__version__


def test_lazy_modules_are_module_type() -> None:
    for package in _LAZY_PACKAGES:
        mod = importlib.import_module(package)
        assert isinstance(mod, types.ModuleType)
