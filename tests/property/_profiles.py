"""Example budgets for the adversarial property suite.

``HYPOTHESIS_PROFILE=ci`` (the default) is the pull-request budget.
``HYPOTHESIS_PROFILE=nightly`` draws more examples. These names are local to
this suite: Hypothesis already owns a built-in ``ci`` profile, and overwriting
it would change every other property test when ``CI`` is set.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

_HEALTH = (
    HealthCheck.too_slow,
    HealthCheck.filter_too_much,
    HealthCheck.data_too_large,
)


def register_adversarial_profiles() -> None:
    settings.register_profile(
        "adversarial-ci",
        max_examples=20,
        deadline=None,
        derandomize=True,
        database=None,
        print_blob=True,
        suppress_health_check=list(_HEALTH),
    )
    settings.register_profile(
        "adversarial-nightly",
        max_examples=80,
        deadline=None,
        derandomize=False,
        database=None,
        print_blob=True,
        suppress_health_check=list(_HEALTH),
    )


register_adversarial_profiles()


def adversarial_settings() -> settings:
    """Settings object usable as ``@adversarial_settings()``."""
    register_adversarial_profiles()
    selected = os.environ.get("HYPOTHESIS_PROFILE", "ci")
    name = "adversarial-nightly" if selected == "nightly" else "adversarial-ci"
    return settings(settings.get_profile(name))
