"""Narrow stub for names ``mypy --strict`` cannot see on cvxpy.

``cvxpy/__init__.py`` pulls atom factories in with ``import *``. That form is
not a re-export when implicit re-exports are off, so ``cp.sum`` and the other
factories are missing. Classes that cvxpy already exports explicitly are
re-exported here unchanged. Untyped factories and ``Problem.solve`` are given
call signatures only; runtime objects are untouched.
"""

from typing import Any

from cvxpy import error as error
from cvxpy.atoms.affine.wraps import psd_wrap as psd_wrap
from cvxpy.atoms.elementwise.abs import abs as abs
from cvxpy.atoms.elementwise.log import log as log
from cvxpy.atoms.log_sum_exp import log_sum_exp as log_sum_exp
from cvxpy.atoms.norm1 import norm1 as norm1
from cvxpy.expressions.constants.constant import Constant as Constant
from cvxpy.expressions.variable import Variable as Variable
from cvxpy.problems.objective import Maximize as Maximize
from cvxpy.problems.objective import Minimize as Minimize
from cvxpy.problems.problem import Problem as _Problem
from cvxpy.atoms.sum_squares import sum_squares as sum_squares
from cvxpy.settings import (
    INFEASIBLE as INFEASIBLE,
)
from cvxpy.settings import (
    INFEASIBLE_INACCURATE as INFEASIBLE_INACCURATE,
)
from cvxpy.settings import (
    OPTIMAL as OPTIMAL,
)
from cvxpy.settings import (
    OPTIMAL_INACCURATE as OPTIMAL_INACCURATE,
)
from cvxpy.settings import (
    UNBOUNDED as UNBOUNDED,
)

def sum(
    expr: object,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool = ...,
) -> Any: ...
def pos(x: object) -> Any: ...
def power(x: object, p: object, max_denom: int = ..., approx: bool = ...) -> Any: ...
def quad_form(x: object, P: object, assume_PSD: bool = ...) -> Any: ...

class Problem(_Problem):
    def solve(self, *args: object, **kwargs: object) -> Any: ...
