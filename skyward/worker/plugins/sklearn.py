"""Scikit-learn, with its joblib parallelism fanned out over the compute.

Scikit-learn parallelises through joblib, so the whole of it — every estimator run
with ``n_jobs=-1``, every cross-validation — is redirected by pointing that one
backend at the pool, exactly as the joblib plugin does. This plugin adds the two
things the bare joblib plugin does not: scikit-learn itself on the nodes, and a
scrub of the client's warning filters before the run.

Scikit-learn's own ``Parallel`` captures ``warnings.filters`` and ships them inside
every batch. A filter whose category class comes from a third-party package names a
module the node may not have, and the batch then fails to unpickle there — a
``ModuleNotFoundError`` that reads like a missing dependency and is a warning filter.
Dropping every non-stdlib filter before the run leaves only the categories every
process is guaranteed to carry; the packages the nodes do have re-register their own
at import time.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

from msgspec.structs import replace

from skyward.shared.schemas import Image
from skyward.worker.plugins.plugin import Plugin

if TYPE_CHECKING:
    from skyward.core.compute import Compute


class Sklearn(Plugin, frozen=True):
    """Install scikit-learn on the nodes, and route its joblib backend through the pool.

    Attributes
    ----------
    version : str | None
        Pin, if the code needs one. Otherwise whatever the index has.
    """

    kind: ClassVar[str] = "sklearn"

    version: str | None = None

    def image(self, image: Image) -> Image:
        package = f"scikit-learn=={self.version}" if self.version else "scikit-learn"
        return replace(image, pip=(*image.pip, package, "joblib"))

    @contextmanager
    def client(self, compute: Compute) -> Iterator[None]:
        from skyward.worker.plugins.joblib import Joblib

        _strip_local_warning_filters()
        with Joblib().client(compute):
            yield


_SAFE_ROOTS = sys.stdlib_module_names | {"builtins"}


def _strip_local_warning_filters() -> None:
    """Keep only the warning filters every process is guaranteed to carry.

    A filter's category is a class, and a class comes from a module. Filters whose
    module is stdlib or ``builtins`` exist everywhere; the rest travel to a node that
    may not have them and break the unpickling of the batch they were folded into.
    """
    warnings.filters = [f for f in warnings.filters if f[2].__module__.split(".")[0] in _SAFE_ROOTS]
