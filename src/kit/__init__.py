"""kit: the visual toolkit the model works with. It runs inside the sandboxed session.

Everything listed in `__all__` below is in the session's namespace; the model's manual is generated from these
functions' signatures and docstrings (see ../manual.py), so a docstring here is what the model is told.
"""
from . import blend as _blend, check as _check, core as _core, draw as _draw, figures as _figures, fill as _fill, geom as _geom, layers as _layers, masks as _masks, measure as _measure, paint as _paint, sculpt as _sculpt, tone as _tone
from .blend import *  # noqa: F401,F403
from .check import *  # noqa: F401,F403
from .core import *  # noqa: F401,F403
from .draw import *  # noqa: F401,F403
from .figures import *  # noqa: F401,F403
from .fill import *  # noqa: F401,F403
from .geom import *  # noqa: F401,F403
from .layers import *  # noqa: F401,F403
from .measure import *  # noqa: F401,F403
from .paint import *  # noqa: F401,F403
from .sculpt import *  # noqa: F401,F403
from .masks import *  # noqa: F401,F403
from .tone import *  # noqa: F401,F403

#: The manual's sections, in reading order: (heading, module).
SECTIONS = [
    ("Files and looking", _core),
    ("Selecting: masks", _masks),
    ("Blending and compositing", _blend),
    ("Filling and cloning", _fill),
    ("Colour and tone", _tone),
    ("Moving and reshaping", _geom),
    ("Making new content", _draw),
    ("Sculpting: 3D shapes, rendered with light", _sculpt),
    ("Figures: preset animals and people", _figures),
    ("Strokes: painting, hatching, line", _paint),
    ("Layers and stages", _layers),
    ("Measuring and references", _measure),
    ("Checking", _check),
]

__all__ = [name for _, mod in SECTIONS for name in mod.__all__]


def _counted(fn):
    """The same function, noting each use in the session's record (which the delivery checks read)."""
    import functools

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        _core._count(fn.__name__)
        return fn(*args, **kwargs)
    return inner


for _name in __all__:
    _f = globals()[_name]
    if callable(_f) and not isinstance(_f, type):
        globals()[_name] = _counted(_f)
