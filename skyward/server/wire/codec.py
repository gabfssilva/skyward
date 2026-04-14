"""JSON codec with discriminated-union registry.

Generic recursive encoder/decoder converting arbitrary domain objects to and
from JSON-compatible Python structures.  Handles primitives, ``bytes``,
``datetime``/``timedelta``, ``Enum``, standard collections (tagged for
type-preserving round-trips), frozen dataclasses, ``pydantic.BaseModel``, and
discriminated unions registered via :func:`register`.
"""

from __future__ import annotations

import base64
import importlib
import typing
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, get_args, get_origin

from pydantic import BaseModel

__all__ = ["from_dict", "register", "register_encoder", "to_dict"]


type JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


_REGISTRY: dict[Any, dict[str, type]] = {}
_ALIAS_BY_CLASS: dict[type, tuple[Any, str]] = {}
_ENCODERS: dict[type, typing.Callable[[Any], Any]] = {}
_DECODERS: dict[type, typing.Callable[[Any], Any]] = {}


def register_encoder(
    cls: type,
    encoder: typing.Callable[[Any], Any],
    decoder: typing.Callable[[Any], Any],
) -> None:
    """Register custom encode/decode callbacks for *cls*.

    Used for types whose fields cannot round-trip through the generic
    dataclass path (e.g. :class:`skyward.api.plugin.Plugin`, whose callable
    hooks are not JSON-serializable — it is instead wire-encoded by
    factory name + kwargs).

    Parameters
    ----------
    cls
        Target class.  Instances of exactly this type (not subclasses)
        use the hook.
    encoder
        ``obj -> JSON-compatible`` callable.
    decoder
        ``dict -> obj`` callable used when decoding into *cls*.
    """
    _ENCODERS[cls] = encoder
    _DECODERS[cls] = decoder


def register(base: Any, **aliases: type) -> None:
    """Register a discriminated union.

    Parameters
    ----------
    base
        Base type, ``Protocol``, or ``TypeAliasType`` (via the ``type``
        statement) used as the union key.  The value is compared by
        identity, so any hashable marker works.
    **aliases
        Mapping from ``"type"`` tag string to concrete subclass.
    """
    bucket = _REGISTRY.setdefault(base, {})
    for tag, cls in aliases.items():
        bucket[tag] = cls
        _ALIAS_BY_CLASS[cls] = (base, tag)


def to_dict(obj: Any) -> Any:
    """Encode ``obj`` into a JSON-compatible structure.

    Parameters
    ----------
    obj
        Value to encode.  Supported shapes are listed in the module docstring.

    Returns
    -------
    Any
        A nested structure composed of primitives, ``list``, and ``dict``.
    """
    match obj:
        case None | bool() | int() | float() | str():
            return obj
        case bytes():
            return {"$b64": base64.b64encode(obj).decode("ascii")}
        case datetime():
            return {"$dt": obj.isoformat()}
        case timedelta():
            return {"$td": obj.total_seconds()}
        case Enum():
            cls = type(obj)
            return {"$enum": f"{cls.__module__}.{cls.__qualname__}", "name": obj.name}
        case MappingProxyType():
            return {k: to_dict(v) for k, v in obj.items()}
        case dict():
            return {k: to_dict(v) for k, v in obj.items()}
        case tuple():
            return {"$tuple": [to_dict(v) for v in obj]}
        case frozenset():
            return {"$frozenset": [to_dict(v) for v in obj]}
        case set():
            return {"$set": [to_dict(v) for v in obj]}
        case list():
            return [to_dict(v) for v in obj]
        case BaseModel():
            return obj.model_dump(mode="python")
        case _ if (enc := _ENCODERS.get(type(obj))) is not None:
            return enc(obj)
        case _ if is_dataclass(obj) and not isinstance(obj, type):
            payload = {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
            if (entry := _ALIAS_BY_CLASS.get(type(obj))) is not None:
                return {"type": entry[1], **payload}
            return payload
        case _:
            raise TypeError(f"Unsupported type for to_dict: {type(obj)!r}")


def from_dict(data: Any, target: Any = None) -> Any:
    """Decode ``data`` produced by :func:`to_dict`.

    Parameters
    ----------
    data
        Encoded value.
    target
        Optional concrete type or discriminated-union base used to guide
        decoding.  When omitted the codec infers structure from tags.

    Returns
    -------
    Any
        Decoded Python object.
    """
    match data:
        case None | bool() | int() | float() | str():
            return data
        case list():
            return [from_dict(v) for v in data]
        case dict() if "$b64" in data:
            return base64.b64decode(data["$b64"].encode("ascii"))
        case dict() if "$dt" in data:
            return datetime.fromisoformat(data["$dt"])
        case dict() if "$td" in data:
            return timedelta(seconds=data["$td"])
        case dict() if "$enum" in data:
            return _resolve_enum(data["$enum"], data["name"])
        case dict() if "$tuple" in data:
            return tuple(from_dict(v) for v in data["$tuple"])
        case dict() if "$set" in data:
            return {from_dict(v) for v in data["$set"]}
        case dict() if "$frozenset" in data:
            return frozenset(from_dict(v) for v in data["$frozenset"])
        case dict():
            return _decode_dict(data, target)
        case _:
            raise TypeError(f"Unsupported shape for from_dict: {type(data)!r}")


def _decode_dict(data: dict[str, Any], target: Any) -> Any:
    if target is not None and target in _REGISTRY:
        tag = data.get("type")
        if not isinstance(tag, str):
            raise ValueError(f"Missing 'type' discriminator for {target!r}")
        cls = _REGISTRY[target].get(tag)
        if cls is None:
            raise ValueError(f"Unknown tag {tag!r} for {target!r}")
        return _decode_target(data, cls)
    if target is not None:
        return _decode_target(data, target)
    if "type" in data:
        return {k: from_dict(v) for k, v in data.items()}
    return {k: from_dict(v) for k, v in data.items()}


def _decode_target(data: dict[str, Any], cls: type) -> Any:
    if (dec := _DECODERS.get(cls)) is not None:
        return dec(data)
    if issubclass(cls, BaseModel):
        return cls.model_validate(data)
    if is_dataclass(cls):
        try:
            hints = typing.get_type_hints(cls)
        except NameError:
            hints = {}
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            kwargs[f.name] = _decode_field(data[f.name], hints.get(f.name))
        return cls(**kwargs)
    raise TypeError(f"Cannot decode into {cls!r}")


def _decode_field(value: Any, hint: Any) -> Any:
    if hint is None:
        return _infer_decode(value)
    origin = get_origin(hint)
    if origin is None and isinstance(hint, type) and hint in _DECODERS:
        return _DECODERS[hint](value) if isinstance(value, dict) else value
    if origin is None and isinstance(hint, type) and (is_dataclass(hint) or issubclass(hint, BaseModel)):
        return from_dict(value, hint) if isinstance(value, dict) else value
    if hint in _REGISTRY:
        return from_dict(value, hint)
    args = get_args(hint)
    for arg in args:
        if isinstance(arg, type) and arg in _REGISTRY:
            return from_dict(value, arg)
    import types as _types
    if origin in (typing.Union, _types.UnionType):
        is_tagged_tuple = isinstance(value, dict) and "$tuple" in value
        preferred_origin = tuple if is_tagged_tuple else list if isinstance(value, list) else None
        if preferred_origin is not None:
            for arg in args:
                if get_origin(arg) is preferred_origin:
                    return _decode_field(value, arg)
        for arg in args:
            arg_origin = get_origin(arg)
            if arg_origin in (tuple, list):
                return _decode_field(value, arg)
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _decode_field(value, arg)
            except Exception:
                continue
    if origin in (tuple, list) and args:
        elem_hint = args[0]
        if isinstance(elem_hint, type) and (elem_hint in _DECODERS or is_dataclass(elem_hint) or elem_hint in _REGISTRY):
            if isinstance(value, dict) and isinstance(value.get("$tuple"), list):
                raw = value["$tuple"]
            elif isinstance(value, list):
                raw = value
            else:
                return from_dict(value)
            decoded = [_decode_field(v, elem_hint) for v in raw]
            return tuple(decoded) if origin is tuple else decoded
    return from_dict(value)


def _infer_decode(value: Any) -> Any:
    """Best-effort decode when no type hint is available.

    Inspects ``"type"`` tags and dispatches to any registered union or
    plugin decoder before falling back to generic :func:`from_dict`.
    """
    match value:
        case {"$tuple": list(items)}:
            return tuple(_infer_decode(v) for v in items)
        case {"type": str(tag), "args": dict()}:
            # Plugin wire shape — delegate to the Plugin decoder if any is
            # registered with ``register_encoder``.
            for dec_cls, dec in _DECODERS.items():
                if _REGISTRY.get(dec_cls) is None:
                    return dec(value)
            return from_dict(value)
        case {"type": str(tag)}:
            for base, bucket in _REGISTRY.items():
                if tag in bucket:
                    return from_dict(value, base)
            return from_dict(value)
        case list():
            return [_infer_decode(v) for v in value]
        case _:
            return from_dict(value)


def _resolve_enum(path: str, name: str) -> Enum:
    module_name, _, qualname = path.rpartition(".")
    module = importlib.import_module(module_name)
    cls: Any = module
    for part in qualname.split("."):
        cls = getattr(cls, part)
    return cls[name]
