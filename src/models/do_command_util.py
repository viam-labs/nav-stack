"""Shared DoCommand helpers for nav-stack Viam models.

Both ``SlamService`` and ``NavServiceBase`` used long ``if cmd == ...`` chains.
Handlers live in a dict; ``require`` is the one place that pulls typed fields
out of the command payload.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Optional, Union

_MISSING = object()

# type alias: sync or async handler
DoCommandHandler = Callable[..., Any]


def require(
    params: Mapping[str, Any],
    key: str,
    expected: Optional[type] = None,
    *,
    default: Any = _MISSING,
) -> Any:
    """Read ``params[key]``, optionally coerce/check ``expected``.

    ``expected`` may be ``str``, ``float``, ``int``, ``bool``, or a type used
    with ``isinstance``. Missing keys use ``default`` when provided; otherwise
    raise ``ValueError``.
    """
    if key not in params or params[key] is None:
        if default is not _MISSING:
            return default
        raise ValueError(f"missing required field: {key}")
    val = params[key]
    if expected is None:
        return val
    if expected is str:
        return str(val)
    if expected is float:
        try:
            return float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc
    if expected is int:
        try:
            return int(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an int") from exc
    if expected is bool:
        return bool(val)
    if not isinstance(val, expected):
        name = getattr(expected, "__name__", str(expected))
        raise ValueError(f"{key} must be {name}")
    return val


def resolve_command_name(
    command: Mapping[str, Any],
    *,
    aliases: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Normalize ``command["command"]`` through an optional alias table."""
    raw = command.get("command")
    if raw is None:
        return None
    name = str(raw)
    if aliases:
        return aliases.get(name, name)
    return name


async def dispatch_do_command(
    handlers: Mapping[str, DoCommandHandler],
    command: Mapping[str, Any],
    *,
    aliases: Optional[Mapping[str, str]] = None,
    context: Any = None,
) -> Any:
    """Look up and run a DoCommand handler.

    Handlers may be sync or async. When ``context`` is set, it is passed as
    the first positional argument (typically ``self`` bound methods already
    close over self — pass ``None`` then).
    """
    name = resolve_command_name(command, aliases=aliases)
    if not name:
        raise ValueError("missing required field: command")
    handler = handlers.get(name)
    if handler is None:
        raise ValueError(f"unknown command: {name!r}")
    if context is not None:
        result = handler(context, command)
    else:
        result = handler(command)
    if inspect.isawaitable(result):
        return await result
    return result


def build_alias_map(
    groups: Mapping[str, Union[str, tuple, list]],
) -> dict[str, str]:
    """Map every alias onto its canonical handler name.

    ``groups`` maps canonical name → alias or sequence of aliases (canonical
    itself is always included).
    """
    out: dict[str, str] = {}
    for canonical, aliases in groups.items():
        out[canonical] = canonical
        if isinstance(aliases, str):
            out[aliases] = canonical
            continue
        for alias in aliases:
            out[str(alias)] = canonical
    return out
