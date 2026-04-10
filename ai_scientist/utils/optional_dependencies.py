from __future__ import annotations

import importlib
from typing import Any


class MissingOptionalDependencyProxy:
    """Import-safe proxy that delays optional dependency failures until use."""

    def __init__(
        self,
        module_name: str,
        *,
        install_hint: str,
        exception_names: tuple[str, ...] = (),
    ) -> None:
        self._module_name = module_name
        self._install_hint = install_hint
        for exception_name in exception_names:
            setattr(self, exception_name, type(exception_name, (Exception,), {}))

    def __getattr__(self, name: str) -> Any:
        raise ModuleNotFoundError(
            f"Optional dependency '{self._module_name}' is required to access "
            f"'{name}'. {self._install_hint}"
        )


class BackoffFallback:
    @staticmethod
    def expo(*_args, **_kwargs):
        return None

    @staticmethod
    def on_exception(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    @staticmethod
    def on_predicate(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


def import_optional_module(
    module_name: str,
    *,
    install_hint: str,
    exception_names: tuple[str, ...] = (),
) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = str(exc.name or "").strip()
        if missing_name not in {
            module_name,
            *[
                ".".join(module_name.split(".")[:idx])
                for idx in range(1, len(module_name.split(".")))
            ],
        }:
            raise
        return MissingOptionalDependencyProxy(
            module_name,
            install_hint=install_hint,
            exception_names=exception_names,
        )


def import_backoff() -> Any:
    try:
        return importlib.import_module("backoff")
    except ModuleNotFoundError as exc:
        if exc.name != "backoff":
            raise
        return BackoffFallback()


def resolve_exception_types(
    module: Any,
    exception_names: tuple[str, ...],
) -> tuple[type[BaseException], ...]:
    resolved: list[type[BaseException]] = []
    for exception_name in exception_names:
        try:
            candidate = getattr(module, exception_name)
        except Exception:
            candidate = None
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            resolved.append(candidate)
            continue
        resolved.append(type(exception_name, (Exception,), {}))
    return tuple(resolved)
