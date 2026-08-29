"""Stable public exception identities shared across reloadable modules."""


class LLMResponseContractError(RuntimeError):
    """A provider response cannot safely drive a research action."""


__all__ = ["LLMResponseContractError"]
