"""Deterministic, provenance-aware evaluation of structured experiment data.

Generated experiments commonly save ``experiment_data.npy`` as a pickled
NumPy object.  Loading that file with ``allow_pickle=True`` would let an
untrusted artifact execute arbitrary Python while it is being evaluated.  This
module therefore uses a restricted unpickler, validates the resulting object
graph, and only verifies metrics whose semantics are unambiguous.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import pickletools
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    from numpy._core import multiarray as _numpy_multiarray
except ImportError:  # pragma: no cover - compatibility with older NumPy
    from numpy.core import multiarray as _numpy_multiarray


EVALUATOR_SCHEMA_VERSION = "deterministic_evaluation.v1"
EVALUATOR_VERSION = "1.0.0"
DEFAULT_MAX_FILE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_ARRAY_ELEMENTS = 5_000_000
DEFAULT_MAX_CONTAINER_ITEMS = 100_000
DEFAULT_MAX_DEPTH = 16

_TRUE_KEYS = ("ground_truth", "y_true", "targets", "target", "labels", "label")
_PRED_KEYS = ("predictions", "y_pred", "preds", "predicted")
_SCORE_KEYS = (
    "probabilities",
    "probability",
    "probs",
    "scores",
    "score",
    "y_score",
    "y_prob",
)

_METRIC_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("roc_auc", ("roc auc", "roc-auc", "roc_auc", "auc")),
    ("accuracy", ("accuracy", "accurate", "acc")),
    ("precision", ("precision",)),
    ("recall", ("recall", "sensitivity")),
    ("f1", ("f1 score", "f1-score", "f1_score", "f1")),
    ("rmse", ("root mean squared error", "root mean square error", "rmse")),
    ("mse", ("mean squared error", "mean square error", "mse")),
    ("mae", ("mean absolute error", "mae")),
    ("r2", ("r squared", "r-squared", "r_squared", "r2", "r²")),
)


class EvaluationInputError(ValueError):
    """The artifact cannot be safely or unambiguously evaluated."""


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Allow only the globals required to reconstruct plain NumPy arrays."""

    _ALLOWED_GLOBALS = {
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("numpy._core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy.core.multiarray", "_reconstruct"): _numpy_multiarray._reconstruct,
        ("numpy._core.multiarray", "scalar"): _numpy_multiarray.scalar,
        ("numpy.core.multiarray", "scalar"): _numpy_multiarray.scalar,
    }

    def find_class(self, module: str, name: str) -> Any:
        allowed = self._ALLOWED_GLOBALS.get((module, name))
        if allowed is None:
            raise EvaluationInputError(
                f"unsafe pickle global rejected: {module}.{name}"
            )
        return allowed

    def persistent_load(self, pid: Any) -> Any:
        raise EvaluationInputError("pickle persistent IDs are not supported")


def _screen_pickle_stream(
    data: bytes, *, max_container_items: int, max_array_elements: int
) -> None:
    """Reject suspicious pickle structures before NumPy can allocate objects."""

    allowed_globals = set(_RestrictedNumpyUnpickler._ALLOWED_GLOBALS)
    stack: list[Any] = []
    memo: dict[int, Any] = {}
    next_memo_index = 0
    marks: list[int] = []
    constructed_items = 0
    last_global_module: str | None = None

    def push(value: Any = None) -> None:
        stack.append(value)

    def pop_mark() -> list[Any]:
        if not marks:
            raise EvaluationInputError("malformed pickle mark stack")
        index = marks.pop()
        values = stack[index:]
        del stack[index:]
        return values

    try:
        for opcode, argument, _ in pickletools.genops(data):
            name = opcode.name
            if name == "MARK":
                marks.append(len(stack))
            elif name in {
                "NONE",
                "NEWTRUE",
                "NEWFALSE",
                "BININT",
                "BININT1",
                "BININT2",
                "LONG",
                "LONG1",
                "LONG4",
                "FLOAT",
                "BINFLOAT",
                "INT",
                "STRING",
                "BINSTRING",
                "SHORT_BINSTRING",
                "BINUNICODE",
                "SHORT_BINUNICODE",
                "BINUNICODE8",
                "BYTEARRAY8",
                "BINBYTES",
                "SHORT_BINBYTES",
                "BINBYTES8",
            }:
                push(argument)
                if isinstance(argument, str):
                    last_global_module = argument
            elif name == "EMPTY_TUPLE":
                push(())
            elif name == "EMPTY_LIST":
                push([])
            elif name == "EMPTY_DICT":
                push({})
            elif name == "EMPTY_SET":
                push(set())
            elif name == "TUPLE1":
                push((stack.pop(),))
            elif name == "TUPLE2":
                second, first = stack.pop(), stack.pop()
                push((first, second))
            elif name == "TUPLE3":
                third, second, first = stack.pop(), stack.pop(), stack.pop()
                push((first, second, third))
            elif name in {"TUPLE", "LIST", "DICT", "FROZENSET"}:
                items = pop_mark()
                constructed_items += len(items)
                push(tuple(items))
            elif name == "APPEND":
                constructed_items += 1
                if stack:
                    stack.pop()
            elif name == "SETITEM":
                constructed_items += 2
                if stack:
                    stack.pop()
                if stack:
                    stack.pop()
            elif name in {"APPENDS", "SETITEMS", "ADDITEMS"}:
                items = pop_mark()
                constructed_items += len(items)
            elif name in {"PUT", "BINPUT", "LONG_BINPUT"}:
                memo[int(argument)] = stack[-1] if stack else None
            elif name == "MEMOIZE":
                memo[next_memo_index] = stack[-1] if stack else None
                next_memo_index += 1
            elif name in {"GET", "BINGET", "LONG_BINGET"}:
                push(memo.get(int(argument)))
            elif name == "STACK_GLOBAL":
                global_name = stack.pop() if stack else None
                module = stack.pop() if stack else last_global_module
                if (str(module), str(global_name)) not in allowed_globals:
                    raise EvaluationInputError(
                        f"unsafe pickle global rejected: {module}.{global_name}"
                    )
                push((str(module), str(global_name)))
            elif name == "GLOBAL":
                module, global_name = str(argument).split(" ", 1)
                if (module, global_name) not in allowed_globals:
                    raise EvaluationInputError(
                        f"unsafe pickle global rejected: {module}.{global_name}"
                    )
                push((module, global_name))
            elif name == "REDUCE":
                args = stack.pop() if stack else None
                function = stack.pop() if stack else None
                if not isinstance(function, tuple) or function not in allowed_globals:
                    raise EvaluationInputError(
                        "pickle REDUCE target is not an approved NumPy constructor"
                    )
                if function[1] == "ndarray":
                    raise EvaluationInputError(
                        "direct ndarray construction is not supported in evaluation artifacts"
                    )
                if function[1] == "_reconstruct":
                    if not isinstance(args, tuple) or len(args) < 2:
                        raise EvaluationInputError(
                            "malformed NumPy array reconstruction arguments"
                        )
                    shape = args[1]
                    if not isinstance(shape, tuple) or not all(
                        isinstance(dim, (int, np.integer)) and int(dim) >= 0
                        for dim in shape
                    ):
                        raise EvaluationInputError(
                            "pickle declares an invalid NumPy array shape"
                        )
                    elements = math.prod(int(dim) for dim in shape)
                    if elements > max_array_elements:
                        raise EvaluationInputError(
                            "pickle declares an array larger than the evaluator limit"
                        )
                push(("reduced", function, args))
            elif name in {
                "BUILD",
                "NEWOBJ",
                "NEWOBJ_EX",
                "OBJ",
                "INST",
                "EXT1",
                "EXT2",
                "EXT4",
                "PERSID",
                "BINPERSID",
            }:
                if name != "BUILD":
                    raise EvaluationInputError(f"unsupported pickle opcode: {name}")
                state = stack.pop() if stack else None
                shape = _array_shape_from_build_state(state)
                if shape is not None:
                    elements = math.prod(shape)
                    if elements > max_array_elements:
                        raise EvaluationInputError(
                            "pickle declares an array larger than the evaluator limit"
                        )
            elif name in {"PROTO", "FRAME", "STOP", "POP", "POP_MARK", "DUP"}:
                if name == "POP" and stack:
                    stack.pop()
                elif name == "POP_MARK":
                    pop_mark()
                elif name == "DUP" and stack:
                    push(stack[-1])
            else:
                raise EvaluationInputError(f"unsupported pickle opcode: {name}")

            if constructed_items > max_container_items:
                raise EvaluationInputError(
                    f"pickle containers exceed {max_container_items} total items"
                )
    except EvaluationInputError:
        raise
    except Exception as exc:
        raise EvaluationInputError(f"malformed pickle stream: {exc}") from exc


def _array_shape_from_build_state(state: Any) -> tuple[int, ...] | None:
    if not isinstance(state, tuple) or len(state) < 2:
        return None
    shape = state[1]
    if not isinstance(shape, tuple) or not all(
        isinstance(dim, (int, np.integer)) and int(dim) >= 0 for dim in shape
    ):
        return None
    return tuple(int(dim) for dim in shape)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _implementation_hash() -> str:
    try:
        return _sha256_bytes(Path(__file__).read_bytes())
    except OSError:
        return _canonical_hash(
            {"schema_version": EVALUATOR_SCHEMA_VERSION, "version": EVALUATOR_VERSION}
        )


def _read_npy_safely(
    path: Path,
    *,
    max_file_bytes: int,
    max_container_items: int,
    max_array_elements: int,
) -> tuple[Any, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvaluationInputError(f"artifact cannot be opened safely: {exc}") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise EvaluationInputError("evaluation artifact is not a regular file")
        if file_stat.st_size > max_file_bytes:
            raise EvaluationInputError(
                f"artifact is {file_stat.st_size} bytes; limit is {max_file_bytes} bytes"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(max_file_bytes + 1)
        if len(raw) > max_file_bytes:
            raise EvaluationInputError(
                f"artifact exceeds the {max_file_bytes}-byte limit while reading"
            )
    finally:
        os.close(descriptor)

    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=10_000
            )
        elif version in {(2, 0), (3, 0)}:
            shape, _, dtype = np.lib.format.read_array_header_2_0(
                stream, max_header_size=10_000
            )
        else:
            raise EvaluationInputError(f"unsupported NPY version: {version!r}")
    except EvaluationInputError:
        raise
    except Exception as exc:
        raise EvaluationInputError(f"invalid NPY header: {exc}") from exc

    if dtype.hasobject:
        _screen_pickle_stream(
            stream.getbuffer()[stream.tell() :].tobytes(),
            max_container_items=max_container_items,
            max_array_elements=max_array_elements,
        )
        try:
            loaded = _RestrictedNumpyUnpickler(stream).load()
        except EvaluationInputError:
            raise
        except Exception as exc:
            raise EvaluationInputError(
                f"restricted NPY decoding failed: {exc}"
            ) from exc
    else:
        try:
            loaded = np.load(
                io.BytesIO(raw), allow_pickle=False, max_header_size=10_000
            )
        except Exception as exc:
            raise EvaluationInputError(f"NPY decoding failed: {exc}") from exc

    expected_shape = tuple(int(dim) for dim in shape)
    if not isinstance(loaded, np.ndarray) or loaded.shape != expected_shape:
        raise EvaluationInputError(
            "decoded array does not match the declared NPY shape"
        )
    if loaded.dtype != dtype:
        raise EvaluationInputError(
            "decoded array does not match the declared NPY dtype"
        )

    if loaded.dtype.hasobject and loaded.size == 1:
        loaded = loaded.item()
    return loaded, raw


def _validate_object_graph(
    value: Any,
    *,
    max_depth: int,
    max_array_elements: int,
    max_container_items: int,
) -> None:
    seen: set[int] = set()
    array_elements = 0
    container_items = 0

    def visit(current: Any, depth: int) -> None:
        nonlocal array_elements, container_items
        if depth > max_depth:
            raise EvaluationInputError(f"artifact nesting exceeds depth {max_depth}")

        if current is None or isinstance(current, (str, bytes, bool, int, float)):
            return
        if isinstance(current, np.generic):
            if current.dtype.hasobject:
                visit(current.item(), depth + 1)
            return

        identity = id(current)
        if identity in seen:
            return
        seen.add(identity)

        if isinstance(current, np.ndarray):
            array_elements += int(current.size)
            if array_elements > max_array_elements:
                raise EvaluationInputError(
                    f"artifact arrays exceed {max_array_elements} total elements"
                )
            if current.dtype.hasobject:
                for item in current.flat:
                    visit(item, depth + 1)
            return

        if isinstance(current, Mapping):
            container_items += len(current)
            if container_items > max_container_items:
                raise EvaluationInputError(
                    f"artifact containers exceed {max_container_items} total items"
                )
            for key, item in current.items():
                if not isinstance(key, (str, int, float, bool)):
                    raise EvaluationInputError(
                        f"unsupported mapping key type: {type(key).__name__}"
                    )
                visit(item, depth + 1)
            return

        if isinstance(current, (list, tuple, set, frozenset)):
            container_items += len(current)
            if container_items > max_container_items:
                raise EvaluationInputError(
                    f"artifact containers exceed {max_container_items} total items"
                )
            for item in current:
                visit(item, depth + 1)
            return

        raise EvaluationInputError(
            f"unsupported decoded type: {type(current).__name__}"
        )

    visit(value, 0)


def _normalise_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _matching_items(
    mapping: Mapping[Any, Any], aliases: tuple[str, ...]
) -> list[tuple[str, Any]]:
    alias_set = {_normalise_key(alias) for alias in aliases}
    return [
        (str(key), value)
        for key, value in mapping.items()
        if _normalise_key(key) in alias_set
    ]


def _dataset_name(path: tuple[str, ...]) -> str:
    return "/".join(path) if path else "default"


def _find_prediction_pairs(value: Any) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    visited: set[int] = set()

    def walk(current: Any, path: tuple[str, ...]) -> None:
        if not isinstance(current, (Mapping, list, tuple)):
            return
        identity = id(current)
        if identity in visited:
            return
        visited.add(identity)

        if isinstance(current, Mapping):
            true_items = _matching_items(current, _TRUE_KEYS)
            pred_items = _matching_items(current, _PRED_KEYS)
            score_items = _matching_items(current, _SCORE_KEYS)
            if true_items or pred_items or score_items:
                if len(true_items) != 1:
                    raise EvaluationInputError(
                        f"{_dataset_name(path)} has {len(true_items)} ground-truth candidates"
                    )
                if len(pred_items) > 1 or len(score_items) > 1:
                    raise EvaluationInputError(
                        f"{_dataset_name(path)} has ambiguous prediction/score candidates"
                    )
                if not pred_items and not score_items:
                    raise EvaluationInputError(
                        f"{_dataset_name(path)} has ground truth but no predictions"
                    )
                pairs.append(
                    {
                        "dataset_name": _dataset_name(path),
                        "ground_truth": true_items[0][1],
                        "predictions": pred_items[0][1] if pred_items else None,
                        "scores": score_items[0][1] if score_items else None,
                    }
                )
            for key, item in current.items():
                if isinstance(item, (Mapping, list, tuple)):
                    walk(item, path + (str(key),))
        else:
            for index, item in enumerate(current):
                if isinstance(item, (Mapping, list, tuple)):
                    walk(item, path + (str(index),))

    walk(value, ())
    names = [pair["dataset_name"] for pair in pairs]
    if len(names) != len(set(names)):
        raise EvaluationInputError("duplicate dataset paths found in artifact")
    return pairs


def _select_metric(requested_metric: str | None) -> str | None:
    text = " ".join(str(requested_metric or "").lower().replace("_", " ").split())
    if not text:
        return None
    matches: list[tuple[int, int, str]] = []
    for order, (metric, aliases) in enumerate(_METRIC_ALIASES):
        positions = []
        for alias in aliases:
            match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text)
            if match:
                positions.append(match.start())
        if positions:
            matches.append((min(positions), order, metric))
    if not matches:
        return None
    matches.sort()
    earliest = matches[0][0]
    earliest_metrics = {metric for pos, _, metric in matches if pos == earliest}
    if len(earliest_metrics) != 1:
        return None
    return matches[0][2]


def _as_vector(value: Any, *, field: str, dataset: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise EvaluationInputError(
            f"{dataset} {field} must be one-dimensional; got shape {array.shape}"
        )
    if array.size == 0:
        raise EvaluationInputError(f"{dataset} {field} is empty")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise EvaluationInputError(f"{dataset} {field} contains non-finite values")
    return array


def _labels_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    try:
        return np.equal(left, right)
    except Exception as exc:
        raise EvaluationInputError(f"labels cannot be compared: {exc}") from exc


def _binary_labels(y_true: np.ndarray, dataset: str) -> tuple[np.ndarray, Any]:
    try:
        labels = np.unique(y_true)
    except Exception as exc:
        raise EvaluationInputError(
            f"{dataset} labels cannot be normalized: {exc}"
        ) from exc
    if labels.size != 2:
        raise EvaluationInputError(
            f"{dataset} requires exactly two classes; found {labels.size}"
        )
    label_values = set(labels.tolist())
    if label_values not in ({0, 1}, {-1, 1}, {False, True}):
        raise EvaluationInputError(
            f"{dataset} positive class is ambiguous; use conventional binary labels containing 1"
        )
    positive = 1
    return labels, positive


def _validate_class_labels(labels: np.ndarray, *, field: str, dataset: str) -> None:
    if labels.dtype.kind == "c":
        raise EvaluationInputError(f"{dataset} {field} contains complex values")
    if labels.dtype.kind == "f" and not np.all(labels == np.floor(labels)):
        raise EvaluationInputError(
            f"{dataset} {field} contains continuous values, not class labels"
        )
    if labels.dtype.kind == "O":
        for item in labels.tolist():
            if isinstance(item, (str, bool, int, np.integer)):
                continue
            if (
                isinstance(item, (float, np.floating))
                and math.isfinite(float(item))
                and float(item).is_integer()
            ):
                continue
            raise EvaluationInputError(
                f"{dataset} {field} contains unsupported class label {item!r}"
            )


def _classification_value(
    metric: str,
    y_true: np.ndarray,
    predictions: Any,
    scores: Any,
    dataset: str,
) -> float:
    if metric == "roc_auc":
        if scores is None:
            raise EvaluationInputError(f"{dataset} ROC AUC requires probability scores")
        y_score = np.asarray(scores)
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        y_score = _as_vector(y_score, field="scores", dataset=dataset).astype(float)
        if y_score.size != y_true.size:
            raise EvaluationInputError(
                f"{dataset} scores and ground truth lengths differ"
            )
        _, positive = _binary_labels(y_true, dataset)
        positive_mask = _labels_equal(y_true, positive)
        positives = int(np.sum(positive_mask))
        negatives = int(y_true.size - positives)
        if positives == 0 or negatives == 0:
            raise EvaluationInputError(f"{dataset} ROC AUC requires both classes")
        order = np.argsort(y_score, kind="mergesort")
        sorted_scores = y_score[order]
        ranks = np.empty(y_score.size, dtype=float)
        start = 0
        while start < y_score.size:
            end = start + 1
            while end < y_score.size and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        rank_sum = float(np.sum(ranks[positive_mask]))
        return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    if predictions is None:
        raise EvaluationInputError(f"{dataset} has no discrete predictions")
    _validate_class_labels(y_true, field="ground truth", dataset=dataset)
    y_pred_raw = np.asarray(predictions)
    if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] > 1:
        try:
            expected_labels = set(range(y_pred_raw.shape[1]))
            observed_labels = set(np.unique(y_true).tolist())
        except Exception as exc:
            raise EvaluationInputError(
                f"{dataset} class labels cannot be normalized: {exc}"
            ) from exc
        if not observed_labels.issubset(expected_labels):
            raise EvaluationInputError(
                f"{dataset} probability columns require integer labels in 0..{y_pred_raw.shape[1] - 1}"
            )
        y_pred = np.argmax(y_pred_raw, axis=1)
    else:
        y_pred = _as_vector(y_pred_raw, field="predictions", dataset=dataset)
        _validate_class_labels(y_pred, field="predictions", dataset=dataset)
    if y_pred.size != y_true.size:
        raise EvaluationInputError(
            f"{dataset} predictions and ground truth lengths differ"
        )

    if metric == "accuracy":
        return float(np.mean(_labels_equal(y_true, y_pred)))

    _, positive = _binary_labels(y_true, dataset)
    pred_labels = np.unique(y_pred)
    true_labels = np.unique(y_true)
    if not set(pred_labels.tolist()).issubset(set(true_labels.tolist())):
        raise EvaluationInputError(
            f"{dataset} predictions contain labels absent from ground truth"
        )
    true_positive = _labels_equal(y_true, positive)
    pred_positive = _labels_equal(y_pred, positive)
    tp = int(np.sum(true_positive & pred_positive))
    fp = int(np.sum(~true_positive & pred_positive))
    fn = int(np.sum(true_positive & ~pred_positive))
    if metric == "precision":
        if tp + fp == 0:
            raise EvaluationInputError(f"{dataset} precision is undefined")
        return tp / (tp + fp)
    if metric == "recall":
        if tp + fn == 0:
            raise EvaluationInputError(f"{dataset} recall is undefined")
        return tp / (tp + fn)
    if metric == "f1":
        denominator = 2 * tp + fp + fn
        if denominator == 0:
            raise EvaluationInputError(f"{dataset} F1 is undefined")
        return 2 * tp / denominator
    raise EvaluationInputError(f"unsupported classification metric: {metric}")


def _regression_value(
    metric: str, y_true: np.ndarray, predictions: Any, dataset: str
) -> float:
    if predictions is None:
        raise EvaluationInputError(f"{dataset} has no predictions")
    y_pred = _as_vector(predictions, field="predictions", dataset=dataset).astype(float)
    try:
        y_true_float = y_true.astype(float)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{dataset} ground truth is not numeric") from exc
    if y_pred.size != y_true_float.size:
        raise EvaluationInputError(
            f"{dataset} predictions and ground truth lengths differ"
        )
    residual = y_true_float - y_pred
    mse = float(np.mean(np.square(residual)))
    if metric == "mse":
        return mse
    if metric == "rmse":
        return math.sqrt(mse)
    if metric == "mae":
        return float(np.mean(np.abs(residual)))
    if metric == "r2":
        denominator = float(np.sum(np.square(y_true_float - np.mean(y_true_float))))
        if denominator == 0:
            raise EvaluationInputError(
                f"{dataset} R2 is undefined for constant targets"
            )
        return 1.0 - float(np.sum(np.square(residual))) / denominator
    raise EvaluationInputError(f"unsupported regression metric: {metric}")


def _metric_payload(metric: str, values: list[dict[str, Any]]) -> dict[str, Any]:
    lower_is_better = metric in {"mse", "rmse", "mae"}
    descriptions = {
        "accuracy": "Classification accuracy computed directly from predictions and ground truth.",
        "precision": "Binary precision computed directly from predictions and ground truth.",
        "recall": "Binary recall computed directly from predictions and ground truth.",
        "f1": "Binary F1 score computed directly from predictions and ground truth.",
        "roc_auc": "Binary ROC AUC computed directly from scores and ground truth.",
        "mse": "Mean squared error computed directly from predictions and ground truth.",
        "rmse": "Root mean squared error computed directly from predictions and ground truth.",
        "mae": "Mean absolute error computed directly from predictions and ground truth.",
        "r2": "Coefficient of determination computed directly from predictions and ground truth.",
    }
    return {
        "metric_names": [
            {
                "metric_name": metric,
                "lower_is_better": lower_is_better,
                "description": descriptions[metric],
                "data": [
                    {
                        "dataset_name": item["dataset_name"],
                        "final_value": item["value"],
                        "best_value": item["value"],
                    }
                    for item in values
                ],
            }
        ]
    }


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    report["result_hash"] = _canonical_hash(report)
    return report


def evaluate_experiment_data(
    path: str | Path,
    *,
    requested_metric: str | None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_array_elements: int = DEFAULT_MAX_ARRAY_ELEMENTS,
    max_container_items: int = DEFAULT_MAX_CONTAINER_ITEMS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> dict[str, Any]:
    """Independently compute a supported primary metric from an NPY artifact.

    The function never guesses the requested metric or silently resolves
    ambiguous prediction fields.  A non-verified result is intended to coexist
    with the legacy agent-reported metric path, not to certify it.
    """

    artifact = Path(path)
    base = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluator_hash": _implementation_hash(),
        "requested_metric": requested_metric,
        "selected_metric": None,
        "trust_tier": "unverified",
        "input": {
            "path": artifact.name,
            "sha256": None,
            "size_bytes": None,
        },
        "metric": None,
        "datasets": [],
        "sample_count": 0,
        "safe_for_legacy_parser": False,
        "warnings": [],
    }

    if not artifact.is_file():
        return _finalize_report(
            {**base, "status": "missing", "reason": "experiment_data.npy is missing"}
        )
    if artifact.is_symlink():
        return _finalize_report(
            {
                **base,
                "status": "invalid",
                "reason": "symbolic-link evaluation artifacts are not trusted",
            }
        )

    try:
        artifact_size = artifact.stat().st_size
    except OSError as exc:
        return _finalize_report(
            {**base, "status": "invalid", "reason": f"artifact cannot be read: {exc}"}
        )
    base["input"]["size_bytes"] = artifact_size
    if artifact_size > max_file_bytes:
        return _finalize_report(
            {
                **base,
                "status": "invalid",
                "reason": (
                    f"artifact is {artifact_size} bytes; limit is {max_file_bytes} bytes"
                ),
            }
        )

    try:
        loaded, raw = _read_npy_safely(
            artifact,
            max_file_bytes=max_file_bytes,
            max_container_items=max_container_items,
            max_array_elements=max_array_elements,
        )
        base["input"] = {
            "path": artifact.name,
            "sha256": _sha256_bytes(raw),
            "size_bytes": len(raw),
        }
        _validate_object_graph(
            loaded,
            max_depth=max_depth,
            max_array_elements=max_array_elements,
            max_container_items=max_container_items,
        )
        base["safe_for_legacy_parser"] = True
        selected_metric = _select_metric(requested_metric)
        if selected_metric is None:
            return _finalize_report(
                {
                    **base,
                    "status": "unsupported",
                    "reason": "the requested primary metric is not a supported unambiguous metric",
                }
            )

        base["selected_metric"] = selected_metric
        pairs = _find_prediction_pairs(loaded)
        if not pairs:
            return _finalize_report(
                {
                    **base,
                    "status": "unsupported",
                    "reason": "no colocated prediction and ground-truth fields were found",
                }
            )

        values: list[dict[str, Any]] = []
        sample_count = 0
        for pair in pairs:
            dataset = pair["dataset_name"]
            y_true = _as_vector(
                pair["ground_truth"], field="ground truth", dataset=dataset
            )
            if selected_metric in {"accuracy", "precision", "recall", "f1", "roc_auc"}:
                value = _classification_value(
                    selected_metric,
                    y_true,
                    pair["predictions"],
                    pair["scores"],
                    dataset,
                )
            else:
                value = _regression_value(
                    selected_metric, y_true, pair["predictions"], dataset
                )
            if not math.isfinite(value):
                raise EvaluationInputError(f"{dataset} produced a non-finite metric")
            values.append(
                {
                    "dataset_name": dataset,
                    "value": float(value),
                    "samples": int(y_true.size),
                }
            )
            sample_count += int(y_true.size)

        metric = _metric_payload(selected_metric, values)
        return _finalize_report(
            {
                **base,
                "status": "verified",
                "trust_tier": "deterministic_verified",
                "metric": metric,
                "datasets": values,
                "sample_count": sample_count,
                "reason": "metric recomputed from structured predictions and ground truth",
            }
        )
    except (OSError, EvaluationInputError, ValueError, TypeError) as exc:
        return _finalize_report(
            {
                **base,
                "status": "invalid",
                "safe_for_legacy_parser": False,
                "reason": str(exc),
            }
        )


def evaluation_hash_binding(report: Any) -> dict[str, Any] | None:
    """Return the stable subset that binds a verified evaluation into a node hash."""

    if (
        not isinstance(report, Mapping)
        or report.get("status") != "verified"
        or report.get("trust_tier") != "deterministic_verified"
    ):
        return None
    report_without_result_hash = dict(report)
    recorded_result_hash = report_without_result_hash.pop("result_hash", None)
    try:
        expected_result_hash = _canonical_hash(report_without_result_hash)
    except (TypeError, ValueError):
        return None
    if not recorded_result_hash or recorded_result_hash != expected_result_hash:
        return None
    input_info = report.get("input")
    if not isinstance(input_info, Mapping):
        return None
    required = {
        "schema_version": report.get("schema_version"),
        "evaluator_version": report.get("evaluator_version"),
        "evaluator_hash": report.get("evaluator_hash"),
        "input_hash": input_info.get("sha256"),
        "result_hash": recorded_result_hash,
    }
    if not all(required.values()):
        return None
    for key in ("evaluator_hash", "input_hash", "result_hash"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(required[key])):
            return None
    return required
