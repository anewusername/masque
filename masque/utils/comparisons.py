from typing import Any

from .types import annotations_t, layer_t
from ..ports import Port
from ..repetition import Repetition


def annotation2key(aaa: int | float | str) -> tuple[bool, Any]:
    return (isinstance(aaa, str), aaa)


def annotations_lt(aa: annotations_t, bb: annotations_t) -> bool:
    if aa is None:
        return bb is not None
    elif bb is None:
        return False

    if len(aa) != len(bb):
        return len(aa) < len(bb)

    keys_a = tuple(sorted(aa.keys()))
    keys_b = tuple(sorted(bb.keys()))
    if keys_a != keys_b:
        return keys_a < keys_b

    for key in keys_a:
        va = aa[key]
        vb = bb[key]
        if len(va) != len(vb):
            return len(va) < len(vb)

        for aaa, bbb in zip(va, vb):
            if aaa != bbb:
                return annotation2key(aaa) < annotation2key(bbb)
    return False


def annotations_eq(aa: annotations_t, bb: annotations_t) -> bool:
    if aa is None:
        return bb is None
    elif bb is None:
        return False

    if len(aa) != len(bb):
        return False

    keys_a = tuple(sorted(aa.keys()))
    keys_b = tuple(sorted(bb.keys()))
    if keys_a != keys_b:
        return keys_a < keys_b

    for key in keys_a:
        va = aa[key]
        vb = bb[key]
        if len(va) != len(vb):
            return False

        for aaa, bbb in zip(va, vb):
            if aaa != bbb:
                return False

    return True


def layer2key(layer: layer_t) -> tuple[bool, bool, Any]:
    is_int = isinstance(layer, int)
    is_str = isinstance(layer, str)
    layer_tup = (layer) if (is_str or is_int) else layer
    tup = (
        is_str,
        not is_int,
        layer_tup,
        )
    return tup


def rep2key(repetition: Repetition | None) -> tuple[bool, Repetition | None]:
    return (repetition is None, repetition)


def ports_eq(aa: dict[str, Port], bb: dict[str, Port]) -> bool:
    if len(aa) != len(bb):
        return False

    keys = sorted(aa.keys())
    if keys != sorted(bb.keys()):
        return False

    return all(aa[kk] == bb[kk] for kk in keys)


def ports_lt(aa: dict[str, Port], bb: dict[str, Port]) -> bool:
    if len(aa) != len(bb):
        return len(aa) < len(bb)

    aa_keys = tuple(sorted(aa.keys()))
    bb_keys = tuple(sorted(bb.keys()))
    if aa_keys != bb_keys:
        return aa_keys < bb_keys

    for key in aa_keys:
        pa = aa[key]
        pb = bb[key]
        if pa != pb:
            return pa < pb
    return False
