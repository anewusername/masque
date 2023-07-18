"""
Functions for writing port data into Pattern geometry/annotations/labels (`ports_to_data`)
and retrieving it (`data_to_ports`).

  These use the format 'name:ptype angle_deg' written into labels, which are placed at
the port locations. This particular approach is just a sensible default; feel free to
to write equivalent functions for your own format or alternate storage methods.
"""
from typing import Sequence, Mapping
import logging
from itertools import chain

import numpy

from ..pattern import Pattern
from ..utils import layer_t
from ..ports import Port
from ..error import PatternError
from ..library import ILibraryView, LibraryView


logger = logging.getLogger(__name__)


def ports_to_data(pattern: Pattern, layer: layer_t) -> Pattern:
    """
    Place a text label at each port location, specifying the port data in the format
        'name:ptype angle_deg'

    This can be used to debug port locations or to automatically generate ports
      when reading in a GDS file.

    NOTE that `pattern` is modified by this function

    Args:
        pattern: The pattern which is to have its ports labeled. MODIFIED in-place.
        layer: The layer on which the labels will be placed.

    Returns:
        `pattern`
    """
    for name, port in pattern.ports.items():
        if port.rotation is None:
            angle_deg = numpy.inf
        else:
            angle_deg = numpy.rad2deg(port.rotation)
        pattern.label(layer=layer, string=f'{name}:{port.ptype} {angle_deg:g}', offset=port.offset)
    return pattern


def data_to_ports(
        layers: Sequence[layer_t],
        library: Mapping[str, Pattern],
        pattern: Pattern,               # Pattern is good since we don't want to do library[name] to avoid infinite recursion.
                                        # LazyLibrary protects against library[ref.target] causing a circular lookup.
                                        # For others, maybe check for cycles up front? TODO
        name: str | None = None,     # Note: name optional, but arg order different from read(postprocess=)
        max_depth: int = 0,
        skip_subcells: bool = True,
        # TODO missing ok?
        ) -> Pattern:
    """
    # TODO fixup documentation in ports2data
    # TODO move to utils.file?
    Examine `pattern` for labels specifying port info, and use that info
      to fill out its `ports` attribute.

    Labels are assumed to be placed at the port locations, and have the format
      'name:ptype angle_deg'

    Args:
        layers: Search for labels on all the given layers.
        pattern: Pattern object to scan for labels.
        max_depth: Maximum hierarcy depth to search. Default 999_999.
            Reduce this to 0 to avoid ever searching subcells.
        skip_subcells: If port labels are found at a given hierarcy level,
            do not continue searching at deeper levels. This allows subcells
            to contain their own port info without interfering with supercells'
            port data.
            Default True.

    Returns:
        The updated `pattern`. Port labels are not removed.
    """
    if pattern.ports:
        logger.warning(f'Pattern {name if name else pattern} already had ports, skipping data_to_ports')
        return pattern

    if not isinstance(library, ILibraryView):
        library = LibraryView(library)

    data_to_ports_flat(layers, pattern, name)
    if (skip_subcells and pattern.ports) or max_depth == 0:
        return pattern

    # Load ports for all subpatterns, and use any we find
    found_ports = False
    for target in pattern.refs:
        if target is None:
            continue
        pp = data_to_ports(
            layers=layers,
            library=library,
            pattern=library[target],
            name=target,
            max_depth=max_depth - 1,
            skip_subcells=skip_subcells,
            )
        found_ports |= bool(pp.ports)

    if not found_ports:
        return pattern

    for target, refs in pattern.refs.items():
        if target is None:
            continue
        if not refs:
            continue

        for ref in refs:
            aa = library.abstract(target)
            if not aa.ports:
                break

            aa.apply_ref_transform(ref)
            pattern.check_ports(other_names=aa.ports.keys())
            pattern.ports.update(aa.ports)
    return pattern


def data_to_ports_flat(
        layers: Sequence[layer_t],
        pattern: Pattern,
        cell_name: str | None = None,
        ) -> Pattern:
    """
    Examine `pattern` for labels specifying port info, and use that info
      to fill out its `ports` attribute.

    Labels are assumed to be placed at the port locations, and have the format
      'name:ptype angle_deg'

    The pattern is assumed to be flat (have no `refs`) and have no pre-existing ports.

    Args:
        layers: Search for labels on all the given layers.
        pattern: Pattern object to scan for labels.
        cell_name: optional, used for warning message only

    Returns:
        The updated `pattern`. Port labels are not removed.
    """
    labels = list(chain.from_iterable((pattern.labels[layer] for layer in layers)))
    if not labels:
        return pattern

    pstr = cell_name if cell_name is not None else repr(pattern)
    if pattern.ports:
        raise PatternError(f'Pattern "{pstr}" has pre-existing ports!')

    local_ports = {}
    for label in labels:
        name, property_string = label.string.split(':')
        properties = property_string.split(' ')
        ptype = properties[0]
        angle_deg = float(properties[1]) if len(ptype) else 0

        xy = label.offset
        angle = numpy.deg2rad(angle_deg)

        if name in local_ports:
            logger.warning(f'Duplicate port "{name}" in pattern "{pstr}"')

        local_ports[name] = Port(offset=xy, rotation=angle, ptype=ptype)

    pattern.ports.update(local_ports)
    return pattern

