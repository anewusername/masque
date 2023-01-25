"""
Functions for writing port data into a Pattern (`dev2pat`) and retrieving it (`pat2dev`).

  These use the format 'name:ptype angle_deg' written into labels, which are placed at
the port locations. This particular approach is just a sensible default; feel free to
to write equivalent functions for your own format or alternate storage methods.
"""
from typing import Sequence, Optional, Mapping, Tuple, Dict
import logging

import numpy
from numpy.typing import NDArray

from ..pattern import Pattern
from ..label import Label
from ..utils import rotation_matrix_2d, layer_t
from ..ports import Port
from ..error import PatternError
from ..library import Library, WrapROLibrary


logger = logging.getLogger(__name__)


def dev2pat(pattern: Pattern, layer: layer_t) -> Pattern:
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
        pattern.labels += [
            Label(string=f'{name}:{port.ptype} {angle_deg:g}', layer=layer, offset=port.offset)
            ]
    return pattern


def pat2dev(
        layers: Sequence[layer_t],
        library: Mapping[str, Pattern],
        pattern: Pattern,               # Pattern is good since we don't want to do library[name] to avoid infinite recursion.
                                        # LazyLibrary protects against library[ref.target] causing a circular lookup.
                                        # For others, maybe check for cycles up front? TODO
        name: Optional[str] = None,
        max_depth: int = 999_999,
        skip_subcells: bool = True,
        ) -> Pattern:
    """
    # TODO fixup documentation in port_utils
    # TODO move port_utils to utils.file?
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
        blacklist: If a cell name appears in the blacklist, do not ea

    Returns:
        The updated `pattern`. Port labels are not removed.
    """
    print(f'TODO pat2dev {name}')
    if pattern.ports:
        logger.warning(f'Pattern {name if name else pattern} already had ports, skipping pat2dev')
        return pattern

    if not isinstance(library, Library):
        library = WrapROLibrary(library)

    pat2dev_flat(layers, pattern, name)
    if (skip_subcells and pattern.ports) or max_depth == 0:
        return pattern

    # Load ports for all subpatterns, and use any we find
    found_ports = False
    for target in set(rr.target for rr in pattern.refs):
        pp = pat2dev(
            layers=layers,
            library=library,
            pattern=library[target],
            name=target,
            max_depth=max_depth - 1,
            skip_subcells=skip_subcells,
            blacklist=blacklist + {name},
            )
        found_ports |= bool(pp.ports)

    if not found_ports:
        return pattern

    for ref in pattern.refs:
        aa = library.abstract(ref.target)
        if not aa.ports:
            continue

        aa.apply_ref_transform(ref)

        pattern.check_ports(other_names=aa.ports.keys())
        pattern.ports.update(aa.ports)
    return pattern


def pat2dev_flat(
        layers: Sequence[layer_t],
        pattern: Pattern,
        cell_name: Optional[str] = None,
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
    labels = [ll for ll in pattern.labels if ll.layer in layers]
    if not labels:
        return pattern

    pstr = cell_name if cell_name is not None else repr(pattern)
    if pattern.ports:
        raise PatternError('Pattern "{pstr}" has pre-existing ports!')

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

