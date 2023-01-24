"""
Functions for writing port data into a Pattern (`dev2pat`) and retrieving it (`pat2dev`).

  These use the format 'name:ptype angle_deg' written into labels, which are placed at
the port locations. This particular approach is just a sensible default; feel free to
to write equivalent functions for your own format or alternate storage methods.
"""
from typing import Sequence, Optional, Mapping
import logging

import numpy

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
        library: Mapping[str, Pattern],
        top: str,
        layers: Sequence[layer_t],
        max_depth: int = 999_999,
        skip_subcells: bool = True,
        ) -> Pattern:
    """
    Examine `pattern` for labels specifying port info, and use that info
      to fill out its `ports` attribute.

    Labels are assumed to be placed at the port locations, and have the format
      'name:ptype angle_deg'

    Args:
        pattern: Pattern object to scan for labels.
        layers: Search for labels on all the given layers.
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
    if not isinstance(library, Library):
        library = WrapROLibrary(library)

    ports = {}
    annotated_cells = set()

    def find_ports_each(pat, hierarchy, transform, memo) -> Pattern:
        if len(hierarchy) > max_depth:
            if max_depth >= 999_999:
                logger.warning(f'pat2dev reached max depth ({max_depth})')
            return pat

        if skip_subcells and any(parent in annotated_cells for parent in hierarchy):
            return pat

        cell_name = hierarchy[-1]
        pat2dev_flat(pat, cell_name)

        if skip_subcells:
            annotated_cells.add(cell_name)

        mirr_factor = numpy.array((1, -1)) ** transform[3]
        rot_matrix = rotation_matrix_2d(transform[2])
        for name, port in pat.ports.items():
            port.offset = transform[:2] + rot_matrix @ (port.offset * mirr_factor)
            port.rotation = port.rotation * mirr_factor[0] * mirr_factor[1] + transform[2]
            ports[name] = port

        return pat

    # update `ports`
    library.dfs(top=top, visit_before=find_ports_each, transform=True)

    pattern = library[top]
    pattern.check_ports(other_names=ports.keys())
    pattern.ports.update(ports)
    return pattern


def pat2dev_flat(
        pattern: Pattern,
        layers: Sequence[layer_t],
        cell_name: Optional[str] = None,
        ) -> Pattern:
    """
    Examine `pattern` for labels specifying port info, and use that info
      to fill out its `ports` attribute.

    Labels are assumed to be placed at the port locations, and have the format
      'name:ptype angle_deg'

    The pattern is assumed to be flat (have no `refs`) and have no pre-existing ports.

    Args:
        pattern: Pattern object to scan for labels.
        layers: Search for labels on all the given layers.
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

