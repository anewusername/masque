"""
Functions for writing port data into a Pattern (`dev2pat`) and retrieving it (`pat2dev`).

  These use the format 'name:ptype angle_deg' written into labels, which are placed at
the port locations. This particular approach is just a sensible default; feel free to
to write equivalent functions for your own format or alternate storage methods.
"""
from typing import Sequence
import logging

import numpy

from ..pattern import Pattern
from ..label import Label
from ..utils import rotation_matrix_2d, layer_t
from .devices import Device, Port


logger = logging.getLogger(__name__)


def dev2pat(device: Device, layer: layer_t) -> Pattern:
    """
    Place a text label at each port location, specifying the port data in the format
        'name:ptype angle_deg'

    This can be used to debug port locations or to automatically generate ports
      when reading in a GDS file.

    NOTE that `device` is modified by this function, and `device.pattern` is returned.

    Args:
        device: The device which is to have its ports labeled. MODIFIED in-place.
        layer: The layer on which the labels will be placed.

    Returns:
        `device.pattern`
    """
    for name, port in device.ports.items():
        if port.rotation is None:
            angle_deg = numpy.inf
        else:
            angle_deg = numpy.rad2deg(port.rotation)
        device.pattern.labels += [
            Label(string=f'{name}:{port.ptype} {angle_deg:g}', layer=layer, offset=port.offset)
            ]
    return device.pattern


def pat2dev(
        pattern: Pattern,
        layers: Sequence[layer_t],
        max_depth: int = 999_999,
        skip_subcells: bool = True,
        ) -> Device:
    """
    Examine `pattern` for labels specifying port info, and use that info
      to build a `Device` object.

    Labels are assumed to be placed at the port locations, and have the format
      'name:ptype angle_deg'

    Args:
        pattern: Pattern object to scan for labels.
        layers: Search for labels on all the given layers.
        max_depth: Maximum hierarcy depth to search. Default 999_999.
            Reduce this to 0 to avoid ever searching subcells.
        skip_subcells: If port labels are found at a given hierarcy level,
            do not continue searching at deeper levels. This allows subcells
            to contain their own port info (and thus become their own Devices).
            Default True.

    Returns:
        The constructed Device object. Port labels are not removed from the pattern.
    """
    ports = {}      # Note: could do a list here, if they're not unique
    annotated_cells = set()
    def find_ports_each(pat, hierarchy, transform, memo) -> Pattern:
        if len(hierarchy) > max_depth - 1:
            return pat

        if skip_subcells and any(parent in annotated_cells for parent in hierarchy):
            return pat

        labels = [ll for ll in pat.labels if ll.layer in layers]

        if len(labels) == 0:
            return pat

        if skip_subcells:
            annotated_cells.add(pat)

        mirr_factor = numpy.array((1, -1)) ** transform[3]
        rot_matrix = rotation_matrix_2d(transform[2])
        for label in labels:
            name, property_string = label.string.split(':')
            properties = property_string.split(' ')
            ptype = properties[0]
            angle_deg = float(properties[1]) if len(ptype) else 0

            xy_global = transform[:2] + rot_matrix @ (label.offset * mirr_factor)
            angle = numpy.deg2rad(angle_deg) * mirr_factor[0] * mirr_factor[1] + transform[2]

            if name in ports:
                logger.info(f'Duplicate port {name} in pattern {pattern.name}')

            ports[name] = Port(offset=xy_global, rotation=angle, ptype=ptype)

        return pat

    pattern.dfs(visit_before=find_ports_each, transform=True)
    return Device(pattern, ports)
