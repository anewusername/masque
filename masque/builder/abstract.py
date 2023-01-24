from typing import Dict, Union, Optional
from typing import MutableMapping, TYPE_CHECKING
import copy
import logging

from ..pattern import Pattern
from ..library import MutableLibrary
from ..ports import PortList, Port
from .tools import Tool

if TYPE_CHECKING:
    from .builder import Builder


logger = logging.getLogger(__name__)


class Abstract(PortList):
    __slots__ = ('name', 'ports')

    name: str
    """ Name of the pattern this device references """

    ports: Dict[str, Port]
    """ Uniquely-named ports which can be used to snap instances together"""

    def __init__(
            self,
            name: str,
            ports: Dict[str, Port],
            ) -> None:
        self.name = name
        self.ports = copy.deepcopy(ports)

    def build(
            self,
            library: MutableLibrary,
            tools: Union[None, Tool, MutableMapping[Optional[str], Tool]] = None,
            ) -> 'Builder':
        """
        Begin building a new device around an instance of the current device
          (rather than modifying the current device).

        Returns:
            The new `Builder` object.
        """
        pat = Pattern(ports=self.ports)
        pat.ref(self.name)
        new = Builder(library=library, pattern=pat, tools=tools)   # TODO should Ref have tools?
        return new

    # TODO do we want to store a Ref instead of just a name? then we can translate/rotate/mirror...

    def __repr__(self) -> str:
        s = f'<Abstract {self.name} ['
        for name, port in self.ports.items():
            s += f'\n\t{name}: {port}'
        s += ']>'
        return s
