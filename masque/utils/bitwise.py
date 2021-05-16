from typing import Any


def get_bit(bit_string: Any, bit_id: int) -> bool:
    """
    Interprets bit number `bit_id` from the right (lsb) of `bit_string` as a boolean

    Args:
        bit_string: Bit string to test
        bit_id: Bit number, 0-indexed from the right (lsb)

    Returns:
        Boolean value of the requested bit
    """
    return bit_string & (1 << bit_id) != 0


def set_bit(bit_string: Any, bit_id: int, value: bool) -> Any:
    """
    Returns `bit_string`, with bit number `bit_id` set to boolean `value`.

    Args:
        bit_string: Bit string to alter
        bit_id: Bit number, 0-indexed from right (lsb)
        value: Boolean value to set bit to

    Returns:
        Altered `bit_string`
    """
    mask = (1 << bit_id)
    bit_string &= ~mask
    if value:
        bit_string |= mask
    return bit_string
