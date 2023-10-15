masque Tutorial
===============

Contents
--------

- [basic_shapes](basic_shapes.py):
    * Draw basic geometry
    * Export to GDS
- [devices](devices.py)
    * Reference other patterns
    * Add ports to a pattern
    * Snap ports together to build a circuit
    * Check for dangling references
- [library](library.py)
    * Create a `LazyLibrary`, which loads / generates patterns only when they are first used
    * Explore alternate ways of specifying a pattern for `.plug()` and `.place()`
    * Design a pattern which is meant to plug into an existing pattern (via `.interface()`)
- [pather](pather.py)
    * Use `Pather` to route individual wires and wire bundles
    * Use `BasicTool` to generate paths
    * Use `BasicTool` to automatically transition between path types
- [renderpather](rendpather.py)
    * Use `RenderPather` and `PathTool` to build a layout similar to the one in [pather](pather.py),
        but using `Path` shapes instead of `Polygon`s.


Additionaly, [pcgen](pcgen.py) is a utility module for generating photonic crystal lattices.


Running
-------

Run from inside the examples directory:
```bash
cd examples/tutorial
python3 basic_shapes.py
klayout -e basic_shapes.gds
```
