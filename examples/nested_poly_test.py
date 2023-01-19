import numpy
from pyclipper import (
    Pyclipper, PT_CLIP, PT_SUBJECT, CT_UNION, CT_INTERSECTION, PFT_NONZERO,
    scale_to_clipper, scale_from_clipper,
    )
p = Pyclipper()
p.AddPaths([
  [(-10, -10), (-10, 10), (-9, 10), (-9, -10)],
  [(-10, 10), (10, 10), (10, 9), (-10, 9)],
  [(10, 10), (10, -10), (9, -10), (9, 10)],
  [(10, -10), (-10, -10), (-10, -9), (10, -9)],
  ], PT_SUBJECT, closed=True)
#p.Execute2?
#p.Execute?
p.Execute(PT_UNION, PT_NONZERO, PT_NONZERO)
p.Execute(CT_UNION, PT_NONZERO, PT_NONZERO)
p.Execute(CT_UNION, PFT_NONZERO, PFT_NONZERO)

p = Pyclipper()
p.AddPaths([
  [(-10, -10), (-10, 10), (-9, 10), (-9, -10)],
  [(-10, 10), (10, 10), (10, 9), (-10, 9)],
  [(10, 10), (10, -10), (9, -10), (9, 10)],
  [(10, -10), (-10, -10), (-10, -9), (10, -9)],
  ], PT_SUBJECT, closed=True)
r = p.Execute2(CT_UNION, PFT_NONZERO, PFT_NONZERO)

#r.Childs

