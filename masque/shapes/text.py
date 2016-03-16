
#
# class Text(Shape):
#     _string = ''
#     _height = 1.0
#     _rotation = 0.0
#     font_path = ''
#
#     # vertices property
#     @property
#     def string(self):
#         return self._string
#
#     @string.setter
#     def string(self, val):
#         self._string = val
#
#     # Rotation property
#     @property
#     def rotation(self):
#         return self._rotation
#
#     @rotation.setter
#     def rotation(self, val):
#         if not is_scalar(val):
#             raise PatternError('Rotation must be a scalar')
#         self._rotation = val % (2 * pi)
#
#     # Height property
#     @property
#     def height(self):
#         return self._height
#
#     @height.setter
#     def height(self, val):
#         if not is_scalar(val):
#             raise PatternError('Height must be a scalar')
#         self._height = val
#
#     def __init__(self, text, height, font_path, rotation=0.0, offset=(0.0, 0.0), layer=0, dose=1.0):
#         self.offset = offset
#         self.layer = layer
#         self.dose = dose
#         self.text = text
#         self.height = height
#         self.rotation = rotation
#         self.font_path = font_path
#
#     def to_polygon(self, _poly_num_points=None, _poly_max_arclen=None):
#
#         return copy.deepcopy(self)
#
#     def rotate(self, theta):
#         self.rotation += theta
#
#     def scale_by(self, c):
#         self.height *= c
