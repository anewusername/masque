# pip install pillow scikit-image
# or
# sudo apt install python3-pil python3-skimage

from PIL import Image
from skimage.measure import find_contours
from matplotlib import pyplot
import numpy

from masque import Pattern, Polygon
from masque.file.gdsii import writefile

#
# Read the image into a numpy array
#
im = Image.open('./Desktop/Camera/IMG_20220626_091101.jpg')

aa = numpy.array(im.convert(mode='L').getdata()).reshape(im.height, im.width)

threshold = (aa.max() - aa.min()) / 2

#
# Find edge contours and plot them
#
contours = find_contours(aa, threshold)

pyplot.imshow(aa)
for contour in contours:
    pyplot.plot(contour[:, 1], contour[:, 0], linewidth=2)
pyplot.show(block=False)

#
# Create the layout from the contours
#
pat = Pattern()
pat.shapes[(0, 0)].extend([
    Polygon(vertices=vv) for vv in contours if len(vv) < 1_000
    ])

lib = {}
lib['my_mask_name'] = pat

writefile(lib, 'test_contours.gds', meters_per_unit=1e-9)
