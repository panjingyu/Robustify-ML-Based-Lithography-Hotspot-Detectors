import gdspy
import numpy as np
from skimage.draw import rectangle


LEN = 2048
STEP = 1e-3     # GDS sampling step size (in microns)
TOP_KEY = 'TOP_new'
SRAFS_KEY = (2, 0)
VIAS_KEY = (0, 0)


def get_gds_top_layers(gds_path):
    gdsii = gdspy.GdsLibrary(infile=gds_path)
    layers = gdsii.cells[TOP_KEY].get_polygons(by_spec=True)
    return layers

def get_srafs_vias_from_gds(gds_path):
    layers = get_gds_top_layers(gds_path)
    srafs = layers[SRAFS_KEY]
    vias = layers[VIAS_KEY]
    return srafs, vias

def get_layout_location(layout):
    co = np.array(layout).reshape(-1, 2)
    x_min, y_min = np.min(co, axis=0)
    x_max, y_max = np.max(co, axis=0)
    return x_min, y_min, x_max, y_max

def to_grid(co, offset, step):
    return np.uint32(np.floor((co + offset) / step))

def shape_to_grid(shape, x_offset, y_offset, step):
    x1, y1, x2, y2 = get_layout_location(shape)
    x1 = to_grid(x1, x_offset, step)
    x2 = to_grid(x2, x_offset, step)
    y1 = to_grid(y1, y_offset, step)
    y2 = to_grid(y2, y_offset, step)
    return x1, x2, y1, y2

def gen_shapes(shapes, out_shape, x_offset, y_offset, step):
    gen_img = np.zeros(out_shape, dtype=np.uint8)
    for v in shapes:
        x1, x2, y1, y2 = shape_to_grid(v, x_offset, y_offset, step)
        rec = rectangle((y1, x1), end=(y2, x2))
        gen_img[tuple(rec)] = 255
    return gen_img

def gen_merge_img_from_gds(gds_path):
    srafs, vias = get_srafs_vias_from_gds(gds_path)
    x_min, y_min, x_max, y_max = get_layout_location(srafs)
    x_offset = (LEN * STEP - x_max - x_min) / 2
    y_offset = (LEN * STEP - y_max - y_min) / 2
    img_vias = gen_shapes(vias, (LEN, LEN), x_offset, y_offset, STEP)
    img_srafs = gen_shapes(srafs, (LEN, LEN), x_offset, y_offset, STEP)
    return img_vias + img_srafs

def gen_srafs_vias_from_gds(gds_path):
    srafs, vias = get_srafs_vias_from_gds(gds_path)
    x_min, y_min, x_max, y_max = get_layout_location(srafs)
    x_offset = (LEN * STEP - x_max - x_min) / 2
    y_offset = (LEN * STEP - y_max - y_min) / 2
    for i, s in enumerate(srafs):
        x1, x2, y1, y2 = shape_to_grid(s, x_offset, y_offset, STEP)
        srafs[i] = [y1, x1, y2-y1, x2-x1]
    for i, v in enumerate(vias):
        x1, x2, y1, y2 = shape_to_grid(v, x_offset, y_offset, STEP)
        vias[i] = [y1, x1, y2-y1, x2-x1]
    return srafs, vias
