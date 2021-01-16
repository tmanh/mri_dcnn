import numpy as np
import mahotas as mt


# NOTE: The image need to be delineated first as stated in the paper
# "each PCa lesion was first delineated manually by experienced radiologists"
def compute_handcraft_features(data):
    h, w = np.shape(data)

    x = range(w)
    y = range(h)

    X, Y = np.meshgrid(x,y)

    # Centroid (mean)
    cx = np.sum(data * X) / np.sum(data)
    cy = np.sum(data * Y) / np.sum(data)

    # Standard deviation
    x2 = (range(w) - cx)**2
    y2 = (range(h) - cy)**2

    X2, Y2 = np.meshgrid(x2,y2)

    # Find the variance
    vx = np.sum(data * X2) / np.sum(data)
    vy = np.sum(data * Y2) / np.sum(data)

    # SD is the sqrt of the variance
    sx, sy = np.sqrt(vx),np.sqrt(vy)

    # Skewness
    x3 = (range(w) - cx) ** 3
    y3 = (range(h) - cy) ** 3

    X3, Y3 = np.meshgrid(x3, y3)

    # Find the thid central moment
    m3x = np.sum(data * X3) / np.sum(data)
    m3y = np.sum(data * Y3) / np.sum(data)

    # Skewness is the third central moment divided by SD cubed
    skx = m3x / sx**3
    sky = m3y / sy**3

    # Kurtosis
    x4 = (range(w) - cx) ** 4
    y4 = (range(h) - cy) ** 4

    X4, Y4 = np.meshgrid(x4, y4)

    # Find the fourth central moment
    m4x = np.sum(data * X4) / np.sum(data)
    m4y = np.sum(data * Y4) / np.sum(data)

    # Kurtosis is the fourth central moment divided by SD to the fourth power
    kx = m4x / sx ** 4
    ky = m4y / sy ** 4

    return np.array([cx, cy, sx, sy, skx, sky, kx, ky], dtype=np.float32)


# NOTE: The image need to be delineated first as stated in the paper
# "each PCa lesion was first delineated manually by experienced radiologists"
def compute_haralick_features(data):
    textures = mt.features.haralick(data, compute_14th_feature=True)
    return textures.reshape(-1)
