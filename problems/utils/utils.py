
import numpy as np
def deg2vec(deg):
    raid = np.deg2rad(deg)
    cosx = np.cos(raid)
    sinx = np.sin(raid)

    print('ok')
    return (cosx,sinx)

