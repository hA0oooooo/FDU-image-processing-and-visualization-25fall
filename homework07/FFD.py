import numpy as np

def cubic_spline(u, index: int):
    r"""another expression of FFD where u belongs to [0, 1)

    Args:
        u: one component of the control point coordinates
        index: -1, 0, 1, 2
    Return:
        weight: the weight of the control point in certain dimension
    """
    if index == -1:
        weight = (1 - u)**3 / 6
    elif index == 0:
        weight = (3*u**3 - 6*u**2 + 4) / 6
    elif index == 1:
        weight = (-3*u**3 + 3*u**2 + 3*u + 1) / 6
    elif index == 2:
        weight = u**3 / 6
    else:
        raise ValueError("index is not {-1, 0, 1, 2}")
    
    return weight


def construct_controlShiftDist(height: int, width: int, m: int, n: int):
    r"""
    construct initial controlShiftDist

    Args:
        height, width: about image
        m, n: the number of intervals in height and width direction
              I'll choose m and n by which the image size is divisible.
    
    Returns:
        controlShiftDist: {(index_x, index_y): (0, 0)}, note that x and y are index, not true coordinates
        lx: length of intervals in height direciton
        ly: length of intervals in width direciton
    """
    controlShiftDist = {}
    lx = height / m
    ly = width / n
    for i in range(m + 1):
        for j in range(n + 1):
            controlShiftDist[(i, j)] = np.array([0, 0])
    return controlShiftDist, lx, ly

def ffd(height: int, width: int, lx: int, ly: int, controlShiftDist: dict) -> np.array:
    r"""
    Free-form deformation, given the shift of control points.
    obtain the coordinates of the target image inversely mapped back to the original image.

    Args:
        height, width: about image
        lx, ly: the length of interval in either height and width direction
        controlShiftDict: {(index_x, index_y): (delta_x, delta_y)}
    Return:
        backmap: tensor(height * width * 2), backmap[:, x, y] gets the backmap coordinates of (x, y)
    """
    # backmap[x, y, 1] has the first coordinate, backmap[x, y, 2] has the second one
    backmap = np.zeros((height, width, 2))
    for x in range(height):
        for y in range(width):
            backmap[x, y, :] = np.array([x, y])
    
    # compyte index: int and residual: [0, 1)
    for x in range(height):
        dist_1 = (x - 0) / lx
        ix = int(dist_1)  
        u = dist_1 - ix  
        for y in range(width):
            dist_2 = (y - 0) / ly
            iy = int(dist_2)  
            v = dist_2 - iy  

            # compute weighted shift, using cubic spline
            shiftSum = np.array([0.0, 0.0])
            for i in [-1, 0, 1, 2]:
                for j in [-1, 0, 1, 2]:
                    delta = controlShiftDist.get((ix + i, iy + j), np.array([0, 0]))
                    shiftSum[0] += delta[0] * cubic_spline(u, i) * cubic_spline(v, j)
                    shiftSum[1] += delta[1] * cubic_spline(u, i) * cubic_spline(v, j)
            
            backmap[x, y, :] = np.array([x, y]) + shiftSum
    
    # trip
    backmap[:, :, 0] = np.clip(backmap[:, :, 0], 0, height - 1)
    backmap[:, :, 1] = np.clip(backmap[:, :, 1], 0, width - 1)

    return backmap