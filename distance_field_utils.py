""" utility functions for signed distance fields """
import vtk
import numpy as np

def load_vti(filename):
    """ load sdf in vti format """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    bounds = data.GetBounds()
    extent = data.GetExtent()

    pd = data.GetPointData()
    a = pd.GetArray(0)

    xshape = np.array((extent[1],extent[3],extent[5])) - np.array((extent[0],extent[2],extent[4])) + 1
    xshape_rev = xshape[-1::-1]
    x = np.zeros(xshape_rev,np.float32)

    for zi in range(xshape[2]):
        for yi in range(xshape[1]):
            for xi in range(xshape[0]):
                idx = zi*xshape[0]*xshape[1] + yi*xshape[0] + xi
                x[zi,yi,xi] = a.GetValue(idx)

    return x, bounds

