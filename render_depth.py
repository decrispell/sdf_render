""" render a depth image using a distance transform """
import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt

def main():
    """ main """
    vol_og = np.zeros(3,np.float32)
    vol_width = np.float32(100)
    vol_dims = np.int32(100)

    img_dims = np.array((1000,1000), np.int32)

    # the distance field
    df = np.zeros((vol_dims,vol_dims,vol_dims), np.float32)

    xvals = np.linspace(vol_og[0],vol_og[0] + vol_width, vol_dims)
    yvals = np.linspace(vol_og[1],vol_og[1] + vol_width, vol_dims)
    zvals = np.linspace(vol_og[2],vol_og[2] + vol_width, vol_dims)

    # reverse x,y,z so x is the fastest-changing index
    Z,Y,X = np.meshgrid(zvals,yvals,xvals, indexing='ij')
    # plane at z = 25
    #df[:] = 25.0 - Z
    df[:] = (X-50.0)*(X-50.0)+(Y-50)*(Y-50)+(Z-50)*(Z-50) - 40*40

    # the image rays
    ray_origins = np.zeros((img_dims[0], img_dims[1], 4), np.float32)
    img_og = np.array((0,0,-50,1))
    pix_size = 0.1
    img_x_axis = np.array((pix_size,0,0,0))
    img_y_axis = np.array((0,pix_size,0,0))
    for i in range(img_dims[0]):
        for j in range(img_dims[1]):
            ray_origins[i,j] = img_og + img_x_axis*j + img_y_axis*i
    ray_dirs = np.zeros((img_dims[0], img_dims[1], 4), np.float32)
    ray_dirs[:,:] = np.array((0,0,1,0))
    # the depth image to be filled in
    depth = np.zeros(img_dims[0:2], np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    df_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=df)
    vol_og_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vol_og)
    img_dims_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img_dims)
    ray_origins_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ray_origins)
    ray_dirs_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ray_dirs)

    depth_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=depth)

    with open('./distance_render.cl','r') as fd:
        kernel_str = fd.read()

    prg = cl.Program(ctx, kernel_str).build()

    prg.render_distance_field(queue, depth.shape, None, df_cl, vol_og_cl, vol_dims, vol_width, ray_origins_cl, ray_dirs_cl, img_dims_cl, depth_cl)

    cl.enqueue_copy(queue, depth, depth_cl)

    print('min(depth) = ' + str(np.nanmin(depth)))
    print('max(depth) = ' + str(np.nanmax(depth)))

    #plt.interactive(True)
    plt.imshow(depth, cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
