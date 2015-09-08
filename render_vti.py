""" render a depth image using a distance transform """
import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import distance_field_utils
import io_utils

def main():
    """ main """

    # the distance field
    filename = '/Users/dec/projects/df_render/models/mean_face_watertight.vti'
    print('reading ' + filename)
    df, bounds = distance_field_utils.load_vti(filename)
    print('done.')
    vol_og = np.array((bounds[0],bounds[2],bounds[4]),np.float32)
    df_extents = np.array((bounds[1],bounds[3],bounds[5])) - np.array((bounds[0],bounds[2],bounds[4]))
    vol_width = np.float32(np.max(df_extents))
    vol_dims =  np.int32(np.max(df.shape))
    df_cube = np.zeros((vol_dims,vol_dims,vol_dims),np.float32) + 10000.0
    df_cube[0:df.shape[0],0:df.shape[1],0:df.shape[2]] = df

    print('volume dimensions = ' + str(df.shape))

    # the image rays
    img_dims = np.array((2000,2000), np.int32)
    ray_origins = np.zeros((img_dims[0], img_dims[1], 4), np.float32)
    img_og = np.array((-1000,150,-150,1))
    pix_size = 0.15
    img_x_axis = np.array((0,0,pix_size,0))
    img_y_axis = np.array((0,-pix_size,0,0))
    for i in range(img_dims[0]):
        for j in range(img_dims[1]):
            ray_origins[i,j] = img_og + img_x_axis*j + img_y_axis*i
    ray_dirs = np.zeros((img_dims[0], img_dims[1], 4), np.float32)
    ray_dirs[:,:] = np.array((1,0,0,0))
    # the depth image to be filled in
    depth = np.zeros(img_dims[0:2], np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    df_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=df_cube)
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

    good_mask = np.isfinite(depth)
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)

    print('min(depth) = ' + str(depth_min))
    print('max(depth) = ' + str(depth_max))

    #depth_min,depth_max = np.percentile(depth[good_mask],(0.01,99.99))

    # save out image 
    depth_norm = np.zeros_like(depth)
    depth_norm[good_mask] = (depth[good_mask] - depth_min)/(depth_max - depth_min)
    depth_norm[depth_norm < 0] = 0
    depth_norm[depth_norm > 1] = 1
    byte_img = np.zeros_like(depth,np.uint8)
    byte_img[:] = depth_norm[:] * 255
    byte_img[~good_mask] = 255;
    io_utils.imwrite(byte_img,"./df_depth.png")

    #plt.interactive(True)
    plt.imshow(depth, cmap=plt.cm.gray)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
