static bool unit_cube_intersection_points(float4 origin, float4 dir, float* t_in, float* t_out)
{
    *t_in = NAN;
    *t_out = NAN;
    // enter/exit through min x plane?
    float t = -origin.x/dir.x;
    float4 pt = origin + dir*t;
    if (all(pt.yz >= 0) && all(pt.yz < 1)) {
        if (dir.x > 0) {
            *t_in = t;
        }
        if (dir.x < 0) {
            *t_out = t;
        }
    }
    // enter/exit through the max x plane?
    t = (1.0f - origin.x)/dir.x;
    pt = origin + dir*t;
    if (all(pt.yz >= 0) && all(pt.yz < 1)) {
        if (dir.x > 0) {
            *t_out = t;
        }
        if (dir.x < 0) {
            *t_in = t;
        }
    }
    // enter/exit through min y plane?
    t = -origin.y/dir.y;
    pt = origin + dir*t;
    if (all(pt.xz >= 0) && all(pt.xz < 1)) {
        if (dir.y > 0) {
            *t_in = t;
        }
        if (dir.y < 0) {
            *t_out = t;
        }
    }
    // enter/exit through the max y plane?
    t = (1.0f - origin.y)/dir.y;
    pt = origin + dir*t;
    if (all(pt.xz >= 0) && all(pt.xz < 1)) {
        if (dir.y > 0) {
            *t_out = t;
        }
        if (dir.y < 0) {
            *t_in = t;
        }
    }
    // enter/exit through min z plane?
    t = -origin.z/dir.z;
    pt = origin + dir*t;
    if (all(pt.xy >= 0) && all(pt.xy < 1)) {
        if (dir.z > 0) {
            *t_in = t;
        }
        if (dir.z < 0) {
            *t_out = t;
        }
    }
    // enter/exit through the max z plane?
    t = (1.0f - origin.z)/dir.z;
    pt = origin + dir*t;
    if (all(pt.xy >= 0) && all(pt.xy < 1)) {
        if (dir.z > 0) {
            *t_out = t;
        }
        if (dir.z < 0) {
            *t_in = t;
        }
    }
    return (isfinite(*t_in) && isfinite(*t_out));
}

static bool get_value_nearest(__global const float* values, int dims, float4 pos, float* value)
{
    int4 posi = convert_int4_rte(pos);
    posi.w = 0;
    if (any(posi >= dims) || any(posi < 0)) {
        *value = NAN;
        return false;
    }
    int idx = posi.z * dims*dims + posi.y*dims + posi.x;
    *value = values[idx];
    return true;
}

static bool get_value_trilin(__global const float* values, int dims, float4 pos, float* value)
{
    int4 posi0 = convert_int4_rtn(pos);
    posi0 = max(posi0,(int4)(0,0,0,0));
    posi0 = min(posi0,(int4)(dims-2,dims-2,dims-2,0));
    int4 posi1 = posi0 + 1;
    posi1.w = 0;

    float4 d1 = pos - convert_float4(posi0);
    float4 d0 = convert_float4(posi1) - pos;
    //float4 d0 = (float4)(1.0f) - d1;

    int idx0 = posi0.z * dims*dims + posi0.y*dims + posi0.x;

    float c00 = values[idx0]*d0.x + values[idx0+1]*d1.x;
    // advance idx to next y
    float c10 = values[idx0 + dims]*d0.x + values[idx0+dims+1]*d1.x;
    // advance idx to next z
    float c01 = values[idx0 + dims*dims]*d0.x + values[idx0+dims*dims+1]*d1.x;
    // advance idx to next z and y
    float c11 = values[idx0 + dims*dims + dims]*d0.x + values[idx0+dims*dims + dims + 1]*d1.x;

    float c0 = c00*d0.y + c10*d1.y;
    float c1 = c01*d0.y + c11*d1.y;

    float c = c0*d0.z + c1*d1.z;
    *value = c;
    return true;
}

// assumes df is defined on the unit cube
static bool get_depth(__global const float* df, int dims, float4 ray_origin, float4 ray_dir, float* intersection_t)
{
    float t_in, t_out;
    if (!unit_cube_intersection_points(ray_origin, ray_dir, &t_in, &t_out)) {
        // no intersection with unit cube
        *intersection_t = NAN;
        return false;
    }
    // step through the cell from t_in to t_out
    const float dt = 0.01f;
    float prev_t = t_in;
    float prev_val = -1.0f;
    for (float t=t_in; t<t_out; t+=dt) {
        float4 pt = ray_origin + ray_dir*t;
        float4 indexf = pt * (dims-1); // unit cube
        float value;
        //if(!get_value_nearest(df, dims, indexf, &value)) {
        if(!get_value_trilin(df, dims, indexf, &value)) {
            // shouldnt happen
            *intersection_t = NAN;
            return false;
        }
        // check for zero crossing (+ -> -)
        if ((prev_val > 0) && (value <= 0)) {
            // linearly interpolate between prev and current t
            float curr_weight = prev_val / (prev_val - value);
            *intersection_t =  curr_weight*t + (1.0f - curr_weight)*prev_t;
            return true;
        }
        prev_t = t;
        prev_val = value;
    }
    // No surface intersection
    *intersection_t = NAN;
    return false;
}

__kernel void render_distance_field(__global const float* df, __global const float4* origin, int dims, float volume_width,
                                    __global const float4* ray_origins, __global const float4* ray_dirs, __global const int* img_dims,
                                    __global float* depth_img)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if ((r >= img_dims[0]) || (c >= img_dims[1])) {
        return;
    }
    int idx = r*img_dims[1] + c;

    // convert volume to unit cube
    float4 ray_origin_u = (ray_origins[idx] - *origin) / volume_width;
    float4 ray_dir = ray_dirs[idx];

    float intersection_t;
    if (get_depth(df, dims, ray_origin_u, ray_dir, &intersection_t)) {
        depth_img[idx] = intersection_t * volume_width;
    }
    else {
        depth_img[idx] = intersection_t;
    }
    return;
}

#if 0
__kernel void render_distance_field_affine(__global const float* df,
                                           __global const float4* origin, int dims, float volume_width,
                                           __global const float* cam_P, __global const float4* view_dir, float view_dist,
                                           __global const int* img_dims, __global float* depth_img)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if ((r >= img_dims[0]) || (c >= img_dims[1])) {
        return;
    }
    int idx = r*img_dims[1] + c;

    // convert volume to unit cube
    float4 ray_origin_u = (ray_origins[idx] - *origin) / volume_width;
    float4 ray_dir = ray_dirs[idx];

    float intersection_t;
    if (get_depth(df, dims, ray_origin_u, ray_dir, &intersection_t)) {
        depth_img[idx] = intersection_t;
    }
    else {
        depth_img[idx] = NAN;
    }
    return;
}
#endif
