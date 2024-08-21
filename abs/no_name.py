from abs.sampling import curve_sampler
from abs.sampling import surface_sampler
import numpy as np
from abs import poisson_disk_downsample



def get_data_geo(shape, num_samples, lambda_func):

    ss = np.empty((0, 3))
    pts = np.empty((0, 3))

    sample2d_flag = True
    sample3d_flag = False

    # update the flags based on the lambda function later!
    # also need to hae a way of switching between random and uniform sampling ?!!
    # need to decide on the spacing !

    for part_index, part in enumerate(shape.Geometry):

        if sample2d_flag:
            for curve2d in part._curves2d:
                uv_points, pt = curve_sampler.random_parametric_sample(curve2d, 0.1)
                pts.append(pt)

    if sample3d_flag:
        for curve3d in shape.Geometry._curves3d:
            uv_points, pt = curve_sampler.random_parametric_sample(curve3d, 0.1)
            pts.append(pt)

    for surface in shape.Geometry._surfaces:
        uv_points, pt = surface_sampler.random_parametric_sample(surface, 0.5, 10, 5 * num_samples)
        s = lambda_func(shape, surface, uv_points)
        index = shape.filter_outside_points(face_index, uv_points)
        ss = np.concatenate((ss, s), axis=0)
        pts.append(pt)


def get_data(shape, num_samples, lambda_func):
    # for tracking normals and sample points
    ss = np.empty((0, 3))
    pts = np.empty((0, 3))

    for part_index, part in enumerate(shape.Topology._topology):
        for solid_index, solid in enumerate(part.solids):
            for shell_index in solid['shells']:
                shell = part.shells[shell_index]
                for (face_index, _) in shell['faces']:
                    face = part.faces[face_index]
                    surface_index = face['surface']

                    surface = shape.Geometry._surfaces[surface_index]

                    uv_points, pt = surface_sampler.random_parametric_sample(surface, 0.5, 10, 5*num_samples)

                    s = lambda_func(shape, surface, uv_points)

                    index = shape.filter_outside_points(face_index, uv_points)

                    ss = np.concatenate((ss, s[index, :]), axis=0)
                    pts = np.concatenate((pts, pt[index, :]), axis=0)
        # break


    idx = poisson_disk_downsample(pts, num_samples)
    ss, pts = ss[idx, :], pts[idx, :]

    # for part_index, part in enumerate(shape.Topology._topology):
    #     for edge in part.edges:
    #         curve_index = edge['3dcurve']
    #         curve = shape.Geometry._curves3d[curve_index]
    #
    #         # Sample points along the curve
    #         uv_points, pt = curve_sampler.random_parametric_sample(curve, num_samples)
    #         pts = np.concatenate((pts, pt), axis=0)

    #idx = poisson_disk_downsample(pts, num_samples)
    # return ss[idx,:], pts[idx,:]
    return ss, pts
