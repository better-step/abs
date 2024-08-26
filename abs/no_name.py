from abs import sampler
import numpy as np
from abs import poisson_disk_downsample



def get_data_parts(parts, num_samples, lambda_func):

    ss = np.empty((0, 3))
    pts = np.empty((0, 3))

    for part in parts:
        for p_index, p in enumerate(part.Topology._topology):
            for face_index, face in enumerate(p.faces):
                surface_index = face['surface']
                surface = part.Geometry._surfaces[surface_index]
                uv_points, pt = sampler.random_sample(surface, 0.8, 10, 5 * num_samples)
                s = lambda_func(part, surface, uv_points)
                index = part.filter_outside_points(face_index, uv_points)
                ss = np.concatenate((ss, s[index, :]), axis=0)
                pts = np.concatenate((pts, pt[index, :]), axis=0)

    indices = poisson_disk_downsample(pts, num_samples)
    return pts[indices], ss[indices]


def get_data_test(shape, num_samples, lambda_func):
    ss = np.empty((0, 3))
    pts = np.empty((0, 3))

    for part_index, part in enumerate(shape.Topology._topology):
        for face_index, face in enumerate(part.faces):
            surface_index = face['surface']
            surface = shape.Geometry._surfaces[surface_index]
            uv_points, pt = surface_sampler.random_parametric_sample(surface, 0.8, 10, 5 * num_samples)
            s = lambda_func(shape, surface, uv_points)
            index = shape.filter_outside_points(face_index, uv_points)
            ss = np.concatenate((ss, s[index, :]), axis=0)
            pts = np.concatenate((pts, pt[index, :]), axis=0)

    # idx = poisson_disk_downsample(pts, num_samples)
    # ss, pts = ss[idx, :], pts[idx, :]
    return ss, pts

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

                    uv_points, pt = surface_sampler.random_parametric_sample(surface, 0.8, 10, 5*num_samples)

                    s = lambda_func(shape, surface, uv_points)

                    index = shape.filter_outside_points(face_index, uv_points)

                    ss = np.concatenate((ss, s[index, :]), axis=0)
                    pts = np.concatenate((pts, pt[index, :]), axis=0)



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
    #
    # idx = poisson_disk_downsample(pts, num_samples)
    # return ss[idx,:], pts[idx,:]
    return pts, ss
