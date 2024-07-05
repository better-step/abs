from abs.sampling import curve_sampler
from abs.sampling import surface_sampler


def get_data(shape, num_samples, lambda_func):
    ss = []
    for curve2d in shape.Geometry._curves2d:
        uv_points, _ = curve_sampler.uniform_parametric_sample(curve2d, 0.01)
        s = lambda_func(shape, curve2d, uv_points)
        ss.append(s)

    for curve3d in shape.Geometry._curves3d:
        uv_points, _ = curve_sampler.uniform_parametric_sample(curve3d, 0.01)
        s = lambda_func(shape, curve3d, uv_points)
        ss.append(s)

    for surface in shape.Geometry._surfaces:
        uv_points, _ = surface_sampler.uniform_parametric_sample(surface, 0.01)
        s = lambda_func(shape, surface, uv_points)
        ss.append(s)




