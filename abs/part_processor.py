from abs import sampler
import numpy as np
from abs import poisson_disk_downsample
import point_cloud_utils as pcu



def estimate_total_surface_area(part):

    #TODO each face has only ONE surface?
    total_area = 0
    for face in part.Solid.faces:
        surface = face._surface
        total_area += surface.area()

    return total_area



def process_part(part, num_samples, lambda_func, points_ratio=5):

    # Initial setup for sampling
    initial_num_points = points_ratio * num_samples
    num_points = initial_num_points
    total_area = estimate_total_surface_area(part)

    # Sampling loop for surfaces
    while True:

        current_pts, current_ss = [], []

        # Iterate through faces in topology

        for face in part.Solid.faces:

            surface = face._surface
            current_surface_num_points = int(np.ceil((surface.area() / total_area) * num_points))

            # Sample points
            uv_points, pt = sampler.random_sample(surface, current_surface_num_points, 2)

            s = lambda_func(part, face, uv_points)

            if s is not None:

                if len(s) != len(pt):
                    s = np.full((pt.shape[0], pt.shape[1]), s)

                elif s.shape[1] != pt.shape[1]:
                    s = np.tile(s, (1, pt.shape[1]))

                index = part.filter_outside_points(face, uv_points)
                current_pts.append(pt[index, :])
                current_ss.append(s[index, :])




        pts = np.concatenate(current_pts, axis=0)
        ss = np.concatenate(current_ss, axis=0)

        if len(pts) >= initial_num_points:
            break
        else:
            num_points = np.ceil(num_points *  initial_num_points / len(pts) * 1.2)




    # # sample points for 3d curves
    for edge in part.Solid.edges:

        curve = edge._3dcurve

        # what was the purpose of this?
        if curve is None:
            continue
        # continue

        # Sample points
        uv_points, pt = sampler.random_sample(curve, num_samples, 0, num_points)

        s = lambda_func(part, edge, uv_points)

        if s is not None:
            if len(s) != len(pt):
                s = np.full((pt.shape[0], pt.shape[1]), s)

            elif s.shape[1] != pt.shape[1]:
                s = np.tile(s, (1, pt.shape[1]))

            pts = np.concatenate((pts, pt), axis=0)
            ss = np.concatenate((ss, s), axis=0)

    indices = poisson_disk_downsample(pts, num_samples)
    #indices = pcu.downsample_point_cloud_poisson_disk(pts, 0, target_num_samples=num_samples)


    if len(indices) < num_samples:
        remaining_pts = [i for i in range(len(pts)) if i not in indices]
        additional_indices = np.random.choice(remaining_pts, num_samples - len(indices), replace=False)
        indices = np.concatenate([indices, additional_indices])
    elif len(indices) > num_samples:
        indices = np.random.choice(indices, num_samples, replace=False)

    return pts[indices], ss[indices]


def get_parts(parts, num_samples, lambda_func):

    # Initialize empty lists to hold part arrays -
    pts_list = []
    ss_list = []

    for part in parts:

        # Process each part to get points and ss values
        pts, ss = process_part(part, num_samples, lambda_func)

        # Convert points and ss to NumPy arrays and append them to the list
        pts_list.append(np.array(pts))
        ss_list.append(np.array(ss))

    # Return lists of arrays
    return pts_list, ss_list


