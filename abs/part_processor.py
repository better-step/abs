from abs import sampler
import numpy as np
from abs import poisson_disk_downsample


def estimate_total_surface_area(part):

    total_area = 0
    for face in part.Solid.faces:
        total_area += face.get_area()

    return total_area


def estimate_total_curve_length(part):

    total_length = 0
    for edge in part.Solid.edges:
        total_length += edge.get_length()

    return total_length


def process_part(part, num_samples, lambda_func, points_ratio=5):

    initial_num_points = points_ratio * num_samples
    num_points = initial_num_points
    total_area = estimate_total_surface_area(part)
    total_length = estimate_total_curve_length(part)

    while True:
        current_pts, current_ss = [], []

        # Sampling loop for surfaces
        for face in part.Solid.faces:

            current_surface_num_points = int(np.ceil((face.get_area() / total_area) * num_points))

            # Sample points
            uv_points, pt = sampler.random_sample(face, current_surface_num_points, 2)

            s = lambda_func(part, face, uv_points)


            if s is not None:

                R = np.linalg.inv(face.surface.transform[:, :3])
                t = face.surface.transform[:, 3]
                pt = (pt - t) @ R.T

                index = face.filter_outside_points(uv_points)
                current_pts.append(pt[index, :])

                if type(s) == list:

                    for i in range(len(s)):

                        if len(s[i]) != len(pt):
                            s[i] = np.full((pt.shape[0], pt.shape[1]), s[i])

                        elif s[i].shape[1] != pt.shape[1]:
                            s[i] = np.tile(s[i], (1, pt.shape[1]))

                        if len(current_ss) == 0:
                            current_ss = [[] for _ in range(len(s))]

                        current_ss[i].extend(s[i][index, :])

                else:
                    if len(s) != len(pt):
                        s = np.full((pt.shape[0], pt.shape[1]), s)

                    elif s.shape[1] != pt.shape[1]:
                        s = np.tile(s, (1, pt.shape[1]))

                    current_ss.append(s[index, :])



        # sample points for edges
        for edge in part.Solid.edges:
            if edge.curve3d is None:
                continue

            current_edge_num_points = int(np.ceil((edge.get_length() / total_length) * num_points))

            # Sample points
            uv_points, pt = sampler.random_sample(edge, current_edge_num_points, 2)

            s = lambda_func(part, edge, uv_points)

            if s is not None:

                R = np.linalg.inv(edge.curve3d.transform[:, :3])
                t = edge.curve3d.transform[:, 3]
                pt = (pt - t) @ R.T

                index = face.filter_outside_points(uv_points)
                current_pts.append(pt[index, :])

                if type(s) == list:

                    for i in range(len(s)):

                        if len(s[i]) != len(pt):
                            s[i] = np.full((pt.shape[0], pt.shape[1]), s[i])

                        elif s[i].shape[1] != pt.shape[1]:
                            s[i] = np.tile(s[i], (1, pt.shape[1]))

                        if len(current_ss) == 0:
                            current_ss = [[] for _ in range(len(s))]

                        current_ss[i].extend(s[i][index, :])

                else:
                    if len(s) != len(pt):
                        s = np.full((pt.shape[0], pt.shape[1]), s)

                    elif s.shape[1] != pt.shape[1]:
                        s = np.tile(s, (1, pt.shape[1]))

                    current_ss.append(s[index, :])


        if len(current_pts) == 0:
            pts = np.zeros((0,3))
        else:
            pts = np.concatenate(current_pts, axis=0)
            if type(current_ss[0]) == list:
                ss = current_ss.copy()
            else:
                ss = np.concatenate(current_ss, axis=0)

        if len(pts) >= initial_num_points or len(pts) == 0:
            break
        else:
            num_points = np.ceil(num_points *  initial_num_points / len(pts) * 1.2)


    if len(pts) ==0:
        return pts, pts

    indices = poisson_disk_downsample(pts, num_samples)


    if len(indices) < num_samples:
        remaining_pts = [i for i in range(len(pts)) if i not in indices]
        additional_indices = np.random.choice(remaining_pts, num_samples - len(indices), replace=False)
        indices = np.concatenate([indices, additional_indices])
    elif len(indices) > num_samples:
        indices = np.random.choice(indices, num_samples, replace=False)

    if type(ss) == list:
        new_ss = [[sublist[i] for i in indices] for sublist in ss]
        return pts[indices], new_ss
    else:
        return pts[indices], ss[indices]


def sample_parts(parts, num_samples, lambda_func, points_ratio=5):

    pts_list = []
    ss_list = []

    for part in parts:
        pts, ss = process_part(part, num_samples, lambda_func, points_ratio)
        pts_list.append(np.array(pts))
        if type(ss) == list:
            ss_list.extend([np.array(sublist) for sublist in ss])
        else:
            ss_list.append(np.array(ss))

    return pts_list, ss_list


#current_ss[i].extend(s[i][index, :])
