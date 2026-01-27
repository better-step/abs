"""
Functions for processing Shape parts: sampling points and computing normals.
"""

import numpy as np
from . import sampler
from abspy import poisson_disk_downsample
# from abspy import poisson_grid_downsample


def estimate_total_surface_area(part):
    """Estimate the total surface area of all faces in a Shape part."""
    total_area = 0.0
    for face in part.Solid.faces:
        total_area += face.get_area()
    return total_area

def estimate_total_curve_length(part):
    """Estimate the total curve length of all edges in a Shape part."""
    total_length = 0.0
    for edge in part.Solid.edges:
        total_length += edge.get_length()
    return total_length


def process_part(part, num_samples, lambda_func, points_ratio, apply_transform):
    """
    Sample points on all surfaces and curves of a shape part according to specified number.
    Uses an iterative strategy with oversampling (points_ratio) and Poisson disk downsampling to refine points.
    The lambda_func is a function(part, topo_entity, param_points) that returns associated values (e.g., normals).
    """
    initial_num_points = points_ratio * num_samples
    num_points = initial_num_points
    total_area = estimate_total_surface_area(part)
    total_length = estimate_total_curve_length(part)
    pts = np.zeros((0, 3))
    ss = []
    while True:
        current_pts = []
        current_ss = []
        # Sample points on each face (surface)
        for face in part.Solid.faces:
            n_surf = int(np.ceil((face.get_area() / total_area) * num_points))
            uv_points, pt = sampler.random_sample(face, n_surf, min_pts=2)
            s = lambda_func(part, face, uv_points)
            if s is not None:
                if apply_transform:
                    # Transform points to surface's local coordinate system
                    R = np.linalg.inv(face.surface.transform[:, :3])
                    t = face.surface.transform[:, 3]
                    pt = (pt - t) @ R.T
                # Filter out points outside trimming loops
                index = face.filter_outside_points(uv_points)
                current_pts.append(pt[index, :])
                if isinstance(s, list):
                    # If lambda_func returns multiple arrays
                    for i, val in enumerate(s):
                        if len(val) != len(pt):
                            val = np.full((pt.shape[0], pt.shape[1]), val)
                        elif val.shape[1] != pt.shape[1]:
                            val = np.tile(val, (1, pt.shape[1]))
                        if len(current_ss) == 0:
                            current_ss = [[] for _ in range(len(s))]
                        current_ss[i].extend(val[index, :])
                else:
                    if len(s) != len(pt):
                        s = np.full((pt.shape[0], pt.shape[1]), s)
                    elif s.shape[1] != pt.shape[1]:
                        s = np.tile(s, (1, pt.shape[1]))
                    current_ss.append(s[index, :])
        # Sample points on each edge (curve)
        for edge in part.Solid.edges:
            if edge.curve3d is None:
                continue
            n_edge = int(np.ceil((edge.get_length() / total_length) * num_points))
            uv_points, pt = sampler.random_sample(edge, n_edge, min_pts=2)
            s = lambda_func(part, edge, uv_points)
            if s is not None:
                if apply_transform:
                    R = np.linalg.inv(edge.curve3d.transform[:, :3])
                    t = edge.curve3d.transform[:, 3]
                    pt = (pt - t) @ R.T
                # Edges have no trimming curves; include all sampled points
                index = np.ones(uv_points.shape[0], dtype=bool)
                current_pts.append(pt[index, :])
                if isinstance(s, list):
                    for i, val in enumerate(s):
                        if len(val) != len(pt):
                            val = np.full((pt.shape[0], pt.shape[1]), val)
                        elif val.shape[1] != pt.shape[1]:
                            val = np.tile(val, (1, pt.shape[1]))
                        if len(current_ss) == 0:
                            current_ss = [[] for _ in range(len(s))]
                        current_ss[i].extend(val[index, :])
                else:
                    if len(s) != len(pt):
                        s = np.full((pt.shape[0], pt.shape[1]), s)
                    elif s.shape[1] != pt.shape[1]:
                        s = np.tile(s, (1, pt.shape[1]))
                    current_ss.append(s[index, :])
        # Combine all collected points and associated data
        if len(current_pts) == 0:
            pts = np.zeros((0, 3))
            ss = []
        else:
            pts = np.concatenate(current_pts, axis=0)
            if isinstance(current_ss, list) and len(current_ss) > 0 and isinstance(current_ss[0], list):
                ss = current_ss.copy()
            else:
                ss = np.concatenate(current_ss, axis=0) if len(current_ss) > 0 else []
        # Stop when enough points collected or nothing collected
        if pts.shape[0] >= initial_num_points or pts.shape[0] == 0:
            break
        else:
            # Increase num_points for next iteration (upscale by factor to approach initial count)
            num_points = np.ceil(num_points * initial_num_points / max(len(pts), 1) * 1.2)
    # If no points at all, return empty structures
    if pts.shape[0] == 0:
        return pts, pts
    # Poisson disk downsample to exactly num_samples points
    indices = poisson_disk_downsample(pts, num_samples)
    # indices = poisson_grid_downsample(pts, num_samples)
    if len(indices) < num_samples:
        remaining_idx = [i for i in range(len(pts)) if i not in indices]
        additional_indices = np.random.choice(remaining_idx, num_samples - len(indices), replace=False)
        indices = np.concatenate([indices, additional_indices])
    elif len(indices) > num_samples:
        indices = np.random.choice(indices, num_samples, replace=False)
    if isinstance(ss, list):
        # Multiple associated arrays (e.g., normals in separate array)
        new_ss = [[sublist[i] for i in indices] for sublist in ss]
        return pts[indices], new_ss
    else:
        return pts[indices], ss[indices] if isinstance(ss, np.ndarray) else ss


def sample_parts(parts, num_samples, lambda_func, points_ratio=5, apply_transform=True):
    """Process a list of parts by sampling each part and returning lists of points and values."""
    pts_list = []
    ss_list = []
    for part in parts:
        pts, ss = process_part(part, num_samples, lambda_func, points_ratio, apply_transform)
        pts_list.append(np.array(pts))
        if isinstance(ss, list):
            ss_list.extend([np.array(sublist) for sublist in ss])
        else:
            ss_list.append(np.array(ss))
    return pts_list, ss_list

