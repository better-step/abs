"""
Functions for processing Shape parts: sampling points and computing normals.
"""
import numpy as np
from abs import sampler
from abspy import poisson_disk_downsample
# from abspy import poisson_grid_downsample

def _slice(res, indices, n):
    if isinstance(res, list) or isinstance(res, tuple):
        assert(len(res) == n)
        return [res[i] for i in range(n) if indices[i]]

    assert (isinstance(res, np.ndarray))
    return res[indices]

def estimate_total_surface_area(part):
    """Estimate the total surface area of all faces in a Shape part."""
    total_area = 0.0
    for face in part.faces:
        total_area += face.get_area()
    return total_area

def estimate_total_curve_length(part):
    """Estimate the total curve length of all edges in a Shape part."""
    total_length = 0.0
    for edge in part.edges:
        total_length += edge.get_length()
    return total_length


def process_part(part,
                 num_samples,
                 face_func,
                 edge_func,
                 points_ratio,
                 apply_transform,
                 uniform_sample,
                 use_poisson,
                 force_num_points,
                 sample_num_tolerance):
    """
    FIXME
    Sample points on all surfaces and curves of a shape part according to specified number.
    Uses an iterative strategy with oversampling (points_ratio) and Poisson disk downsampling to refine points.
    The lambda_func is a function(part, topo_entity, param_points) that returns associated values (e.g., normals).
    """
    initial_num_points = points_ratio * num_samples
    num_points = initial_num_points
    total_area = estimate_total_surface_area(part)
    total_length = estimate_total_curve_length(part)
    n_face_points = 0
    while True:
        current_pts = []
        current_ss = []

        if face_func is not None:
            for face in part.faces:
                n_surf = int(np.ceil((face.get_area() / total_area) * num_points))
                if uniform_sample:
                    uv_points, pt = sampler.uniform_sample(face, n_surf, min_pts=2)
                else:
                    uv_points, pt = sampler.random_sample(face, n_surf, min_pts=2)
                s = face_func(face, uv_points)
                if s is not None:
                    if apply_transform:
                        # Transform points to surface's local coordinate system
                        R = np.linalg.inv(face.surface.transform[:, :3])
                        t = face.surface.transform[:, 3]
                        pt = (pt - t) @ R.T
                    # Filter out points outside trimming loops
                    index = face.filter_outside_points(uv_points)
                    current_pts.append(pt[index, :])
                    if isinstance(s, list) or isinstance(s, tuple):
                        current_ss += _slice(s, index, pt.shape[0])
                    else:
                        current_ss.append(_slice(s, index, pt.shape[0]))

        n_face_points = np.concatenate(current_pts, axis=0).shape[0]

        if edge_func is not None:
            for edge in part.edges:
                if edge.curve3d is None or edge.curve3d.shape_name == "Other":
                    continue
                n_edge = int(np.ceil((edge.get_length() / total_length) * num_points))
                if uniform_sample:
                    uv_points, pt = sampler.uniform_sample(edge, n_edge, min_pts=2)
                else:
                    uv_points, pt = sampler.random_sample(edge, n_edge, min_pts=2)
                s = edge_func(edge, uv_points)
                if s is not None:
                    if apply_transform:
                        R = np.linalg.inv(edge.curve3d.transform[:, :3])
                        t = edge.curve3d.transform[:, 3]
                        pt = (pt - t) @ R.T
                    # Edges have no trimming curves; include all sampled points
                    index = np.ones(uv_points.shape[0], dtype=bool)
                    current_pts.append(pt[index, :])
                    if isinstance(s, list) or isinstance(s, tuple):
                        current_ss += _slice(s, index, pt.shape[0])
                    else:
                        current_ss.append(_slice(s, index, pt.shape[0]))
        # Combine all collected points and associated data
        if len(current_pts) == 0:
            pts = np.zeros((0, 3))
            ss = []
        else:
            pts = np.concatenate(current_pts, axis=0)
            try:
                ss = np.concatenate(current_ss, axis=0)
            except ValueError:
                ss = current_ss
        # Stop when enough points collected or nothing collected
        if not force_num_points or pts.shape[0] >= initial_num_points or pts.shape[0] == 0:
            break
        else:
            # Increase num_points for next iteration (upscale by factor to approach initial count)
            num_points = np.ceil(num_points * initial_num_points / max(len(pts), 1) * 1.2)

    # If no points at all, return empty structures
    if pts.shape[0] == 0:
        return pts, pts, pts, pts
    # Poisson disk downsample to exactly num_samples points

    if use_poisson:
        indices = poisson_disk_downsample(pts, num_samples, sample_num_tolerance=sample_num_tolerance)
        # indices = poisson_grid_downsample(pts, num_samples)
    else:
        indices = np.arange(pts.shape[0])

    if force_num_points:
        if len(indices) < num_samples:
            remaining_idx = [i for i in range(len(pts)) if i not in indices]
            additional_indices = np.random.choice(remaining_idx, num_samples - len(indices), replace=False)
            indices = np.concatenate([indices, additional_indices])
        elif len(indices) > num_samples:
            indices = np.random.choice(indices, num_samples, replace=False)

    indices = np.sort(indices)

    face_indices = indices[indices < n_face_points]
    edge_indices = indices[indices >= n_face_points]

    if isinstance(ss, list):
        return pts[face_indices], [ss[i] for i in face_indices], pts[edge_indices], [ss[i] for i in edge_indices]
    else:
        assert(isinstance(ss, np.ndarray))
        return pts[face_indices], ss[face_indices], pts[edge_indices], ss[edge_indices]


def sample_parts(parts,
                 num_samples,
                 face_func=None,
                 edge_func=None,
                 points_ratio=5,
                 apply_transform=True,
                 uniform_sample=False,
                 use_poisson=True,
                 force_num_points=True,
                 sample_num_tolerance=0.04):

    """
        Process a list of parts by sampling each part and returning lists of points and values.
    """
    pts_listf = []
    pts_liste = []
    ss_listf = []
    ss_liste = []
    for part in parts:
        ptsf, ssf, ptse, sse = process_part(part, num_samples, face_func, edge_func, points_ratio=points_ratio,apply_transform=apply_transform, uniform_sample=uniform_sample, use_poisson=use_poisson, force_num_points=force_num_points,sample_num_tolerance=sample_num_tolerance)
        pts_listf.append(ptsf)
        pts_liste.append(ptse)
        ss_listf.append(ssf)
        ss_liste.append(sse)
    return pts_listf, ss_listf, pts_liste, ss_liste

