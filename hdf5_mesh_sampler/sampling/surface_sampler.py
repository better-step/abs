from .sampler import Sampler
import numpy as np


class SurfaceSampler(Sampler):
    """
    Sampler for surface entities.
    """

    def __init__(self, spacing, method='poisson_disk'):
        """
        Initialize the surface sampler.

        Args:
        spacing (float): The spacing parameter for sampling.
        method (str): The method to use for sampling, e.g., 'poisson_disk'.
        """
        if spacing <= 0:
            raise ValueError("Spacing must be a positive number.")

        super().__init__(spacing)
        self.method = method

        if self.method == 'poisson_disk':
            self.k = 100  # Default value for k, specific to Poisson disk sampling

    def set_poisson_disk_k(self, k):
        """
        Set the 'k' value for Poisson disk sampling.

        Args:
            k (int): The number of new points to generate around each point.
        """
        if self.method != 'poisson_disk':
            raise ValueError("The 'k' value can only be set for Poisson disk sampling.")

        if not isinstance(k, int) or k <= 0:
            raise ValueError("The 'k' value must be a positive integer.")

        self.k = k


    def sample(self, surface):
        """
        Sample points on a surface based on the specified method.

        Args:
        surface: The surface entity to be sampled.

        Returns:
        An array of sampled points on the surface.
        """
        # Implementation of the sampling method
        # This can be expanded to include various sampling strategies
        if self.method == 'poisson_disk':
            return self._poisson_disk_sampling(surface)

        elif self.method == 'uniform':
            return self._uniform_sampling(surface)

        elif self.method == 'random':
            return self._random_sampling(surface)

        else:
            raise ValueError(f"Invalid sampling method: {self.method}")

    def _uniform_sampling(self, surface):
        u_range, v_range = surface._trim_domain
        num_u_samples = max(int(abs(u_range[1] - u_range[0]) / self.spacing), 1)
        num_v_samples = max(int(abs(v_range[1] - v_range[0]) / self.spacing), 1)
        u = np.linspace(u_range[0], u_range[1], num_u_samples)
        v = np.linspace(v_range[0], v_range[1], num_v_samples)
        u, v = np.meshgrid(u, v)
        uv_values = np.column_stack((u.ravel(), v.ravel()))
        return uv_values

    def _random_sampling(self, surface):
        u_range, v_range = surface._trim_domain
        num_u_samples = max(int(abs(u_range[1] - u_range[0]) / self.spacing), 1)
        num_v_samples = max(int(abs(v_range[1] - v_range[0]) / self.spacing), 1)
        u = np.random.uniform(u_range[0], u_range[1], num_u_samples)
        v = np.random.uniform(v_range[0], v_range[1], num_v_samples)
        uv_values = np.column_stack((u, v))
        return uv_values


    def _calculate_cell_size(self, surface):
        # Extract the range of u and v from the trimming domain
        u_range, v_range = surface._trim_domain
        u_size = u_range[1] - u_range[0]
        v_size = v_range[1] - v_range[0]

        # Calculate the number of cells along each axis
        u_cells = int(u_size / self.spacing)
        v_cells = int(v_size / self.spacing)

        # Ensure there is at least one cell along each axis
        u_cells = max(u_cells, 1)
        v_cells = max(v_cells, 1)

        # Calculate cell size for each axis
        cell_size_u = u_size / u_cells
        cell_size_v = v_size / v_cells

        return max(cell_size_u, cell_size_v)

    def _get_grid_shape_old(self, surface, cell_size):
        grid_extent = (surface._trim_domain[:, 1] - surface._trim_domain[:, 0]) / cell_size
        grid_shape = np.ceil(grid_extent).astype(int)
        return grid_shape + 1

    def _get_grid_shape(self, surface, cell_size):
        # Calculate the extent of the grid based on the trimming domain and the cell size
        u_range, v_range = surface._trim_domain
        u_size = u_range[1] - u_range[0]
        v_size = v_range[1] - v_range[0]

        # Calculate the number of cells along each axis
        u_cells = max(int(np.ceil(u_size / cell_size)), 1)
        v_cells = max(int(np.ceil(v_size / cell_size)), 1)

        return np.array([u_cells, v_cells])

    # Add a buffer to avoid negative indices

    def _get_random_point_in_domain(self, surface):
        u_range, v_range = surface._trim_domain
        u = np.random.uniform(*u_range)
        v = np.random.uniform(*v_range)
        return np.array([u, v])

    def _poisson_disk_sampling(self, surface):
        cell_size = self._calculate_cell_size(surface)
        grid_shape = self._get_grid_shape(surface, cell_size)
        grid = np.full(grid_shape, -1, dtype=int)

        active_list = []
        initial_point = self._get_random_point_in_domain(surface)
        grid_index = (initial_point // cell_size).astype(int)
        grid[tuple(grid_index)] = 0
        sample_points = [initial_point]
        active_list.append(initial_point)

        while active_list:
            point_index = np.random.choice(len(active_list))
            current_point = active_list[point_index]
            found = False

            for _ in range(self.k):
                new_point = self._generate_random_point_around(current_point, self.spacing)
                if self._is_valid(new_point, surface, grid, cell_size, sample_points):
                    sample_points.append(new_point)
                    active_list.append(new_point)
                    grid_index = (new_point // cell_size).astype(int)
                    grid[tuple(grid_index)] = len(sample_points) - 1
                    found = True
                    break

            if not found:
                active_list.pop(point_index)

        return np.array(sample_points)

    def _generate_random_point_around(self, point, min_dist):
        r = np.random.uniform(min_dist, 2 * min_dist)
        theta = np.random.uniform(0, 2 * np.pi)
        new_point = point + r * np.array([np.cos(theta), np.sin(theta)])
        return new_point

    def _is_valid(self, point, surface, grid, cell_size, sample_points):
        if not self._is_within_domain(point, surface):
            return False

        grid_index = np.clip((point // cell_size).astype(int), 0, grid.shape - 1)
        for i in range(max(grid_index[0] - 2, 0), min(grid_index[0] + 3, grid.shape[0])):
            for j in range(max(grid_index[1] - 2, 0), min(grid_index[1] + 3, grid.shape[1])):
                if grid[i, j] != -1 and np.linalg.norm(point - sample_points[grid[i, j]]) < self.spacing:
                    return False

        return True

    def _is_within_domain(self, point, surface):
        u_min, u_max = surface._trim_domain[0]
        v_min, v_max = surface._trim_domain[1]
        return u_min <= point[0] <= u_max and v_min <= point[1] <= v_max
