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
        super().__init__(spacing)
        self.method = method

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

    def _calculate_cell_size(self, surface):
        # Calculate an appropriate cell size based on the surface properties
        # and desired spacing. This implementation will vary depending on the
        # surface's characteristics.
        # For instance, you might use the surface's bounding box dimensions.
        bbox = surface._trim_domain
        bbox_diagonal = np.linalg.norm(bbox[1] - bbox[0])
        return bbox_diagonal / np.sqrt(self.spacing)

    def _poisson_disk_sampling(self, surface):
        sample_points = []
        grid, active = {}, {}

        # Initialize the sampling process
        first_point = self._get_random_point_on_surface(surface)
        sample_points, grid, active = self._insert_point(sample_points, first_point, grid, active, surface)

        # Continue sampling until no active points remain
        while active:
            point, index = self._get_random_point(active)
            new_points = self._get_random_points_around(point, surface)
            for new_point in new_points:
                if self._is_valid_point(new_point, grid, surface):
                    sample_points, grid, active = self._insert_point(sample_points, new_point, grid, active, surface)

            if not new_points:
                active.pop(index)

        return np.array(sample_points)

    def _get_random_point_on_surface(self, surface):
        u_range = surface._trim_domain[:, 0]
        v_range = surface._trim_domain[:, 1]
        u = np.random.uniform(*u_range)
        v = np.random.uniform(*v_range)
        return np.array([u, v])  # Return parameterized point

    def _insert_point(self, sample_points, point, grid, active, surface):
        sample_points.append(surface.sample(np.array([point])))
        cell_index = self._get_cell_index(point, surface)
        grid[cell_index] = point
        active.append(point)
        return sample_points, grid, active

    def _get_cell_index(self, point, surface):
        # Assuming point is a parameterized (u, v) point on the surface
        return tuple(np.floor(point / self._calculate_cell_size(surface)).astype(int))

    def _get_random_point(self, active):
        """
        Randomly select an active point.

        Args:
        active: List of active points.

        Returns:
        A randomly chosen point and its index.
        """
        random_index = np.random.randint(len(active))
        return active[random_index], random_index

    def _get_random_points_around(self, point, surface):
        """
        Generate new points around the given point on the surface.

        Args:
        point: The point around which to generate new points.
        surface: The surface entity.

        Returns:
        List of new points generated around the given point.
        """
        new_points = []
        for _ in range(self.k):  # 'self.k' is the number of new points to generate
            r = np.random.uniform(self.spacing, 2 * self.spacing)
            theta = np.random.uniform(0, 2 * np.pi)
            new_u = point[0] + r * np.cos(theta)
            new_v = point[1] + r * np.sin(theta)
            if self._is_within_domain([new_u, new_v], surface):
                new_points.append([new_u, new_v])
        return new_points

    def _is_valid_point(self, new_point, grid, surface):
        cell_index = self._get_cell_index(new_point, surface)
        neighbours = self._get_neighbours(cell_index, int(np.ceil(self.spacing / self._calculate_cell_size(surface))))

        for neighbour in neighbours:
            if neighbour in grid and np.linalg.norm(
                surface.sample(np.array([grid[neighbour]])) - surface.sample(np.array([new_point]))) < self.spacing:
                return False


