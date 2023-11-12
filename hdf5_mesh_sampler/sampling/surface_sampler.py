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

    def _calculate_cell_size(self):
        # Calculate an appropriate cell size based on the surface properties
        # and desired spacing. This implementation will vary depending on the
        # surface's characteristics.
        # For instance, you might use the surface's bounding box dimensions.
        bbox = surface.bounding_box()
        bbox_diagonal = np.linalg.norm(bbox[1] - bbox[0])
        return bbox_diagonal / np.sqrt(self.spacing)

    def _poisson_disk_sampling(self, surface):
        """
        Implement Poisson disk sampling specific to surfaces.

        Args:
        surface: The surface entity to be sampled.

        Returns:
        An array of sampled points.
        """
        sample_points = []
        grid, active = {}, []

        # Initialize the sampling process
        first_point = self._get_random_point_on_surface(surface)
        self._insert_point(sample_points, first_point, grid, active, surface)

        # Continue sampling until no active points remain
        while active:
            point, index = self._choose_random_active_point(active)
            new_point = self._generate_new_point_around(point, surface)
            if self._is_valid_point(new_point, surface, grid):
                self._insert_point(sample_points, new_point, grid, active, surface)

        return np.array(sample_points)

    def _insert_point(self, sample_points, point, grid, active, surface):
        """
        Insert a new point into the sample points, grid, and active list.

        Args:
        sample_points: List of sampled points.
        point: New point to be inserted.
        grid: Spatial grid for efficient searching.
        active: List of active points.
        surface: The surface entity.
        """
        sample_points.append(point)
        cell_index = self._get_cell_index(point, self.cell_size)
        grid[cell_index] = point
        active.append(point)

    def _choose_random_active_point(self, active):
        """
        Randomly select an active point.

        Args:
        active: List of active points.

        Returns:
        A randomly chosen point and its index.
        """
        random_index = np.random.randint(len(active))
        return active[random_index], random_index

    def _get_random_point_on_surface(self, surface):
        """
        Get a random point on the surface within its trimming domain.

        Args:
        surface: The surface entity.

        Returns:
        A random point on the surface.
        """
        u_min, v_min, u_max, v_max = surface._trim_domain.reshape(-1)
        random_u = np.random.uniform(u_min, u_max)
        random_v = np.random.uniform(v_min, v_max)
        return surface.sample(np.array([[random_u, random_v]]))[0]

    def _generate_new_point_around(self, point, surface):
        """
        Generate a new point around the given point on the surface.

        Args:
        point: The point around which to generate a new point.
        surface: The surface entity.

        Returns:
        A new point generated around the given point.
        """
        r = np.random.uniform(self.spacing, 2 * self.spacing)
        theta = np.random.uniform(0, 2 * np.pi)
        u, v = surface.parameterize_point(point)
        new_u = u + r * np.cos(theta)
        new_v = v + r * np.sin(theta)
        return surface.sample(np.array([[new_u, new_v]]))[0]

    def _is_valid_point(self, point, surface, grid):
        """
        Check if the generated point is valid (i.e., not too close to existing points).

        Args:
        point: The point to validate.
        surface: The surface entity.
        grid: Spatial grid for efficient searching.

        Returns:
        Boolean indicating if the point is valid.
        """
        cell_index = self._get_cell_index(point, self.cell_size)
        neighbours = self._get_neighbours(cell_index, self.spacing / self.cell_size)

        for neighbour in neighbours:
            if neighbour in grid and np.linalg.norm(grid[neighbour] - point) < self.spacing:
                return False
        return True

    def _get_cell_index(self, point, cell_size):
        """
        Calculate the cell index for a point in the spatial grid.

        Args:
        point: The point for which the cell index is calculated.
        cell_size: Size of each cell in the grid.

        Returns:
        Tuple representing the cell index.
        """
        # Assuming point is already parameterized (u, v) on the surface
        return tuple(np.floor(point / cell_size).astype(int))

    def _get_neighbours(self, cell_index, r):
        """
        Generate neighboring cells around a given cell index.

        Args:
        cell_index: The cell index.
        r: The radius to search for neighboring cells.

        Returns:
        List of tuples representing neighboring cell indices.
        """
        neighbours = []
        for i in range(cell_index[0] - r, cell_index[0] + r + 1):
            for j in range(cell_index[1] - r, cell_index[1] + r + 1):
                neighbours.append((i, j))
        return neighbours

    def _is_within_domain(self, point, surface):
        """
        Check if a point is within the trimming domain of the surface.

        Args:
        point: The point to check.
        surface: The surface entity.

        Returns:
        Boolean indicating if the point is within the domain.
        """
        u, v = surface.parameterize_point(point)
        u_min, v_min, u_max, v_max = surface._trim_domain.reshape(-1)
        return u_min <= u <= u_max and v_min <= v <= v_max

    def _parameterize_point(self, point, surface):
        """
        Convert a 3D point to its parameterized form (u, v) on the surface.

        Args:
        point: The 3D point.
        surface: The surface entity.

        Returns:
        The parameterized point (u, v).
        """
        pass
        # return surface.parameterize_point(point)
