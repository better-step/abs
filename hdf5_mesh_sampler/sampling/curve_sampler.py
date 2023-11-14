from .sampler import Sampler
import numpy as np


class CurveSampler(Sampler):
    """
    Sampler for curve entities.
    """

    def __init__(self, spacing, method='poisson_disk'):
        """
        Initialize the curve sampler.

        Args:
        curve: The curve entity to be sampled.
        spacing (float): The spacing parameter for sampling.
        method (str): The method to use for sampling, e.g., 'poisson_disk'.
        """
        super().__init__(spacing)
        self.curve = None
        self.cell_size = None
        self.method = method

    def sample(self, curve):
        """
        Sample points on a curve based on the specified method.

        Args:
        curve: The curve entity to be sampled.

        Returns:
        An array of sampled points on the curve.
        """
        # Implementation of the sampling method
        # This can be expanded to include various sampling strategies
        if self.method == 'poisson_disk':
            self.curve = curve
            self.cell_size = self._calculate_cell_size()
            return self._poisson_disk_sampling(curve)

    def _poisson_disk_sampling(self, curve):
        """
        Implement Poisson disk sampling specific to curves.

        Args:
        curve: The curve entity to be sampled.

        Returns:
        An array of sampled points.
        """
        sample_points = []
        grid, active = {}, []

        # Initialize the sampling process
        first_point = self._get_random_point_on_curve(curve)
        self._insert_point(sample_points, first_point, grid, active, curve)

        # Continue sampling until no active points remain
        while active:
            point, index = self._choose_random_active_point(active)
            new_point = self._generate_new_point_around(point, curve)
            if self._is_valid_point(new_point, curve, grid):
                self._insert_point(sample_points, new_point, grid, active, curve)

        return np.array(sample_points)

    def _get_random_point_on_curve(self, curve):
        """
        Get a random point on the curve within its defined interval.

        Args:
        curve: The curve entity.

        Returns:
        A random point on the curve.
        """
        min_i, max_i = curve._interval
        random_param = np.random.uniform(min_i, max_i)
        return curve.sample(np.array([[random_param]]))[0]

    def _insert_point(self, sample_points, point, grid, active, curve):
        """
        Insert a new point into the sample points, grid, and active list.

        Args:
        sample_points: List of sampled points.
        point: New point to be inserted.
        grid: Spatial grid for efficient searching.
        active: List of active points.
        curve: The curve entity.
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

    def _generate_new_point_around_old(self, point, curve):
        """
        Generate a new point around the given point.

        Args:
        point: The point around which to generate a new point.
        curve: The curve entity.

        Returns:
        A new point generated around the given point.
        """
        r = np.random.uniform(self.spacing, 2 * self.spacing)
        theta = np.random.uniform(0, 2 * np.pi)
        new_point_param = curve.parameterize_point(point) + r * np.cos(theta)
        return curve.sample(np.array([[new_point_param]]))[0]

    def _generate_new_point_around(self, point, curve):
        """
        Generate a new point around the given point.

        Args:
        point: The point around which to generate a new point.
        curve: The curve entity.

        Returns:
        A new point generated around the given point.
        """
        r = np.random.uniform(self.spacing, 2 * self.spacing)
        theta = np.random.uniform(0, 2 * np.pi)
        new_param = curve.parameterize_point(point) + r * np.cos(theta)

        # Ensuring new parameter is within the curve's interval
        min_i, max_i = curve._interval
        new_param = np.clip(new_param, min_i, max_i)

        return curve.sample(np.array([[new_param]]))[0]

    def _is_valid_point(self, point, curve, grid):
        """
        Check if the generated point is valid (i.e., not too close to existing points).

        Args:
        point: The point to validate.
        curve: The curve entity.
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

    def _calculate_cell_size(self):
        # Calculate cell size based on curve length and spacing
        curve_length = self.curve.length()
        if self.spacing <= 0:
            raise ValueError("Spacing must be greater than zero")
        return curve_length / np.sqrt(self.spacing)
       # return curve_length / np.sqrt(curve_length / self.spacing) // or

    def _get_cell_index(self, point, cell_size):
        """
        Calculate the cell index for a given point based on the cell size.

        Args:
        point: A point on the curve.
        cell_size: The size of the cell in the grid.

        Returns:
        The index of the cell in the grid.
        """
        # Assuming the point is a 1D parameter value on the curve
        return int(np.floor(point / cell_size))

    def _get_neighbours(self, cell_index, r):
        """
        Get neighboring cells around a given cell index within a radius.

        Args:
        cell_index: Index of the cell.
        r: Radius to consider for neighboring cells.

        Returns:
        A list of indices for neighboring cells.
        """
        return [cell_index + i for i in range(-r, r + 1)]
