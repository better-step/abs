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

        elif self.method == 'uniform':
            return self._uniform_sampling(curve)

        elif self.method == 'random':
            return self._random_sampling(curve)

        else:
            raise ValueError(f"Invalid sampling method: {self.method}")

    def _other_uniform_sampling(self, curve):

        # TODO: Fix for zero case

        curve_box = curve.sample(curve._interval)
        curve_diag = np.linalg.norm(curve_box[0, :] - curve_box[1, :])
        # For closed curves, the number of samples is determined by the curve length
        if curve_diag == 0.0:
            num_samples = max(int(curve.length() / self.spacing), 1)
        else:
            num_samples = max(int(abs(curve._interval[1] - curve._interval[0]) / self.spacing), 1)

        return np.linspace(curve._interval[0], curve._interval[1], num_samples)

    def _uniform_sampling(self, curve):
        """
            General function to sample points on various curve types.
            :param curve: Dictionary containing curve information.
            :return: Sample points for the given curve.
        """
        type_to_function = {
            'Line': self.sample_line,
            'Circle': self.sample_circle,
            'Ellipse': self.sample_ellipse,
            'BSpline': self._other_uniform_sampling,
            'Other': self._other_uniform_sampling
        }

        sampling_function = type_to_function.get(curve._type)
        if sampling_function:
            return sampling_function(curve)
        else:
            raise ValueError(f"Unknown curve type: {curve._type}")

        return uv_values

    def sample_line(self, curve):
        interval = sorted(curve._interval)
        length = interval[1] - interval[0]
        if length == 0:
            return np.array([])
        num_points = int(np.ceil(length / self.spacing))

        return np.linspace(interval[0], interval[1], num_points)

    def sample_circle(self, curve):
        radius = curve._radius
        interval = curve._interval
        if radius <= 0:
            return np.array([])

        circumference = 2 * np.pi * radius
        num_points = int(np.ceil(circumference / self.spacing))

        return np.linspace(interval[0], interval[1], num_points)
    def sample_ellipse(self,curve):
        maj_radius = curve._maj_radius
        min_radius = curve._min_radius

        # Edge case: Zero or negative radii
        if maj_radius <= 0 or min_radius <= 0:
            return np.array([])

        interval = curve._interval

        # Edge case: Major and minor radii are equal (circle)
        if maj_radius == min_radius:
            return self.sample_circle(curve)

        # Approximate circumference of the ellipse
        h = ((maj_radius - min_radius) ** 2) / ((maj_radius + min_radius) ** 2)
        circumference = np.pi * (maj_radius + min_radius) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        num_points = int(np.ceil(circumference / self.spacing))
        return np.linspace(interval[0], interval[1], num_points)


    def _random_sampling(self, curve):
        num_samples = max(int(abs(curve._interval[1] - curve._interval[0]) / self.spacing), 1)
        return np.random.uniform(curve._interval[0], curve._interval[1], num_samples)


    def _poisson_disk_sampling(self, curve):
            grid = {}
            active = []
            sample_points = []

            cell_size = self._calculate_cell_size(curve)
            first_point = np.random.uniform(*curve._interval)
            sample_points, grid, active = self._insert_sample(sample_points, first_point, grid, active, cell_size)

            while active:
                point, index = self._get_random_point(active)
                k_points = self._get_random_points_around(point, self.spacing, self.k, curve._interval)

                insert_flag = False
                for k_point in k_points:
                    closest_point = self._get_closest_point(k_point, grid, cell_size, self.spacing)
                    if closest_point is None or np.linalg.norm(closest_point - k_point) > self.spacing:
                        sample_points, grid, active = self._insert_sample(sample_points, k_point, grid, active, cell_size)
                        insert_flag = True

                if not insert_flag:
                    active.pop(index)

            return np.array(sample_points)

    def _insert_sample(self, sample_points, point, grid, active, cell_size):
        sample_points.append(point)
        cell_index = self._get_cell_index(point, cell_size)
        grid[cell_index] = point
        active.append(point)
        return sample_points, grid, active

    def _get_cell_index(self, point, cell_size):
        return int(np.floor(point / cell_size))

    def _get_random_point(self, active):
        random_index = np.random.randint(0, len(active))
        return active[random_index], random_index

    def _get_random_points_around(self, point, spacing, k, interval):
        r_values = np.random.uniform(spacing, 2 * spacing, k)
        points = point + np.concatenate((r_values, -r_values))
        return np.clip(points, *interval)

    def _get_closest_point(self, point, grid, cell_size, spacing):
        cell_index = self._get_cell_index(point, cell_size)
        r = int(np.ceil(spacing / cell_size))
        neighbours = self._get_neighbours(cell_index, r)

        min_distance = np.inf
        closest_point = None
        for neighbour in neighbours:
            if neighbour in grid:
                distance = np.linalg.norm(grid[neighbour] - point)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = grid[neighbour]
        return closest_point

    def _get_neighbours(self, cell_index, r):
        return range(max(cell_index - r, 0), min(cell_index + r + 1, int(1 / self.spacing)))

    def _calculate_cell_size(self, curve):
        curve_length = np.abs(curve._interval[1] - curve._interval[0])
        return curve_length / np.sqrt(curve_length / self.spacing)
