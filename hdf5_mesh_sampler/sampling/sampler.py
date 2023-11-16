from abc import ABC, abstractmethod

class Sampler(ABC):
    """
    Abstract base class for all sampler types.
    """

    def __init__(self, spacing):
        """
        Initialize the sampler with common properties.

        Args:
        spacing (float): The spacing parameter for sampling.
        """
        self.spacing = spacing

    @abstractmethod
    def sample(self, geometry):
        """
        Abstract method to be implemented for sampling.

        Args:
        geometry: The geometric entity to be sampled.

        Returns:
        An array of sampled points.
        """
        pass
