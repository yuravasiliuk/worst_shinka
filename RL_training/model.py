from abc import ABC, abstractmethod


class DQLModel(ABC):
    """Abstract base class for Deep Q-Learning models."""

    def __init__(self, input_size, output_size, hidden_layers=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers or []

    @abstractmethod
    def predict(self, state):
        """Return Q-values for the given state."""
        pass

    @abstractmethod
    def train_step(self, state, action, reward, next_state, done):
        """Perform one training step."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the model."""
        pass

    @abstractmethod
    def load(self, path):
        """Load the model."""
        pass
