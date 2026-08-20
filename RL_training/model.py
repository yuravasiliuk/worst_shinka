import torch
import torch.nn as nn
import torch.optim as optim


class DQLModel:
    """PyTorch Deep Q-Learning model."""

    def __init__(self, input_size, output_size, hidden_layers=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers or []

        # Build neural network
        layers = []
        previous_size = input_size

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.network = nn.Sequential(*layers)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=0.001
        )

        self.loss_function = nn.MSELoss()

    def predict(self, state):
        """Return Q-values for the given state."""
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():
            q_values = self.network(state_tensor)

        return q_values.numpy()

    def train_step(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        gamma=0.99
    ):
        """Perform one DQN training step."""

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        next_state_tensor = torch.tensor(
            next_state,
            dtype=torch.float32
        )

        # Current Q-values
        q_values = self.network(state_tensor)

        # Q-value for selected action
        current_q = q_values[action]

        # Target Q-value
        with torch.no_grad():
            next_q_values = self.network(next_state_tensor)
            max_next_q = torch.max(next_q_values)

            if done:
                target_q = torch.tensor(
                    reward,
                    dtype=torch.float32
                )
            else:
                target_q = torch.tensor(
                    reward,
                    dtype=torch.float32
                ) + gamma * max_next_q

        # Calculate loss
        loss = self.loss_function(
            current_q,
            target_q
        )

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        """Save the model."""
        torch.save(
            {
                "model_state": self.network.state_dict(),
                "input_size": self.input_size,
                "output_size": self.output_size,
                "hidden_layers": self.hidden_layers,
            },
            path
        )

    def load(self, path):
        """Load the model."""
        checkpoint = torch.load(
            path,
            map_location="cpu"
        )

        self.network.load_state_dict(
            checkpoint["model_state"]
        )