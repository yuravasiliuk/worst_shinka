import torch
import torch.nn as nn
import torch.optim as optim


class DQLModel(nn.Module):
    """PyTorch Deep Q-Learning model."""

    def __init__(self, input_size, output_size, hidden_layers=None):
        super().__init__()
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

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.loss_function = nn.MSELoss()

    def forward(self, x):
        return self.network(x.to(self.device))

    def predict(self, state):
        """Return Q-values for the given state."""
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            q_values = self.network(state_tensor)

        return q_values.cpu().numpy()

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
        ).to(self.device)

        next_state_tensor = torch.tensor(
            next_state,
            dtype=torch.float32
        ).to(self.device)

        # Current Q-values
        q_values = self.network(state_tensor)

        # Q-value for selected action
        current_q = q_values[action]

        # Target Q-value
        with torch.no_grad():
            next_q_values = self.network(next_state_tensor)
            max_next_q = torch.max(next_q_values)

            target_q = torch.tensor(reward, dtype=torch.float32).to(self.device)
            if not done:
                target_q += gamma * max_next_q

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
        """Save the whole model object (utils._load_model expects a plain
        torch.load(path) to hand back a ready-to-use DQLModel)."""
        torch.save(self, path)