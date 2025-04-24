
import pytest
import torch

# we are located in the _test folder so do we need to exit out of this? 
from poser.models.st_gcn_lstm import ST_GCN_LSTM  # Update with the correct file path

@pytest.fixture
def mock_model():
    """Initialize a test instance of the ST-GCN LSTM model."""
    model = ST_GCN_LSTM(
        in_channels=3, num_class=5, graph_cfg={},
        # data cfg is ok... we need to match the input for the outputs expected in widget? 
        data_cfg={"data_dir": "", "T2": 50, "ideal_sample_no": 10, "shift": False},
        hparams={"learning_rate": 0.001, "batch_size": 2, "dropout": 0.3}
    )
    return model

def test_forward_pass(mock_model):
    """Test if the forward pass produces expected output dimensions."""
    batch_size, seq_len, num_nodes, in_channels = 2, 50, 17, 3
    test_input = torch.randn(batch_size, in_channels, seq_len, num_nodes)

    with torch.no_grad():
        output = mock_model(test_input)

    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[-1] == mock_model.num_classes, "Class output mismatch"

def test_gradient_propagation(mock_model):
    """Ensure gradients flow properly through the model."""
    batch_size, seq_len, num_nodes, in_channels = 2, 50, 17, 3
    test_input = torch.randn(batch_size, in_channels, seq_len, num_nodes, requires_grad=True)
    target = torch.randint(0, mock_model.num_classes, (batch_size,))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mock_model.parameters())

    optimizer.zero_grad()
    output = mock_model(test_input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    assert any(p.grad is not None for p in mock_model.parameters()), "No gradients found"

def test_lstm_hidden_state(mock_model):
    """Verify LSTM hidden states do not vanish or explode."""
    batch_size, seq_len, num_nodes, in_channels = 2, 50, 17, 3
    test_input = torch.randn(batch_size, in_channels, seq_len, num_nodes)

    with torch.no_grad():
        output = mock_model(test_input)
   
    hidden_state_magnitude = output.abs().mean().item()
    assert hidden_state_magnitude < 100, "Hidden state values are too large (possible exploding gradients)"
    assert hidden_state_magnitude > 1e-6, "Hidden state values are too small (possible vanishing gradients)"

def test_training_convergence(mock_model):
    """Run a short training loop to ensure loss decreases."""
    batch_size, seq_len, num_nodes, in_channels = 4, 50, 17, 3
    test_input = torch.randn(batch_size, in_channels, seq_len, num_nodes)
    target = torch.randint(0, mock_model.num_classes, (batch_size,))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mock_model.parameters())

    loss_values = []
    for _ in range(5):  # Run a few iterations
        optimizer.zero_grad()
        output = mock_model(test_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

    assert loss_values[-1] < loss_values[0], "Loss did not decrease"
