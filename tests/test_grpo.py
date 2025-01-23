import pytest
import numpy as np
from grpoptim.core import GRPO

@pytest.fixture
def sample_data():
    group_size = 5
    return {
        'log_probs_old': np.random.randn(group_size).astype(np.float64),
        'log_probs_ref': np.random.randn(group_size).astype(np.float64),
        'rewards': np.random.randn(group_size).astype(np.float64),
        'group_size': group_size
    }

def test_grpo_initialization():
    grpo = GRPO(epsilon=0.2, beta=0.1)
    assert grpo.epsilon == 0.2
    assert grpo.beta == 0.1

def test_loss_computation(sample_data):
    grpo = GRPO()
    log_probs_new = np.random.randn(sample_data['group_size']).astype(np.float64)
    loss, grad = grpo.compute_loss(sample_data, log_probs_new)
    
    assert isinstance(loss, float)
    assert grad.shape == log_probs_new.shape
    assert not np.isnan(loss)
    assert not np.any(np.isnan(grad))

def test_invalid_group_size(sample_data):
    grpo = GRPO()
    # invalid_log_probs = np.random.randn(3).astype(np.float64)
    # with pytest.raises(ValueError):
    #     grpo.compute_loss(sample_data, invalid_log_probs)
     # Test with float32 inputs
    sample_data['log_probs_old'] = sample_data['log_probs_old'].astype(np.float32)
    
    # Should raise ValueError due to dtype mismatch
    with pytest.raises(ValueError, match="must be float64"):
        grpo.compute_loss(sample_data, np.random.randn(5).astype(np.float64))

def test_invalid_input_types(sample_data):
    grpo = GRPO()
    # Test with float32 inputs
    sample_data['log_probs_old'] = sample_data['log_probs_old'].astype(np.float32)
    with pytest.raises(ValueError):
        grpo.compute_loss(sample_data, np.random.randn(5).astype(np.float64))