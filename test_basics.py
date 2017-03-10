import numpy as np
import pytest

from agent import ReplayMemory

# Seeding the RNG with the same value to make tests more deterministic
np.random.seed(0)


def _generate_sample(state_dim, n_actions):
    return np.random.rand(state_dim), np.random.randint(0, n_actions), np.random.rand()


def _np_find(matrix, vector):
    for i in range(matrix.shape[0]):
        if (vector == matrix[i]).all():
            return i
    return np.nan


def test_memory():
    memory_size, state_dim, n_actions = 20, 100, 5
    memory = ReplayMemory(memory_size=memory_size, state_dim=state_dim, n_actions=n_actions)

    items_to_add = memory_size // 2
    for i in range(items_to_add):
        state, action, reward = _generate_sample(state_dim, n_actions)
        memory.add(state, action, reward, terminal=0)
    # Sampling more than the number of items added
    with pytest.raises(AssertionError):
        memory.sample(k=items_to_add + 1)
    for i in range(memory_size + 10):
        state, action, reward = _generate_sample(state_dim, n_actions)
        memory.add(state, action, reward, terminal=0)

    assert memory.actual_size == memory_size, "Memory size was not equal to the memory_size"
    with pytest.raises(AssertionError):
        memory.sample(memory_size)
    prestates, actions, states, rewards = memory.sample(memory_size // 2)
    for pre_s, s in zip(prestates, states):
        assert _np_find(memory.states, pre_s) == (
            _np_find(memory.states, s) - 1) % memory.actual_size, "A prestate was not located before its respective state"
    assert not ((prestates == 0).all(axis=1).any() or (states == 0).all(
        axis=1).any()), "A state or presate was all 0s (uninitialized)."

    prestates1, states1, actions1, rewards1 = memory.sample(memory_size - 1)
    prestates2, states2, actions2, rewards2 = memory.sample(memory_size - 1)
    assert not (prestates1 == prestates2).all()
    assert not (states1 == states2).all()
    assert not (actions1 == actions2).all()
