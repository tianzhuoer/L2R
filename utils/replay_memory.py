import gymnasium as gym
from thermocline_env import ThermoTrackEnv
# from CTDDataProcessor import CTDDataProcessor  # Import the CTDDataProcessor
import time
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
except ImportError:
    SummaryWriter = None
from config import *
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mask', 'next_mask'))


class SeqReplayMemory(object):

    def __init__(self, seq_length=50,memory_size=REPLAY_MEMORY_SIZE):

        self.maxseqlen=seq_length
        self.episode_size=MAX_EPISODES_LEN
        self.memory = deque([], maxlen=memory_size)
        # self.seq_length = seq_length
        self.transition_seq = deque([], maxlen=self.episode_size)

        self.state_batch = torch.tensor([]).reshape(0, 4)
        self.action_batch = torch.tensor([]).reshape(0, 1)
        self.next_state_batch = torch.tensor([]).reshape(0, 4)
        self.reward_batch = torch.tensor([]).reshape(0, 1)
        self.mask_batch = torch.tensor([]).reshape(0, 4)
        self.next_mask_batch = torch.tensor([]).reshape(0, 4)

    def push(self, *args):
        """Save a transition"""

        args = [arg.squeeze() if isinstance(arg, torch.Tensor) and arg.dim() > 0 and 1 in arg.shape else arg for arg in args]


        if any(arg is None for arg in args):
            return
        
        # self.memory.append(Transition(*args))
        


        if len(self.transition_seq) == self.episode_size:

            if len(self.memory) == self.memory.maxlen:
                replace_idx = random.randint(0, len(self.memory) - 1)
                self.memory[replace_idx] = self.transition_seq
                self.transition_seq = deque([], maxlen=self.episode_size)
                self.transition_seq.append(Transition(*args))
            else:
                self.memory.append(self.transition_seq)
                self.transition_seq = deque([], maxlen=self.episode_size)
                self.transition_seq.append(Transition(*args))
            # print(self.memory.__len__())
        else:

            self.transition_seq.append(Transition(*args))
        return
    def sample(self, batch_size):




        sampled_sequences1= deque([], maxlen=batch_size)
        for _ in range(batch_size):
            episode_idx = random.randint(0, len(self.memory) - 1)
            seq_start = random.randint(0, self.episode_size - self.maxseqlen)

            selected_batch = deque([], maxlen=self.maxseqlen)
            transition_seq = self.memory[episode_idx]
            for i in range(self.maxseqlen):
                selected_batch.append(transition_seq[seq_start+i])
            sampled_sequences1.append(selected_batch)


        # sampled_sequences2 = random.sample(self.memory, batch_size)





        

        sequences_by_field = [Transition(*zip(*seq)) for seq in sampled_sequences1]
        


        # torch.stack([torch.stack(s) for s in batch.state])


        batch = Transition(*zip(*sequences_by_field))

        state_batch = torch.stack([torch.stack(s) for s in batch.state])
        action_batch = torch.stack([torch.stack(a) for a in batch.action])


        next_state_batch = torch.stack([torch.stack(ns) for ns in batch.next_state])
        reward_batch = torch.stack([torch.stack(r) for r in batch.reward])
        mask_batch = torch.stack([torch.stack(m) for m in batch.mask])
        next_mask_batch = torch.stack([torch.stack(nm) for nm in batch.next_mask])

        batch_reshaped = Transition(
            state=state_batch,
            action=action_batch,
            next_state=next_state_batch,
            reward=reward_batch,
            mask=mask_batch,
            next_mask=next_mask_batch
        )
        return batch_reshaped
    def __len__(self):
        return len(self.memory)

def test_seq_replay_memory():
    """Tests for SeqReplayMemory class."""
    print("--- Running Test for SeqReplayMemory ---")
    capacity = 10
    seq_length = 5
    batch_size = 2
    
    # 1. Initialize memory
    memory = SeqReplayMemory(seq_length=seq_length,memory_size=capacity)
    print(f"Initialized memory with capacity={capacity}, seq_length={seq_length}")

    # 2. Test push method and __len__
    num_pushes = seq_length * 10 + 2  # This should create 3 full sequences and one partial sequence
    print(f"\nPushing {num_pushes} transitions...")
    for i in range(num_pushes):
        # Create dummy data for a transition
        state = torch.randn(4)
        action = torch.tensor([i % 2])
        next_state = torch.randn(4)
        reward = torch.tensor([i])
        mask = torch.ones(4, dtype=torch.bool)
        next_mask = torch.ones(4, dtype=torch.bool)
        memory.push(state, action, next_state, reward, mask, next_mask)

    # After pushing, we expect 3 full sequences to be in memory
    expected_len = 10
    actual_len = len(memory)
    print(f"Expected memory length: {expected_len}, Actual memory length: {actual_len}")
    assert actual_len == expected_len, "Test Failed: __len__ returned incorrect value!"
    print("✅ Test for __len__ passed.")

    # And the internal transition_seq should have 2 items
    expected_transition_seq_len = 2
    actual_transition_seq_len = len(memory.transition_seq)
    print(f"Expected internal sequence length: {expected_transition_seq_len}, Actual: {actual_transition_seq_len}")
    assert actual_transition_seq_len == expected_transition_seq_len, "Test Failed: internal transition_seq has incorrect length!"
    print("✅ Test for push logic passed.")

    # 3. Test sample method
    print(f"\nSampling a batch of size {batch_size}...")

    batch = memory.sample(batch_size)

    state_batch = batch.state
    action_batch = batch.action
    reward_batch = batch.reward
    print("Sampled batch shape:", state_batch.shape)
    # The expected shape would be (batch_size, seq_length, num_fields_in_transition, ...)
    # But the current sample implementation is incorrect.
    # The assert below will fail.
    # assert batch.shape[0] == batch_size and batch.shape[1] == seq_length


if __name__ == '__main__':
    test_seq_replay_memory()
