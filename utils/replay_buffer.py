import random
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """

        Parameters
        ----------
        buffer_size : int
            Maximum number of samples that can be added to the buffer.
        batch_size : int
            Number of samples to be extracted from the buffer.

        Returns
        -------
        None.

        """
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
        )

    def add(self, state, action, reward, next_state, done):
        """
        Adding samples to replay buffer.

        Parameters
        ----------
        state : numpy array
            Current state.
        action : int
            Action taken by the agent.
        reward : float
            Reward value for an action and state at current time step.
        next_state : numpy array (AGGREGATOR_STATE_SIZE,)
            Next state for the following time period.
        done : bool
            True for the last time step of the episode.
        Returns
        -------
        None.

        """
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """
        Sampling batch of samples from the replay buffer.

        Returns
        -------
        states : numpy aarray (BATCH_SIZE, STATE_SPACE)
            Sampled batch of states.
        actions : numpy array (BATCH_SIZE, ACTION_SPACE)
            Sampled batch of actions.
        rewards : numpy array (BATCH_SIZE, )
            Sampled batch of rewards.
        next_states : numpy array (BATCH_SIZE, STATE_SPACE)
            Sampled batch of next states.
        dones : numpy array (BATCH_SIZE, )
            Sampled batch of indicators of episode termination.

        """
        batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = list(
            map(np.array, list(zip(*batch)))
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        

        Returns
        -------
        int
            Number of samples in replay buffer.

        """
        return len(self.memory)
