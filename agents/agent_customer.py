import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environment.environment import sample_action_customer
from params import (EPSILON_START, 
                    DISCOUNT_RATE, 
                    EPSILON_MIN, 
                    EPSILON_DECAY, 
                    CUSTOMER_ACTION_SIZE, 
                    BUFFER_SIZE, 
                    BATCH_SIZE, 
                    CUSTOMER_STATE_SIZE, 
                    TRAINING_INTERVAL, 
                    # REPLACE_TARGET_INTERVAL, 
                    TAU, 
                    LEARNING_RATE_DQN, 
                    HIDDEN_LAYER_SIZE, 
                    POWER_RATES,
                    )
from utils.replay_buffer import ReplayBuffer
# from memory_profiler import profile

class DQN(nn.Module):
    def __init__(self):
        """
        Defining the neural network layers.

        Returns
        -------
        None.

        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(CUSTOMER_STATE_SIZE, HIDDEN_LAYER_SIZE)
        self.layer2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.layer3 = nn.Linear(HIDDEN_LAYER_SIZE, CUSTOMER_ACTION_SIZE)
    
    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor (BATCH_SIZE, CUSTOMER_STATE_SIZE)
            Batch of customer states consisting of ac demand (1), 
            open delays for 4 time-shiftable devices (4), obligatory demand from 
            non_shiftable and started non_interruptable (1), incentive (1), 
            baseline consumption (1). 

        Returns
        -------
        x : torch.Tensor (BATCH_SIZE, CUSTOMER_ACTION_SIZE)
            Batch of customer's actions.

        """

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class CustomerAgent:
    """ This CustomerAgent is a Reinforcement Learning agent using a Deep-Q 
    network to predict the Q-values of state-action pairs. In the act function 
    the agent calls for the previous reward and the next observation.
    It updates its network based on the previous reward, observation and action. 
    Then it decides upon the next action.
    """

    def __init__(
        self, 
        agent_id,
        data_id,
        env,
        dummy=False,
        q_network=None,
        target_network=None
    ):
        """
        

        Parameters
        ----------
        agent_id : int
            Customer's ID as [1, 25].
        data_id : int
            Household's ID in the dataset.
        env : object
            Environment object.
        dummy : bool, optional
            True if user always chooses maximum power rate. The default is False.
        q_network : object, optional
            Main neural network. The default is None.
        target_network : object, optional
            Target network. The default is None.

        Returns
        -------
        None.

        """
        self.agent_id = agent_id
        self.data_id = data_id
        self.env = env
        self.epsilon = EPSILON_START
        self.dummy = dummy
        self.acc_reward = 0
        self.last_state = None
        self.last_action = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.visited = {}
        self.q_network = q_network
        self.target_network = target_network
        if q_network is None:
            self.q_network = DQN().to(self.device)
        if target_network is None:
            self.target_network = DQN().to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=LEARNING_RATE_DQN
        )
        # Saving trajectory as sum of all features in an observation - v0
        self.trajectory = []
        self.acti = []
        
    def reset(self):
        """
        Reset state, action and reward at the beginning of a new episode. 
        Do a decay in epsilon parameter.

        Returns
        -------
        None.

        """
        # Saving trajectory as sum of all features in an observation - v0
        self.trajectory = []
        self.acti = []
        self.last_state = None
        self.last_action = None
        self.acc_reward = 0
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
     
    def act(self, rand=False, train=True):
        """
        Getting current observation, action and reward to add to the memory 
        buffer through step function (and maybe performing a training step).
        Also, update the values of state and action for next time period. 
        Update this episode's accumulated reward.

        Parameters
        ----------
        rand : bool, optional
            True if choosing just random actions. The default is False.
        train : bool, optional
            True for wanting to train the agent. The default is True.
        
        Returns
        -------
        None.

        """
        observation, reward, done, _ = self.env.last_customer(self.agent_id)
        if train and not self.dummy:
            if self.last_action is not None:
                self.step(
                    self.last_state, self.last_action, reward, observation, done
                )
            if not rand:
                action = self.choose_action(observation, self.epsilon)
            if rand:
                action = sample_action_customer()
        else:
            action = self.choose_action(observation)
        # Add state to trajectory list
        self.acti.append(action)
        self.get_trajectory(observation)
        self.env.act(self.agent_id, action)
        self.last_state = observation
        self.last_action = action
        self.acc_reward += reward
        
    def get_trajectory(self, state):
        """
        Saving states in a trajectory list.

        Parameters
        ----------
        state : np.array
            Observation at one period.

        Returns
        -------
        None.

        """
        self.trajectory.append(state) 
    
    def choose_action(self, state, eps=0.0):
        """
        Get the next action the customer will perform based on the given state. 
        If the agent is a dummy it will simply consume an amount equal to its 
        demand (power rate 1.0). Otherwise, epsilon-greedy strategy is used: if
        the sampled value is lower than the threshold value epsilon, sample 
        customer action randomly. If higher, choose the action for which the 
        Q-value is currently the highest.

        Parameters
        ----------
        state : numpy array (CUSTOMER_STATE_SIZE,)
            Customer state consisting of ac demand (1), 
            open delays for 4 time-shiftable devices (4), obligatory demand from 
            non_shiftable and started non_interruptable (1), incentive (1), 
            baseline consumption (1). 
        eps : float, optional
            A parameter which indicates the threshold for choosing the action 
            randomly or through maximum Q value. The default is 0.0.

        Returns
        -------
        int
            Action that the customer will perform (incentive chosen).

        """
        
        if self.dummy:
            return POWER_RATES.index(1.0)
        elif random.uniform(0, 1) < eps:
            return sample_action_customer()
        else:
            with torch.no_grad():
                state = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                )
                actions = self.q_network(state).cpu().numpy()
                action = np.argmax(actions)
                return action
    
    def step(self, state, action, reward, next_state, done):
        """
        Adding a sample to memory buffer used for generataing training samples, 
        and if the constraints are satisfied, train the agent.

        Parameters
        ----------
        state : numpy array (CUSTOMER_STATE_SIZE,)
            Customer state consisting of ac demand (1), 
            open delays for 4 time-shiftable devices (4), obligatory demand 
            from non_shiftable and started non_interruptable (1), incentive (1), 
            baseline consumption (1). 
        action : int
            Action taken by the customer.
        reward : float
            Reward value for an action and state at current time step.
        next_state : numpy array (CUSTOMER_STATE_SIZE,)
            Aggregator next state consists of of ac demand (1), 
            open delays for 4 time-shiftable devices (4), obligatory demand 
            from non_shiftable and started non_interruptable (1), incentive (1), 
            baseline consumption (1) of the following time period.
        done : bool
            True for the last time step of the episode.
        name : optional
            The default is None.

        Returns
        -------
        None.

        """
        self.memory.add(state, action, reward, next_state, done)

        # Train network on a certain interval and if the replay buffer has enough 
        # samples
        if (
            len(self.memory) >= BATCH_SIZE 
            and self.env.curr_step % TRAINING_INTERVAL == 0
        ):
            sampled_experiences = self.memory.sample()
            self.train(sampled_experiences)

        # # Hard update target network (every REPLACE_TARGET_INTERVAL parameters 
        # # of the target network just get new values from basic q network)
        # if self.env.episode % REPLACE_TARGET_INTERVAL == 0:
        #     self.target_network.load_state_dict(self.q_network.state_dict())
            

    #@profile(precision=8)
    def train(self, experiences):
        """
        Perform a training step of Q-network. Update both networks' weights. 
        Use soft update for target network to stabilize training.

        Parameters
        ----------
        experiences : tuple
            A tuple containing a batch of states, actions, rewards, states of the
            next time step and a flag indicating whether the episode is done.

        Returns
        -------
        None.

        """
        states, actions, rewards, next_states, dones = experiences
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            np.array([[actions]]), device=self.device, dtype=torch.float32
        )
        rewards = torch.tensor(
            np.array([rewards]), device=self.device, dtype=torch.float32
        )
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            np.array([dones]), device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            q_target_pred = self.target_network(next_states)
        next_actions = q_target_pred.max(1).values
        target_values = torch.add(
            rewards, 
            torch.mul(
                DISCOUNT_RATE, 
                torch.mul(next_actions, torch.logical_not(dones))
            )
        )
        q_values = self.q_network(states)
        target_values = torch.reshape(target_values, (target_values.size()[1],))
        additional = torch.clone(q_values)
        additional[
            np.arange(len(states)), actions.view(BATCH_SIZE,).cpu().numpy()
        ] = target_values
        target_values = additional
        criterion = nn.MSELoss()
        loss = criterion(q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()        
        self.soft_update_target()
        
    def soft_update_target(self):
        """
        Do soft update of target network weights. New target network weights are 
        calculated as weighted sum of old weights and new weights from Q network.

        Returns
        -------
        None.

        """
        q_network_weights = self.q_network.state_dict()
        target_net_weights = self.target_network.state_dict()
        for key in q_network_weights:
            target_net_weights[key] = (
                TAU * q_network_weights[key] + (1 - TAU) * target_net_weights[key]
            )
        self.target_network.load_state_dict(target_net_weights)

    def save(self, path, ext=None):
        """
        Save customer's Q network.
        
        Parameters
        ----------
        path : str
            Path for saving the network.

        Returns
        -------
        None.

        """
        if ext != None:
            # torch.save(
            #     self.q_network, path + r'/Q_network_' + str(self.data_id) +
            #      '_' + str(ext) + '.pt'
            # )
            # np.save(
            #     path + r'/dissatisfaction_coefficients_' + str(self.data_id) +
            #      '_' + str(ext) + '.npy', 
            #     self.env.dissatisfaction_coefficients[self.agent_id]
            # )
            torch.save(
                self.q_network, path + r'/Q_network_' + str(self.agent_id) +
                 '_' + str(ext) + '.pt'
            )
            np.save(
                path + r'/dissatisfaction_coefficients_' + str(self.agent_id) +
                 '_' + str(ext) + '.npy', 
                self.env.dissatisfaction_coefficients[self.agent_id]
            )
        else:
            # torch.save(
            #     self.q_network, path + r'/Q_network_' + str(self.data_id) + '.pt'
            # )
            # np.save(
            #     path + r'/dissatisfaction_coefficients_' + str(self.data_id) + 
            #     '.npy', 
            #     self.env.dissatisfaction_coefficients[self.agent_id]
            # )
            torch.save(
                self.q_network, path + r'/Q_network_' + str(self.agent_id) + '.pt'
            )
            np.save(
                path + r'/dissatisfaction_coefficients_' + str(self.agent_id) + 
                '.npy', 
                self.env.dissatisfaction_coefficients[self.agent_id]
            )
            
            
        print("Successfully saved network for agent " + str(self.data_id))

    def load(self, path, ext=None):
        """
        Load customer's Q network.

        Parameters
        ----------
        path : str
            Path where the network is saved.

        Returns
        -------
        None.

        """
        # if ext != None:
        #     self.q_network = torch.load(
        #         path + r'/Q_network_' + str(self.data_id) + '_' + str(ext) + '.pt'
        #         )
        #     self.env.dissatisfaction_coefficients[self.agent_id] = np.load(
        #         path + r'/dissatisfaction_coefficients_' + str(self.data_id) +
        #         '_' + str(ext) + '.npy'
        #         )
        # else:
        #     self.q_network = torch.load(
        #         path + r'/Q_network_' + str(self.data_id) + '.pt'
        #         )
        #     self.env.dissatisfaction_coefficients[self.agent_id] = np.load(
        #         path + r'/dissatisfaction_coefficients_' + str(self.data_id) +
        #         '.npy'
        #         )
        if ext != None:
            self.q_network = torch.load(
                path + r'/Q_network_' + str(self.agent_id) + '_' + str(ext) + '.pt'
                )
            self.env.dissatisfaction_coefficients[self.agent_id] = np.load(
                path + r'/dissatisfaction_coefficients_' + str(self.agent_id) +
                '_' + str(ext) + '.npy'
                )
        else:
            self.q_network = torch.load(
                path + r'/Q_network_' + str(self.data_id) + '.pt'
                )
            self.env.dissatisfaction_coefficients[self.agent_id] = np.load(
                path + r'/dissatisfaction_coefficients_' + str(self.agent_id) +
                '.npy'
                )