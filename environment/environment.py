# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:38:56 2023

@author: Korisnik
"""

import numpy as np

from utils.load_demand import (
    load_requests, 
    load_day, 
    get_device_demands, 
    load_baselines, 
    get_peak_demand,
)
from params import (
    # RHO, 
    CUSTOMER_ACTION_SIZE, 
    TRAINING_START_DAY, 
    TRAINING_END_DAY, 
    INCENTIVE_RATES, 
    TIME_STEPS_TRAIN, 
    DEVICE_CONSUMPTION, 
    DISSATISFACTION_COEFFICIENTS, 
    DEVICE_NON_INTERRUPTIBLE, 
    DEVICES, 
    CRITICAL_THRESHOLD_RELATIVE, 
    MAX_TOTAL_DEMAND, 
    MAX_INCENTIVE, 
    RHO_COMMON, 
    POWER_RATES, 
    BASELINE_START_DAY, 
    NUM_AGENTS, 
    DISSATISFACTION_COEFFICIENTS_STD, 
    DISSATISFACTION_COEFFICIENTS_MIN, 
    TESTING_START_DAY, 
    TESTING_END_DAY,
    PRICES,
    AGENT_IDS
)

from utils.value_func import (
    ValueFunctions
    )

from utils.get_prices import (
    load_prices,
    load_prices_day
    )

import random
#from memory_profiler import profile

def knapsack(values, weights, capacity):
    """
    A function that schedules the devices based on their dissatisfaction "profit" 
    (the values) and their consumption (the weights). The customer selected a 
    fraction of demand as consumption (capacity). This function brute-forces the 
    optimal knapsack solution.
    Dissatisfaction "profit" is calculated as:
        (a) in case of AC: the difference between discomfort cause when doing
            total reduction to 0 (input  "values" to knapsack_ensamble) and 
            reduction to certain percentage (calculated in knapsack_ensamble)
            -> Input 'values' is representing that difference and it is better
                when it is bigger, because then the new action (reduction) is
                causing less discomfort.
        (b) in case of time-shiftable devs: the difference when they are shifted
            and when they are not shifted. So, this is a 0-1 case, either have
            discomfort or do not have it at all. 
            -> The highest difference is when 
                the dev is not shifted.
    The capacity is the maximum reducible demand - in case all available devices
    are shifted and AC totally reduced. 
    The aim of this algorithm is to maximize the "value", i.e. the dissatisfaction
    "profit" because then the least amount of devices is shifted or reduced in 
    power. Also, it wants to keep the consumption as similar to demand as possible
    so weights need to be lower than the available "capacity" for reduction.
            
    Parameters
    ----------
    values : numpy array of floats (len(DEVICES),)
        Dissatisfaction "profit" values of all devices. 
    weights : numpy array of floats (len(DEVICES),)
        Consumptions of all devices (with reduced AC).
    capacity : float
        Wanted power consumption to achieve.

    Returns
    -------
    float
        Total dissatisfaction "profit" of scheduled devices.
    float
        Total consumption of scheduled devices.
    max_actions : list of bool
        List of devices that are scheduled to operate.

    """
    # Get indices of devices that have postive dissatisfaction values (which 
    # means they can be selected), e.g for EV it will return 1
    non_zero_values = np.nonzero(values)[0]
    n = len(non_zero_values)
    max_value = 0
    # The biggest demand that can be reduced
    max_weight = capacity
    # An array that will store which device works and which not (0 or 1)
    max_actions = np.zeros(len(values), dtype=bool)
    # Keeping track of each devices dissatisfaction profit
    max_value_per_dev = []
    # Keeping track of consumption per devices
    max_weight_per_dev = []

    if n == 0:
        return 0, 0, max_actions, [], []

    for i in range(2 ** n):
        # Get all possible actions, e.g. if n=2, actions will be 0-0, 0-1, 1-0, 
        # 1-1
        actions = np.array([int(x) for x in list(f'{i:b}'.zfill(n))], dtype=bool)
        # Return device for whose index the "actions" value says True, e.g. if 
        # actions = [1,1],  values = [11.623 0 2 0 0] -> non_zero_vals = [0, 2] 
        # -> action_indices = [0, 2]. If actions was [0, 1], action_indices would
        # be [2].
        # Return the device index
        action_indices = non_zero_values[actions]
        # Get sum of dissatisfaction values of the devices selected for action
        value = values[action_indices].sum()
        # Get sum of consumption values of the devices selected for this action
        weight = weights[action_indices].sum()
        # In case new reduced demand (weight) is higher than the max_weight, it 
        # corresponds to the first constraint and in order to become new best,
        #  it has to have higher dissatisfaction "profit". Otherwise, there is 
        #  no point in having higher reduced demand then current max_weight. In 
        #  case new weight is lower than max_weight it will probably have higher 
        #  or equal value of dissat as current max and it will be lower than cap
        #  So, in the first constraint it will surely be (1 and 0/1), and in case 
        #  it is (1 and 1) it will pass if constraint and if it is (1 and 0) it 
        #  still leaved the option for value to be "equal" to max_value
        #  -> weight must be <= capacity (and is good that is as lowest as possible)
        #  -> and value needs to be >= than max value
        if (
            (weight <= capacity and value > max_value) 
            or (weight <= max_weight and value == max_value)
        ):
            max_value = value
            max_weight = weight
            max_actions = np.zeros(len(values), dtype=bool)
            # Save the actions that lead to the best consumption and dissatis.
            max_actions[action_indices] = True
            max_value_per_dev = values[action_indices]
            max_weight_per_dev = weights[action_indices]

    return max_value, max_weight, max_actions, max_value_per_dev, max_weight_per_dev


def knapsack_ensemble(values, weights, capacity, dissatisfaction_coefficients):
    """
    This function should return scheduled devices, the overall dissatisfaction 
    and action (power rate) that led to that.

    Parameters
    ----------
    values :  numpy array of floats (len(DEVICES),)
        Calculated dissatisfactions for all selectable devices.
    weights : numpy array of floats (len(DEVICES),)
        Device demand values.
    capacity : float
        Wanted consumption to be achieved.
    dissatisfaction_coefficients : numpy array of floats (len(DEVICES),)
        Dissatisfaction coefficient values for all devices of the user.

    Returns
    -------
    Tuple
        Tuple contains (dissatisfaction "profit" value, consumption that is the 
        closest to the wanted one (capacity), bool list which states which 
        device was shifted/reduced, power rate)

    """
    max_values = []
    max_weights = []
    max_actionss = []
    rates = []
    max_value_per_devs = []
    max_weight_per_devs = []
    ac_index = DEVICES.index('air')
    # Gets AC demand (maximum consumption possible, it was assumed that the whole
    # demand is reduced in the main function)
    ac_consumption = weights[ac_index]
    # Get discomfort value for reducing whole AC demand (to 0!)
    ac_max_value = values[ac_index]
    for rate in POWER_RATES[1:]:
        # Reduce consumption to only a percentage of demand
        ac_weight = ac_consumption * rate
        # Amount of reduction in case the AC demand is reduced to ac_weight
        ac_reduction = ac_consumption - ac_weight
        # Calculate new dissatisfaction value for that case
        ac_value = (
            dissatisfaction_coefficients[ac_index] * np.square(ac_reduction)
        )
        # For how much does the dissatisfaction improve in case of new (smaller)
        # reduction, in comparison to reducing the whole demand
        ac_value = ac_max_value - ac_value
        # New consumption of AC, with reduction to "rate" percentage
        weights[ac_index] = ac_weight
        # New dissatisfaction value for AC, which is a dissat. "profit" of 
        # reducing the consumption to rate of the whole in comparison to whole
        values[ac_index] = ac_value
        # Return the highest dissatisf. "profit", its accompanying consumption
        # and list of devices that were scheduled
        max_value, max_weight, max_actions, max_value_per_dev, max_weight_per_dev = knapsack(values, weights, capacity)
        max_values.append(max_value)
        max_weights.append(max_weight)
        max_actionss.append(max_actions)
        max_value_per_devs.append(max_value_per_dev)
        max_weight_per_devs.append(max_weight_per_dev)
        rates.append(rate)

    # Sort the values in descending order based on dissatisfaction "profit"
    # Actually, ascending, but with negative sign
    # Wanted: highest 'values', lowest 'weights'
    sorted_values = sorted(
        zip(max_values, max_weights, max_actionss, rates, max_value_per_devs, max_weight_per_devs),
        key=lambda elem: (-elem[0], elem[1])
    )
    return sorted_values[0]


def sample_action_customer():
    """
    A method that samples a random action for the customer.

    Returns
    -------
    int
        Random number from 0 to CUSTOMER_ACTION_SIZE defining the action in the 
        list of customers' actions.

    """
    return np.random.randint(0, CUSTOMER_ACTION_SIZE)

class Environment:
    """ The CustomerAgents interact with the Environment. 
    The Environment controls input of device requests and demands. It schedules 
    devices with the knapsack algorithm for the CustomerAgents. Finally, it 
    calculates the rewards.
    """
    def __init__(
            self,
            data_ids,
            heterogeneous=False,
            baseline=False,
            ):
        self.data_ids = data_ids
        self.episode = 0
        self.df = load_requests()
        self.df_prices = load_prices(fif=True)
        self.heterogeneous = heterogeneous
        self.baseline = baseline
        self.dissatisfaction_coefficients = np.full(
            (len(data_ids), len(DEVICES)),
            DISSATISFACTION_COEFFICIENTS
        )
        if heterogeneous:
            dissatisfaction_coefficients = np.random.normal(
                loc=DISSATISFACTION_COEFFICIENTS, 
                scale=DISSATISFACTION_COEFFICIENTS_STD,
                size=(NUM_AGENTS, len(DEVICES))
            )
            self.dissatisfaction_coefficients = np.maximum(
                DISSATISFACTION_COEFFICIENTS_MIN,
                dissatisfaction_coefficients
            )
        self.trajj = {}
    
    def reset(self, day=None, max_steps=TIME_STEPS_TRAIN, irl=False, alphas=None):
        """
        The function resets variables related to customer and demand 
        to default values.
        
        Parameters
        ----------
        day : int
            Number order of the day - day: 177 -> actual date: 2018-06-26. The 
            default is None. 
        max_steps : int
            Maximum number of time steps in one episode. The default is 
            TIME_STEPS_TRAIN.
        irl: bool
            If training is done for IRL purposes.
        Returns
        -------
        None.
        """
        self.day = day
        if day is None:
            day_range = [
                (TRAINING_START_DAY, TESTING_START_DAY), 
                (TESTING_END_DAY, TRAINING_END_DAY)
            ][np.random.randint(0, 2)]
            self.day = np.random.randint(*day_range) 
            
            
        self.curr_step = 0
        self.episode += 1
        self.done = False
        self.max_steps = max_steps

        self.demand = np.zeros(
            (
                max_steps,
                len(self.data_ids)
            )
        )
        self.non_shiftable_load = np.zeros(
            (
                max_steps, len(self.data_ids)
            )
        )
        self.requests_new = np.zeros(
            (
                max_steps,
                len(self.data_ids),
                len(DEVICES)
            ), 
            dtype=bool
        )         
        self.request_loads = np.zeros(
            (
                max_steps,
                len(self.data_ids),
                len(DEVICES)
            )
        )                        
        self.requests_started = np.zeros(
            (
                max_steps,
                len(self.data_ids),
                len(DEVICES)
            ),
            dtype=bool
        )
        self.requests_open = np.zeros(
            (
                max_steps,
                len(self.data_ids),
                len(DEVICES)
            )
        )                    
        self.requests_delayed = np.zeros(
            (
                max_steps,
                len(self.data_ids),
                len(DEVICES)
            )
        )                 

        self.possible_actions = np.zeros(
            (
                max_steps,
                len(self.data_ids)
            )
        )      
        self.power_rates = np.zeros(
            (
                max_steps,
                len(self.data_ids)
            )
        )      
        self.request_actions = np.zeros(
            (
                max_steps,
                len(self.data_ids),
                len(DEVICES)
            ),
            dtype=bool
        )      
        self.ac_rates = np.zeros(
            (
                max_steps,
                len(self.data_ids)
            )
        )
        self.consumptions = np.zeros(
            (
                max_steps,
                len(self.data_ids)
            )
        )
        self.incentive_received = np.zeros(
            (
                max_steps,
                len(self.data_ids)
            )
        )
        self.rewards_customers = np.zeros(
            (
                max_steps,
                len(self.data_ids)
            )
        )
        self.dissatisfaction = np.zeros(
            (
                max_steps,
                len(self.data_ids),
                len(DEVICES)
            )
        )
        self.customer_reward_matrix = np.zeros(
            (
                max_steps, 
                 len(INCENTIVE_RATES),
                 len(self.data_ids),
                 len(POWER_RATES)
             )
        )
        
        # Promijeni u to da učitaš iz excela ili neceG- Simplest version
        self.incentives = np.array(PRICES)

        self.day_df = load_day(self.df, self.day, max_steps)
        self.baselines = load_baselines()
        self.set_demands()
        self.capacity_threshold = (
            get_peak_demand(self.day_df) * CRITICAL_THRESHOLD_RELATIVE
        )
        self.IRL = irl
        self.alphas = alphas
        self.init_state = None
        self.actions = []
        for k in range(0, NUM_AGENTS):
            self.trajj[k] = []
    def last_customer(self, agent_id):
        """ 
        The CustomerAgent can call this method to receive the previous reward 
        and the next observation. The observation consists of the state of the 
        household appliances and the offered incentive. The state of the household
        appliances is defined as an integer, 0 for no request or requests for 
        non-interruptible devices that have been started, 1 for a new request 
        and > 1 if the request has been delayed.

        Parameters
        ----------
        agent_id : int
            Agent's ID.

        Returns
        -------
        observation : numpy array of floats (8,)
            Observation for the next time period (s'). (8,) -> ac demand (1), 
            open delays for 4 time-shiftable devices (4), obligatory demand from 
            non_shiftable and started non_interruptable (1), incentive (1), 
            baseline consumption (1).
        reward : float
            Previous reward (reward[t] for state t-1).
        done : bool
            True if the episode (day) finished.
        TYPE
            Default is None.

        """
        incentive = self.incentives[self.curr_step]
        baseline = self.baselines[agent_id][self.day - BASELINE_START_DAY][self.curr_step]
        new_requests = self.requests_new[self.curr_step][agent_id]
        started_requests = self.requests_started[self.curr_step][agent_id]
        open_requests = self.requests_open[self.curr_step][agent_id] + new_requests
        delays = self.requests_delayed[self.curr_step][agent_id]
        new_delays = new_requests + delays
        open_delays = new_delays * np.invert(started_requests)
        ac_consumption = self.request_loads[self.curr_step][agent_id][0]
        non_shiftable = self.non_shiftable_load[self.curr_step][agent_id]
        non_interruptible = (
            np.logical_and(open_requests, started_requests) * DEVICE_CONSUMPTION
        ).sum()
        
        observation = np.array(
            np.concatenate(
                (
                    [ac_consumption],
                     open_delays[1:],
                     [non_shiftable + non_interruptible, incentive, baseline]
                 )
            )
        )
        if self.curr_step == 0:
            self.init_state = round(observation.sum(), 2)
        reward = self.rewards_customers[self.curr_step][agent_id]
        done = self.done
        return observation, reward, done, None

    def act(self, agent_id, action):
        """
        Apply the action selected by a CustomerAgent.
        The agent selects a power rate and sends it to the environment. 
        Based on this power rate this method calls the knapsack algorithm and 
        determines the devices scheduled for this time step. Afterwards this
        method calculates the new state of the appliances taking device-specific 
        constraints into account.

        Parameters
        ----------
        agent_id : int
            ID of the household agent.
        action : int
            Power rate value for demand reduction (index of the value in actions 
           set).

        Returns
        -------
        None.

        """
        self.actions.append(action)
        # Get power rate and incentive rate
        # Get incentive rate for this time step
        incentive_rate = self.incentives[self.curr_step]
        # Get baseline demand for this time step and agent and day
        baseline_demand = self.baselines[agent_id][self.day - BASELINE_START_DAY][self.curr_step]
        # Get index of EV in list DEVICES
        car_index = DEVICES.index('car')
        # Get index of AC in list DEVICES
        ac_index = DEVICES.index('air')
        # Get power rate based on the action of customer from list of customer 
        # actions
        power_rate = POWER_RATES[action]
        if self.baseline:
            power_rate = POWER_RATES[
                np.argmax(
                    self.customer_reward_matrix[self.curr_step][int(incentive_rate)][agent_id]
                )
            ]

        # Get requests and demands
        # True if non-interruptible device is on, otherwise False
        started_requests = self.requests_started[self.curr_step][agent_id]
        # True if demand for request exists, otherwise False
        new_requests = self.requests_new[self.curr_step][agent_id]
        # Returns the number of unfullfiled requests for each device. Includes 
        # new requests at this time step, as well.
        open_requests = self.requests_open[self.curr_step][agent_id] + new_requests
        # Gets amount of delayed time steps for devices
        delayed_requests = self.requests_delayed[self.curr_step][agent_id]
        # Gets True for each time shiftable device (EV, WM, DW, DRY) that can be 
        # selected for operation (if it is not turned on already). Removes already turned on non-interruptible ones.
        selectable_requests = np.logical_and(
            open_requests, np.invert(started_requests)
        )
        # Gets if there are any requests for a non-interruptable device that is 
        # already on and running (it will just continue running)
        non_interruptible_requests = np.logical_and(
            open_requests, started_requests
        )
        # For those requests, calculated the total consumption
        non_interruptible_demand = (
            non_interruptible_requests * DEVICE_CONSUMPTION
        ).sum()
        # Gets non-shiftable demand
        non_shiftable_demand = self.non_shiftable_load[self.curr_step][agent_id]
        # Gets the consumption value for all time shiftable devices that can be 
        # selected for operation (and others but those will be equal to 0)
        device_consumptions = selectable_requests * DEVICE_CONSUMPTION
        # Get AC consumption demand based on data, since it is a float value and 
        # cannot be calculated as time_steps * fixed_consumption
        device_consumptions[ac_index] = self.request_loads[self.curr_step][agent_id][ac_index]
        # Brute-force knapsack
        # Dissatisfaction of a consumer from delayed time-shiftable devices, for 
        # AC it will be overwritten
        dissatisfaction_values = (
            self.dissatisfaction_coefficients[agent_id] * np.square(delayed_requests + 1)
        )        
        # Dissatisfaction of a consumer from AC if whole AC demand is reduced to 
        # 0 - the WORST dissatisfaction
        dissatisfaction_values[ac_index] = (
            self.dissatisfaction_coefficients[agent_id][ac_index] * np.square(device_consumptions[ac_index])
        )
        # Dissatisfaction set for all selectable devices
        device_values = dissatisfaction_values * selectable_requests
        # Sum of all selectable demand that can be shifted or reduced
        shiftable_demand = (selectable_requests * device_consumptions).sum()
        # Demand amount of total shiftable/reduceable that can be reduced =
        # Maximum reducible consumption amount (TS + PC)
        capacity = power_rate * shiftable_demand
        # Get device schedule based on dissatisfaction values of all devices, 
        # consumption of all devices, reducible amount of power, dissatisfaction 
        # coef of all devices for current agent
        value, weight, actions, ac_rate, value_per_devs, weight_per_devs = knapsack_ensemble(
            device_values,
            device_consumptions,
            capacity,
            self.dissatisfaction_coefficients[agent_id]
        )
        reducing_diss = []
        dev_el = 0
        for dev in actions:
            if dev:
                reducing_diss.append(value_per_devs[dev_el])
                dev_el += 1
            else:
                reducing_diss.append(0.0)
        # Inverted actions is True when device was not working and was delayed
        delayed_devices = np.invert(actions) * selectable_requests
        # Get true dissatisfaction, after scheduling the devices 
        dissatisfaction = device_values.sum() - value

        extract_dis = device_values - np.array(reducing_diss)
        extr_dis = []
        for dev in range(0, len(extract_dis)):
            dels = np.sqrt(extract_dis[dev]/self.dissatisfaction_coefficients[agent_id][dev])
            extr_dis.append(dels)
        # Calculate received incentive
        consumption = weight + non_interruptible_demand + non_shiftable_demand

        energy_diff = baseline_demand - consumption
        incentive_received = incentive_rate * max(0, energy_diff)

        # Calculate reward
        incentive_term = incentive_received
        dissatisfaction_term = -dissatisfaction
        
        if not self.IRL:
            reward = incentive_term + dissatisfaction_term
       
        # AC request load
        ac_consumption = self.request_loads[self.curr_step][agent_id][0]
        # print('ac', ac_consumption)
        new_requests = self.requests_new[self.curr_step][agent_id]
        # Open delays
        delays = self.requests_delayed[self.curr_step][agent_id]
        new_delays = new_requests + delays
        started_requests = self.requests_started[self.curr_step][agent_id]
        open_delays = new_delays * np.invert(started_requests)
        # Non shiftable and non interruptible
        non_shiftable = self.non_shiftable_load[self.curr_step][agent_id]
        open_requests = self.requests_open[self.curr_step][agent_id] + new_requests
        non_interruptible = (
            np.logical_and(open_requests, started_requests) * DEVICE_CONSUMPTION
        ).sum()
        # Incentive
        incentive = self.incentives[self.curr_step]
        # Baseline
        baseline = self.baselines[agent_id][self.day - BASELINE_START_DAY][self.curr_step]

        x = np.array(
            np.concatenate(
                (
                    [weight],
                    [
                        (non_shiftable + non_interruptible),
                        incentive,
                        baseline,
                    ],
                    extr_dis
                )
            )
        )
        
        if self.IRL:
            states = []
            vf = ValueFunctions()
             
            for m in range(0, 6):
                if (m == 0):
                    states.append(vf.true_test(x.reshape((1,-1))).reshape((-1,)))                   
                if (m == 1):
                    states.append(vf.basis_1(x.reshape((1,-1))).reshape((-1,)))
                if (m == 2):
                    states.append(vf.basis_2(x.reshape((1,-1))).reshape((-1,)))
                if (m == 3):
                    states.append(vf.basis_3(x.reshape((1,-1))).reshape((-1,)))                
                if (m == 4):
                    states.append(vf.basis_4(x.reshape((1,-1))).reshape((-1,)))
                if (m == 5):
                    states.append(vf.basis_5(x.reshape((1,-1))).reshape((-1,)))
                    
            states = np.array(states)
            reward = (states * self.alphas.reshape((-1,1))).sum()
          
        # trajectories for V(S)
        self.trajj[agent_id].append(x)
        
        # Save selected devices, energy consumption and received incentive
        fulfilled_requests = np.logical_or(actions, non_interruptible_requests)
        self.possible_actions[self.curr_step][agent_id] = np.count_nonzero(selectable_requests)
        self.request_actions[self.curr_step][agent_id] = fulfilled_requests
        self.consumptions[self.curr_step][agent_id] = consumption
        self.incentive_received[self.curr_step][agent_id] = incentive_received
        self.power_rates[self.curr_step][agent_id] = power_rate if selectable_requests.any() else 1
        self.ac_rates[self.curr_step][agent_id] = ac_rate * actions[ac_index] if selectable_requests[ac_index] else 1
        self.dissatisfaction[self.curr_step][agent_id] = device_values * delayed_devices
        self.dissatisfaction[self.curr_step][agent_id][ac_index] = (
            self.dissatisfaction_coefficients[agent_id][ac_index] * np.square((1 - ac_rate) * device_consumptions[ac_index])
        )

        # Update parameters for use in the next time step
        if self.curr_step < self.max_steps - 1:
            started_non_interruptibles = actions * DEVICE_NON_INTERRUPTIBLE
            open_requests_next = open_requests - fulfilled_requests

            self.rewards_customers[self.curr_step + 1][agent_id] = reward
            self.requests_open[self.curr_step + 1][agent_id] = open_requests_next
            self.requests_started[self.curr_step + 1][agent_id] = (
                started_non_interruptibles + non_interruptible_requests
            )
            self.requests_delayed[self.curr_step + 1][agent_id] = (
                delayed_requests + delayed_devices
            )
            self.requests_delayed[self.curr_step + 1][agent_id][started_non_interruptibles] = 0

            # If all requested time slots for the EV are fulfilled reset the delay
            if open_requests_next[car_index] == 0:
                self.requests_delayed[self.curr_step + 1][agent_id][car_index] = 0

            # AC has no delay
            self.requests_delayed[self.curr_step + 1][agent_id][ac_index] = 0
            self.requests_open[self.curr_step + 1][agent_id][ac_index] = 0

    def step(self):
        """
        This method is called at the end of a time step.
        Increases the time step for one and sets done flag to True if it is the 
        final time step.
        If it was not the final time step, demands for the next time step are 
        retrieved.

        Returns
        -------
        None.

        """
        self.curr_step += 1
        self.done = self.curr_step == self.max_steps
        if not self.done:
            self.set_demands()



    def set_demands(self):
        """ Retrieve the demands per customer and per device for the current 
        time step from the demands DataFrame.
        If the demand is larger than a certain threshold the device is considered 
        requested by the user. The actual load in kW that is requested for the 
        device is fixed, except for the total non-shiftable devices. 
        
        Returns
        ---------
        None.
        
        """
        df = get_device_demands(
            self.day_df,
            self.data_ids,
            self.day,
            self.curr_step
        )
        non_shiftable = df['non-shiftable'].to_numpy()
        total = df['total'].to_numpy()
        requests = df[DEVICES].to_numpy()
        request_new = np.greater(requests, 0)
        self.non_shiftable_load[self.curr_step] = non_shiftable
        self.requests_new[self.curr_step] = request_new
        self.request_loads[self.curr_step] = requests
        self.demand[self.curr_step] = total
    
    def get_total_demand(self, step=None):
        """ Sum the demands of the customer agents for all time steps. 

        Returns
        -------
        list of floats or float
            Total demand of all users summed at a certain time period.
        """
        if step is None:
            return self.demand.sum(axis=1)
        return self.demand[step].sum()
    
    def get_total_consumption(self, step=None):
        """
        A method that returns the sum of the consumptions of all customer agents.

        Parameters
        ----------
        step : int, optional
            Wanted time step. The default is None.

        Returns
        -------
        float
            Total actual consumption of all users summed at a certain time period.

        """
        if step is None:
            return self.consumptions.sum(axis=1)
        return self.consumptions[step].sum()
    
    def get_total_reduction(self, step=None):
        """
        A method that calculates the reduction of power of all customers together.   

        Parameters
        ----------
        step : int, optional
            Wanted time step. The default is None.

        Returns
        -------
        float
            The amount of reduced power at 15-minute time period calculated as 
            the difference between total demand and total actual consumption. 
        """
        if step is None:
            return self.get_total_demand() - self.get_total_consumption()
        return self.get_total_demand(step) - self.get_total_consumption(step)
    
    def set_baseline(self):
        """
        An update of baseline values with the average of the pre-computed baseline 
        with the consumption of the last time step for a more accurate result.
            
        Returns
        -------
        None.

        """
        baseline_demand = self.baselines[:, self.day - BASELINE_START_DAY, self.curr_step]
        new_baseline_demand = (
            baseline_demand + self.consumptions[self.curr_step - 1]
        ) / 2
        self.baselines[:, self.day - BASELINE_START_DAY, self.curr_step] = new_baseline_demand
