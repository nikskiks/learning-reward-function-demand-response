import time
import datetime
import numpy as np

from utils.get_results import (
    plot_schedule, 
    print_metrics,
    export_xlsx,
)

from params import (
    TIME_STEPS_TEST, 
    TESTING_START_DAY, 
    TESTING_END_DAY, 
    NUM_RL_AGENTS
)

def test_function(
    env, 
    customer_agents, 
    path, 
    plot_test=False, 
    plot_agent=1, 
    plot_incentive=True,
    export = True,
    start_day=TESTING_START_DAY, 
    end_day=TESTING_END_DAY,
    alphas=None,
    ext=None,
    irl=False,
    o=False
):
    """
    This function perfoms testing for the designated period. If indicated, it 
    plots the results and prints the average metrics (demand, consumption, 
    incentives, etc.) during that period.
    
    Parameters
    ----------
    env : object
        Environment object.
    aggregator_agent : object
        Aggregator agent object.
    customer_agents : object
        Customer agents object. 
    path : str
        Path to save figures with results.
    plot_test : bool, optional
        True for generating figures with results. The default is False.
    plot_agent : int, optional
        An integer from 1 to NUM_AGENTS deciding for which customer agent to 
        generate results. The default is 0.
    plot_incentive : bool, optional
        True for generating incentive curve in figures. The default is True.
    start_day : int, optional
        An integer deciding the start of testing period. The default is 
        TESTING_START_DAY.
    end_day : int, optional
        An integer deciding the end of testing period. The default is 
        TESTING_END_DAY.

    Returns
    -------
    None.

    """
    TESTING_PERIOD = end_day - start_day
    agents_rewards = np.zeros((TESTING_PERIOD, NUM_RL_AGENTS))
    incentives_received = np.zeros((TESTING_PERIOD, NUM_RL_AGENTS))
    aggregator_rewards = np.zeros(TESTING_PERIOD)
    total_demands = np.zeros((TESTING_PERIOD, TIME_STEPS_TEST))
    total_consumptions = np.zeros((TESTING_PERIOD, TIME_STEPS_TEST))
    peak_demand = np.zeros(TESTING_PERIOD)
    peak_consumption = np.zeros(TESTING_PERIOD)
    mean_demand = np.zeros(TESTING_PERIOD)
    mean_consumption = np.zeros(TESTING_PERIOD)
    thresholds = np.zeros(TESTING_PERIOD)
    runtime = np.zeros(TESTING_PERIOD)
    incentives = np.zeros((TESTING_PERIOD, TIME_STEPS_TEST))
    test_days = []
    
    if irl:
        IRL = irl
        ALP = alphas
    else:
        IRL = False
        ALP = None
        
    print('Testing...')
    
    changes = []
    init_s = []
    
    for day in range(start_day, end_day):
        print('day', day)
        env.reset(day=day, max_steps=TIME_STEPS_TEST, irl=IRL, alphas=ALP)

        # Load agents
        if path is not None:
            for agent in customer_agents:
                agent.load(path, ext)

        # Run single day
        start = time.time()
        for iteration in range(TIME_STEPS_TEST):
            for agent in customer_agents:
                agent.act(train=False)
            env.step()
            
        changes.append(env.demand - env.consumptions)
        init_s.append(env.init_state)
        
        end = time.time()
        test_days.append(datetime.datetime.strptime('{} {}'.format(day, 2018),
                                                    '%j %Y').date())
        agents_rewards[day - start_day] = env.rewards_customers.sum(axis=0)[
            :NUM_RL_AGENTS
        ]
        incentives_received[day - start_day] = env.incentive_received.sum(axis=0)[
            :NUM_RL_AGENTS
        ]
        incentives[day - start_day] = env.incentives
        total_demands[day - start_day] = env.get_total_demand() 
        total_consumptions[day - start_day] = env.get_total_consumption() 
        peak_demand[day - start_day] = np.max(env.get_total_demand())
        peak_consumption[day - start_day] = np.max(env.get_total_consumption())
        mean_demand[day - start_day] = np.mean(env.get_total_demand())
        mean_consumption[day - start_day] = np.mean(env.get_total_consumption())
        thresholds[day - start_day] = env.capacity_threshold  
        runtime[day - start_day] = end - start 
        
        # if plotting is enabled, plot results for current day
        if plot_test:
            time_labels = [
                datetime.datetime(year=2018, month=1, day=1) 
                + datetime.timedelta(days=day - 1, minutes=i * 15)
                for i in range(TIME_STEPS_TEST)
            ]
            plot_agent = customer_agents[plot_agent-1]
            plot_schedule(env, plot_agent, time_labels, path, o)

    # Get numerical results for the testing period - ODKOMENTIRAJ
    print_metrics([
        test_days, 
        agents_rewards, 
        incentives_received, 
        incentives, 
        total_demands, 
        total_consumptions, 
        peak_demand, 
        peak_consumption, 
        mean_demand, 
        mean_consumption, 
        thresholds, 
        runtime
    ], path)

    
    # if export:
    #     export_xlsx([
    #         test_days, 
    #         agents_rewards, 
    #         aggregator_rewards, 
    #         incentives_received, 
    #         incentives, 
    #         total_demands, 
    #         total_consumptions, 
    #         peak_demand, 
    #         peak_consumption, 
    #         mean_demand, 
    #         mean_consumption, 
    #         thresholds, 
    #         runtime
    #     ], path)
        
    return changes, init_s, peak_consumption, agents_rewards[:,0]