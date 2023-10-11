import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os
import os.path
from os import path

from params import BASELINE_START_DAY, TIME_STEPS_TEST

# Parameter for moving average calculation
WINDOW_SIZE = 30

def print_convergence_plots(
        save_path,
        cust_rewards,
        customer_agents
):
    """
    Calling functions for plotting customers' rewards and 
    defining saving paths for generated figures. 

    Parameters
    ----------
    save_path : str
        Path for saving generated figures.
    cust_rewards : dict
        Dictionary with customers as keys and lists with their rewards per 
        episode as values.
    customer_agents : object
        Customer agents object.

    Returns
    -------
    None.

    """
    save_path = save_path + r'\figs/'
    if not(path.exists(save_path)):
        os.mkdir(save_path)
    for agent in customer_agents:
        cust_path = (
            save_path + r'\Customer' + str(agent.agent_id + 1) + 'Reward.png'
        )
        print_reward_customers(cust_path, cust_rewards, agent.agent_id)
    
def print_reward_customers(path, cust_rewards, agent_id):
    """
    Plots customer's reward function through episodes.

    Parameters
    ----------
    path : str
        Path for saving generated figures.
    cust_rewards : dict
        Dictionary with customers as keys and lists with their rewards per 
        episode as values.
    agent_id : int
        ID of a certain agent for which the reward is plotted.

    Returns
    -------
    None.

    """
    window_size = WINDOW_SIZE
    i = 0
    moving_averages = []
    while i < len(cust_rewards[agent_id]) - window_size + 1:
        window_average = round(
            np.sum(cust_rewards[agent_id][i:i+window_size]) / window_size, 2
        )
        moving_averages.append(window_average)
        i += 1   
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(cust_rewards[agent_id], color='lightskyblue')
    plt.plot(moving_averages, color='blue')
    plt.savefig(path)
    plt.show()

def print_metrics(data, path):
    """
    The function prints metrics over testing period. It gives general information 
    such as run time and the period during which the testing was performed.
    Then, it gives mean rewards of customer agents. 
    Lastly, values of demand with and without demand response event are given, 
    together with offered incentives.

    Parameters
    ----------
    data : list
        test_days: gives the list of dates for which the testing was performed.
        agents_rewards: a numpy array of agents' rewards over testing period, 
            (TESTING_PERIOD x NUM_RL_AGENTS).
        incentives_received: a numpy array of incentives received by customer 
                            agents,
            (TESTING_PERIOD, NUM_RL_AGENTS).
        incentives: a numpy array of incentives offered by aggregator at each 
                    time step during testing period, 
            (TESTING_PERIOD, TIME_STEPS_TEST).
        total_demands: a numpy array of total demands at each time step during 
                        testing period, 
            (TESTING_PERIOD, TIME_STEPS_TEST).
        total_consumptions:a numpy array of total consumptions at each time step
                            during testing period,
            (TESTING_PERIOD, TIME_STEPS_TEST).
        peak_demand: a numpy array of daily peak demands during testing period, 
            (TESTING_PERIOD).
        peak_consumption: a numpy array of daily peak consumptions during testing
                            period,
            (TESTING_PERIOD).
        mean_demand: a numpy array of daily mean demands during testing period, 
            (TESTING_PERIOD).
        mean_consumption: a numpy array of daily mean consumptions during testing
                        period,
            (TESTING_PERIOD).
        thresholds: a numpy array of thresholds above which the consumption will
                    be penalized, 
            (TESTING_PERIOD).
        runtime: a numpy array of runtimes for each testing day, (TESTING_PERIOD).
    path : str
        Path to destination of saved files.

    Returns
    -------
    None.

    """
    (test_days,
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
     runtime) = data
    
    start = test_days[0]
    if len(test_days) > 1:
        end = test_days[len(test_days)-1]
        
    # print('-----------------------------------------------------------------------------------------')
    # print('                             Testing Information:')
    # print('-----------------------------------------------------------------------------------------')
    # print('Saving path:', str(path))
    # if len(test_days) == 1:
    #     print('Testing day:', start)
    # if len(test_days) > 1:
    #     print('Testing period:')
    #     print('Start:', start)
    #     print('End:', end)
    # print('Mean run time: %.4f sec' %(np.mean(runtime)))
    # print('\n')
    # print(
    #     '-----------------------------------------------------------------------------------------')
    # print('                                 Rewards:')
    # print('-----------------------------------------------------------------------------------------')
    # print('Mean customer agent rewards:', np.mean(agents_rewards))
    # print('Mean customer reward per agent:', np.mean(agents_rewards, axis=0))
    print('Mean customer reward per day: ', np.mean(agents_rewards, axis=1))
    # print('\n')
    
    # print('-----------------------------------------------------------------------------------------')
    # print('                            Comparison with and without DR:')
    # print('-----------------------------------------------------------------------------------------')
    # print('Metric                   No DR       with DR')
    # print('................................................')
    # print('Peak load:               %.2f kW    %.2f kW' %(
    #     np.mean(peak_demand), np.mean(peak_consumption))
    # )
    # print('Mean load:               %.2f kW     %.2f kW' %(
    #     np.mean(mean_demand), np.mean(mean_consumption))
    # )
    # if len(test_days)==1:
    #     print('Std:                     %.2f        %.2f' %(
    #         np.mean(np.std(total_demands)), np.mean(np.std(total_consumptions)))
    #     )
    # print('PAR:                     %.2f         %.2f' %(
    #     np.mean(peak_demand) / np.mean(mean_demand),
    #     np.mean(peak_consumption) / np.mean(mean_consumption))
    # )
    # print('Mean total incentive paid:     0 €          %.2f €' %(
    #     np.mean(np.sum(incentives_received, axis=1)))
    # )
    print('Mean incentive received per agent:  0 €         %.2f €' %(
        np.mean(incentives_received))
    )
    # print('Threshold exceedance:    %.2f kW    %.2f kW'
    #       %(
    #           np.mean(
    #               np.sum(
    #                   np.maximum(0, total_demands - thresholds[:, None]),
    #                   axis=1
    #               )
    #           ),
    #           np.mean(
    #               np.sum(
    #                   np.maximum(0, total_consumptions - thresholds[:, None]),
    #                   axis=1)
    #               )
    #           )
    # )
    # print('\n')
    # if len(test_days)==1:
    #     print('-----------------------------------------------------------------------------------------')
    #     print('                          Aggregated Load Curve Information:')
    #     print('-----------------------------------------------------------------------------------------')
    #     print('Demand:', np.mean(total_demands))
    #     print('Load curve:', np.mean(total_consumptions))
    #     print('Incentives:', np.mean(incentives, axis=0))
    #     print('Capacity: %.2f kW' %(np.mean(thresholds)))
        

def export_xlsx(data, save_path):
    """
    The function exports metrics over testing period.

    Parameters
    ----------
    data : list
        test_days: gives the list of dates for which the testing was performed.
        agents_rewards: a numpy array of agents' rewards over testing period,
            (TESTING_PERIOD x NUM_RL_AGENTS).
        incentives_received: a numpy array of incentives received by customer 
                            agents,
            (TESTING_PERIOD, NUM_RL_AGENTS).
        incentives: a numpy array of incentives offered by aggregator at each 
                    time step during testing period,
            (TESTING_PERIOD, TIME_STEPS_TEST).
        total_demands: a numpy array of total demands at each time step during
                        testing period,
            (TESTING_PERIOD, TIME_STEPS_TEST).
        total_consumptions:a numpy array of total consumptions at each time step
                            during testing period,
            (TESTING_PERIOD, TIME_STEPS_TEST).
        peak_demand: a numpy array of daily peak demands during testing period,
            (TESTING_PERIOD).
        peak_consumption: a numpy array of daily peak consumptions during testing
                        period,
            (TESTING_PERIOD).
        mean_demand: a numpy array of daily mean demands during testing period, 
            (TESTING_PERIOD).
        mean_consumption: a numpy array of daily mean consumptions during testing 
                        period,
            (TESTING_PERIOD).
        thresholds: a numpy array of thresholds above which the consumption will
                    be penalized, 
            (TESTING_PERIOD).
        runtime: a numpy array of runtimes for each testing day, (TESTING_PERIOD).
    path : str
        Path to destination of saved files.

    Returns
    -------
    None.

    """
    (test_days,
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
     runtime) = data
    
    save_path = save_path + r'\xlsx/'
    if not(path.exists(save_path)):
        os.mkdir(save_path)
        
    save_path = save_path + r'\results.xlsx'
    
    df_days = pd.DataFrame(test_days)
    df_runtime = pd.DataFrame(runtime)
    df_agents = pd.DataFrame(agents_rewards)
    df_increc = pd.DataFrame(incentives_received)
    df_incent = pd.DataFrame(incentives)
    df_totdem = pd.DataFrame(total_demands)
    df_totcon = pd.DataFrame(total_consumptions)
    df_peakd = pd.DataFrame(peak_demand)
    df_peakc = pd.DataFrame(peak_consumption)
    df_meand = pd.DataFrame(mean_demand)
    df_meanc = pd.DataFrame(mean_consumption)
    df_thres = pd.DataFrame(thresholds)
    df_exthdem = pd.DataFrame(
        np.sum(np.maximum(0, total_demands - thresholds[:, None]), axis=1)
    )
    df_exthcon = pd.DataFrame(
        np.sum(np.maximum(0, total_consumptions - thresholds[:, None]), axis=1)
    )
    
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
    
    df_days.to_excel(writer, sheet_name='days')
    df_runtime.to_excel(writer, sheet_name='runtime')
    df_agents.to_excel(writer, sheet_name='agents_rewards')
    df_increc.to_excel(writer, sheet_name='incentives_received')
    df_incent.to_excel(writer, sheet_name='incentives')
    df_totdem.to_excel(writer, sheet_name='total_demands')
    df_totcon.to_excel(writer, sheet_name='total_consumptions')
    df_peakd.to_excel(writer, sheet_name='peak_demand')
    df_peakc.to_excel(writer, sheet_name='peak_consumtpion')
    df_meand.to_excel(writer, sheet_name='mean_demand')
    df_meanc.to_excel(writer, sheet_name='mean_consumtpion')
    df_thres.to_excel(writer, sheet_name='threshold')
    df_exthdem.to_excel(writer, sheet_name='exceeding_threshold_demands')
    df_exthcon.to_excel(writer, sheet_name='exceeding_threshold_consum')
    
    writer.close()
    

def plot_schedule(env, plot_agent, time_labels, path, o=1):
    """
    Function for plotting device schedules for a certain agent.

    Parameters
    ----------
    env : object
        Environment object.
    plot_agent : object
        Object of one customer agent for which the plotting is done.
    time_labels : list
        List of time labels for which the results are plot (x axis).
    path : str
        Path to destination for saving the plots.

    Returns
    -------
    None.

    """

    requests = env.requests_new[:, plot_agent.agent_id][:TIME_STEPS_TEST] 
    actions = env.request_actions[:, plot_agent.agent_id][:TIME_STEPS_TEST] 
    
    demanded_reqs = requests + np.zeros(actions.shape)
    realized_reqs = actions + np.zeros(actions.shape)
    
    #If it realized the wanted demand = 0, if it realized when there was not demand
    # = -1, if the demand was not realized =1
    
    res_reqs = demanded_reqs - realized_reqs
    time_labels = time_labels[:TIME_STEPS_TEST]
    incentives = env.incentives[:TIME_STEPS_TEST]
    power_rates = env.power_rates[:, plot_agent.agent_id][:TIME_STEPS_TEST]
    ac_rates = env.ac_rates[:, plot_agent.agent_id][:TIME_STEPS_TEST]
    data = np.hstack((res_reqs[:,1:], ac_rates.reshape(-1,1)) )

    plt.figure(figsize=(11, 6))
    plt.xticks(np.arange(0.5, data.shape[1] + 0.5), ['EV', 'WM', 'DW', 'DRY', 'AC'], fontsize=12)
    plt.yticks(np.arange(0, data.shape[0]/2, 4), np.arange(1,13), fontsize=12)
    plt.ylabel('Time (h)', fontsize=18)
    c = plt.pcolor(data[48*1:48*2,:], edgecolors='k', linewidths=1, cmap='RdBu', vmin=-1.0, vmax=1.0)
    plt.colorbar(c)
    
    path = path + r'/figs/Schedule'
    if o:
        path += '_opt.pdf'
    else:
        path += '.pdf'
    plt.savefig(path, bbox_inches = 'tight')
    plt.show()





