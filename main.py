import torch
import numpy as np

from training import train_function
from testing import test_function
from agents.agent_customer import CustomerAgent
from environment.environment import Environment
from params import EPISODES, NUM_RL_AGENTS, AGENT_IDS, PRICES
from irl.irl import optimizing
from utils.value_optimal import value_optimal

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy import stats
import pandas as pd
import seaborn as sns
# from memory_profiler import profile


if __name__ == "__main__":
    
    # If training on GPU use EPISODES number of episodes for training, instead reduce to 2000
    if torch.cuda.is_available():
        num_episodes = EPISODES
    else:
        num_episodes = 3500
    
    
    data_frame = {}
    # True for training
    TRAIN_PHASE = 1
    # Save training values (rewards and their plots)
    LOG_VAL = 1
    # Save trained networks
    SAVE_NET = 1
    # True for testing
    TEST_PHASE = 1
    # Define the start and the end of testing period, available days [182,213]
    START_TEST_DAY = 182 #182
    END_TEST_DAY = 213 #213
    # Try to recover the reward function 
    IRL = 1
    # Finding basic Q networks - optimal and first (random)
    OPTI = 1
    FIRST = 0
    # Define the number of household for which to do the plotting
    PLOT_AGENT = 1
    # Define whether to plot or not
    PLOT_TEST = 0
    # Plot graphs with incentives on it or not
    PLOT_INCENTIVE = True
    # Export test data to Excel file
    EXPORT_TO_XLSX = True
    # Policy path extension
    EXT = [
        "opt", 
        "1.0"
    ]
    # Path for saving trained networks
    save_path = 'Enter path for saving files'
    cus_a = {}
    env = Environment(AGENT_IDS, heterogeneous=False, baseline=False)
    Vses= {}
        
        
    if TRAIN_PHASE and OPTI:
        alphas = None
        rand = 0
        exten = EXT[0]
        
        cus_a[exten] = [
            CustomerAgent(agent_id, data_id, env, dummy=agent_id >= NUM_RL_AGENTS) 
            for agent_id, data_id in enumerate(AGENT_IDS)
        ]
        
        trs = train_function(
            env, 
            cus_a[exten], 
            num_episodes, 
            save_path, 
            log=LOG_VAL, 
            save=SAVE_NET,
            alphas=alphas,
            ext=exten,
            irl_train=0, 
            rand = rand
        )
        
        # optimal = EXT[0]
    if TEST_PHASE:
        print('Optimal with TRUE reward:')
        ch_opti, ini_opti, pc_opti, ar_opti = test_function(
            env,  
            cus_a[exten], 
            save_path, 
            plot_test=PLOT_TEST, 
            plot_agent=PLOT_AGENT-1, 
            plot_incentive=PLOT_INCENTIVE, 
            export = EXPORT_TO_XLSX,
            start_day=START_TEST_DAY, 
            end_day=END_TEST_DAY,
            alphas=None,
            ext=exten,
            irl=False, 
            o=True
        )
        
    OPTI = 0
    FIRST = 1
    rand = 1
    
    save_path = 'Enter path for saving files'
    
    if TRAIN_PHASE and FIRST:
        exten = EXT[1]
        
        cus_a[exten] = [
            CustomerAgent(agent_id, data_id, env, dummy=agent_id >= NUM_RL_AGENTS) 
            for agent_id, data_id in enumerate(AGENT_IDS)
        ]
        
        trs_first = train_function(
            env, 
            cus_a[exten], 
            num_episodes, 
            save_path, 
            log=LOG_VAL, 
            save=SAVE_NET,
            alphas=alphas,
            ext=exten,
            irl_train=0, 
            rand = rand,
        )
        
    FIRST = 0 
    vs_ext = []
                
    rewards = []
    # Path for saving trained networks
    save_path = 'Enter path for saving files'
    
    print("Calculating V_opti")
    vs_opti = value_optimal(env, cus_a[EXT[0]], trs)
    
    vs_ext.append(vs_opti)
    
    print("Calculating V_first")
    vs_first = value_optimal(env, cus_a[EXT[1]], trs_first)
    vs_ext.append(vs_first)
    
    keys = list(vs_ext[0][0].keys())
    d = len(vs_ext[0][0][keys[0]][0])

    cnt = 0
    aps = []
    aps_val = []
    conss = []
    acc = []
    peakk = []
    reww = []
    
    # vs0 = []
    # vss = []
    if TRAIN_PHASE and not (OPTI or FIRST) and IRL:
        while cnt < 10:
            alphas = np.array(optimizing(vs_ext[0], vs_ext[1:], d))
            aps.append(alphas.sum())
            aps_val.append(alphas)
            print('Alphas', alphas)
            
            # # For only 1 state
            # s = list(vs_ext[0][0].keys())[0] #21.22 je za day 92
            # vs0.append((vs_ext[0][0][s][0]*alphas.reshape((-1,1))).sum())
            # vss.append((vs_ext[-1][0][s][0]*alphas.reshape((-1,1))).sum())
            # plt.figure(1)
            # plt.plot(vs0, label='opt')
            # plt.plot(vss, label='vss')
            # plt.legend()
            # plt.title('V(21.22)') #21.48 je za 131
            # plt.show()
            
            cus_a[str(100+cnt)] = [
                CustomerAgent(agent_id, data_id, env, dummy=agent_id >= NUM_RL_AGENTS) 
                for agent_id, data_id in enumerate(AGENT_IDS)
            ]
            
            vs_k = train_function(
                env, 
                cus_a[str(100+cnt)],
                num_episodes, 
                save_path, 
                log=LOG_VAL, 
                save=SAVE_NET,
                alphas=alphas,
                ext=(100+cnt),
                irl_train=IRL,
            )
            
            vs_ext.append(vs_k)
            EXT.append(str(100+cnt))

            print('Last with TRUE reward:')
            ch_latr, ini_latr, pc_latr, ar_fit = test_function(
                env, 
                cus_a[str(100+cnt)], 
                save_path, 
                plot_test=PLOT_TEST,
                plot_agent=PLOT_AGENT-1, 
                plot_incentive=PLOT_INCENTIVE, 
                export = EXPORT_TO_XLSX,
                start_day=START_TEST_DAY, 
                end_day=END_TEST_DAY,
                alphas=None,
                ext=str(100+cnt),
                irl=False,
                o=False
            )
            
            reww.append(np.mean(ar_fit))
            conss = env.consumptions
            acc = env.actions
           
            # plt.figure(2)
            # plt.title('Rews per iteration')
            # plt.plot(reww, label='f')
            # plt.plot(np.ones((len(reww), 1))*np.mean(ar_opti), label='o')
            # plt.legend()
            # plt.show()
            
            plt.figure(3)
            plt.plot(aps, 'o-')
            plt.grid()
            plt.show()
            cnt += 1

    #Rewards plot test
    plt.rcParams["figure.figsize"] = (11,6)
    plt.ylabel('Reward', fontsize=18)
    plt.xlabel('Iteration', fontsize=18)
    plt.xticks(np.arange(0, len(reww),2), np.arange(1, len(reww)+1, 2), fontsize=12)
    plt.yticks(fontsize=12)
    plt.plot(np.mean(ar_opti)*np.ones(len(reww),), label="optimal", color='blue')
    plt.plot(reww, 'o-', label="learned", color='green')
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig('Enter path for saving files', bbox_inches = 'tight')
    plt.show()
    
    plt.rcParams["figure.figsize"] = (11,6)
    plt.plot(aps, 'o-')
    plt.ylabel("Sum of alphas", fontsize=18)
    plt.xlabel("Iteration", fontsize=18)
    plt.xticks(np.arange(0, len(reww),2), np.arange(1, len(reww)+1, 2), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.savefig('Enter path for saving files', bbox_inches = 'tight')
    plt.show()
            
    if TEST_PHASE:
        
        rangg = np.arange(182, 213) 
        
        # Define whether to plot or not
        PLOT_TEST = 0
        
        maesD = []
        msesD = []
        pearsD = []
    
        for dayyy in rangg:
            
            
            START_TEST_DAY = dayyy #92
            END_TEST_DAY = dayyy + 1 #93
            save_path_opti = 'Enter path for saving files'
            print('Optimal with TRUE reward:')
            ch_opti, ini_opti, pc_opti, ar_opti = test_function(
                env, 
                cus_a[EXT[0]], 
                save_path_opti, 
                plot_test=PLOT_TEST, 
                plot_agent=PLOT_AGENT-1, 
                plot_incentive=PLOT_INCENTIVE, 
                export = EXPORT_TO_XLSX,
                start_day=START_TEST_DAY, 
                end_day=END_TEST_DAY,
                alphas=None,
                ext=EXT[0],
                irl=False,
                o=True
            )
            
            optimal_cons = env.consumptions
            optimal_actions = env.actions
              
            print('Last with TRUE reward:')
            ch_latr, ini_latr, pc_latr, ar_fit = test_function(
                env, 
                cus_a[EXT[-1]], 
                save_path, 
                plot_test=PLOT_TEST,
                plot_agent=PLOT_AGENT-1, 
                plot_incentive=PLOT_INCENTIVE, 
                export = EXPORT_TO_XLSX,
                start_day=START_TEST_DAY, 
                end_day=END_TEST_DAY,
                alphas=None,
                ext=EXT[-2],
                irl=False,
                o=False
            )
            
            conss = env.consumptions
            acc = env.actions
            
    
            for i in range(0, len(ch_opti)):
                for household in range(0, ch_opti[i].shape[1]):
                    plt.rcParams["figure.figsize"] = (11,6)
                    plt.figure(household)
                    # plt.title(f"Fig {household+1}")
                    plt.plot(ch_opti[i][:, household], '*-', label='optimal', color="blue")
                    plt.plot(ch_latr[i][:, household], 'o-', label='learned', color="green")
                    plt.ylabel('Consumption change (kW)', fontsize=18)
                    plt.xlabel('15-minute periods', fontsize=18)
                    plt.xticks(np.arange(0,96,15), np.arange(1,97,15), fontsize=12)
                    plt.yticks(fontsize=12)
                    plt.legend(fontsize=15)
                    plt.grid()
                    
                    path = (
                        'Enter path for saving files'
                        )
                    plt.savefig(path, bbox_inches = 'tight')
                    plt.show()
            

            
            perc = []
            for a in range(0, len(ch_opti)):
                for household in range(0, ch_opti[a].shape[1]):
                    print('Household:', household)
                    maxi_val = 0
                    if (max(abs(ch_opti[0][:,household])) > maxi_val):
                        maxi_val = max(ch_opti[0][:,household])
                    if (max(abs(ch_latr[0][:,household])) > maxi_val):
                        maxi_val = max(ch_latr[0][:,household])
                    br = 0
                    ba  = 0
                    for ee in range(0, ch_opti[a].shape[0]):
                        if not(ch_opti[a][ee, household] == 0 and ch_latr[a][ee, household] == 0):
                            if abs(ch_opti[a][ee, household] - ch_latr[a][ee, household]) > 0:
                                br += 1
                            else:
                                ba += 1
                    if br+ba>0:
                        print('Razlika u %d je %f posto slucajeva.' %(a+1, 100*br/(br+ba)))
                        perc.append(100*br/(br+ba))
            
            maes = []
            mses = []
            pears = []
            for a in range(0, len(ch_opti)):
                for household in range(0, ch_opti[a].shape[1]):
                    print('Household:', household)
                    maxi_val = 0
                    if (max(abs(ch_opti[0][:,household])) > maxi_val):
                        maxi_val = max(ch_opti[0][:,household])
                    if (max(abs(ch_latr[0][:,household])) > maxi_val):
                        maxi_val = max(ch_latr[0][:,household])
                    if maxi_val != 0:
                        maes.append(mean_absolute_error(ch_opti[0][:,household]/maxi_val, ch_latr[0][:,household]/maxi_val))
                        mses.append(mean_squared_error(ch_opti[0][:,household]/maxi_val, ch_latr[0][:,household]/maxi_val))
                        pears.append(stats.pearsonr((ch_opti[0][:,household]/maxi_val).reshape((-1,)), (ch_latr[0][:,household]/maxi_val).reshape((-1,))))
                    else:
                        maes.append(0.)
                        mses.append(0.)
                        pears.append(0.)
            maesD.append(maes)
            msesD.append(mses)
            pearsD.append(pears)

            data_frame = pd.DataFrame({'MAE':maesD, 'MSE':msesD, 'Pear':pearsD})
            
    print('Writing in xlsx..')
    with pd.ExcelWriter('Enter path for saving files'') as writer:
        data_frame.to_excel(writer, sheet_name='results', index=False)
    
    # # DAYS HOUSEHOLDS
    # plt.figure(figsize=(11, 6))
    # dff = pd.DataFrame(maesD, columns=[f'Household {d}' for d in np.arange(1,26)])
    # plt.figure(figsize=(13,10), dpi= 80)
    # sns.boxplot(data=dff, notch=False)
    # plt.xticks(np.arange(0,25,1), np.arange(1,26,1), fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.xlabel("Household no.", fontsize=18)
    # # plt.title('MAE', fontsize=22)
    # save_path = 'Enter path for saving files'
    # plt.savefig(save_path, bbox_inches = 'tight')
    # plt.show()
