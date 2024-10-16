import copy

import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
writer = SummaryWriter('/root/tf-logs/DQNV530-dim256')
writer = SummaryWriter('/root/tf-logs/DQNV530-dim256')
# writer_acloss = SummaryWriter('/root/tf-logs/vrp2.0tacloss')
# writer_crloss = SummaryWriter('/root/tf-logs/vrp2.0tcrloss')
def train_off_policy_agent(env, agent, num_episodes,replay_buffer, minimal_size, batch_size):
    return_list = []
    dqn_losses=[]
    maxre = -10000
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i,ncols=100) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'actveh': [], 'neactveh': []}
                state = env.reset()
                done = False
                # 每个车辆的等待时间
                v_wait = []
                # 每个车辆的总时间
                v_time = []
                # 每个车辆的总路程
                v_distance = []
                # 每个车辆的路径
                v_list=[]
                # 所有车辆的等待时间
                allv_wait=[]
                # 所有车辆的总时间
                allv_time=[]
                # 所有车辆的最长耗时
                allv_maxt = []
                # 所有车辆的总路程
                allv_distance=[]
                # vetotal_time = np.zeros(env.vehicle)
                # vewait_totalt = np.zeros(env.vehicle)
                venodes = [[] for _ in range(env.vehicle)]
                redistance_return = np.zeros(env.vehicle)
                actveh=0
                while not done:
                    action = agent.take_action(state,actveh)
                    next_state, reward, done,total_time,wait_totalt,vehicle_loc,distance,neactveh = env.step(action,actveh)
                    # transition_dict['states'].append(state)
                    # transition_dict['actions'].append(action)
                    # transition_dict['next_states'].append(next_state)
                    # transition_dict['rewards'].append(reward)
                    # transition_dict['dones'].append(done)
                    # transition_dict['actveh'].append(actveh)
                    # transition_dict['neactveh'].append(neactveh)
                    replay_buffer.add(state, action, next_state, reward, done,actveh,neactveh)
                    state = copy.deepcopy(next_state)
                    episode_return += reward
                    # # 每辆车总时间
                    # vetotal_time[actveh]=total_time
                    # # 每辆车等待时间
                    # vewait_totalt[actveh]=wait_totalt
                    #每辆车节点信息
                    venodes[actveh].append(vehicle_loc)
                    #每辆车总路程
                    redistance_return[actveh] += distance
                    # env.upmask(state)
                    actveh=neactveh

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_ns, b_r, b_d , b_ac, b_nac= replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d,'actveh': b_ac,'neactveh': b_nac}
                        dqn_loss=agent.update(transition_dict)
                        dqn_losses.append(dqn_loss.detach().cpu().numpy())

                # transition_dict['actveh'].pop()
                # 保存最大epoisde奖励的参数
                if maxre < episode_return:
                    maxre = episode_return
                    agent.save()
                # 添加标量画图
                writer.add_scalar(tag="reward", scalar_value=episode_return,
                                  global_step=i * num_episodes/10+ i_episode)
                return_list.append(episode_return)

                # actor_loss,critic_loss=agent.update(transition_dict)
                # writer_acloss.add_scalar(tag="actor_loss", scalar_value=actor_loss,
                #                          global_step=i * num_episodes / 10 + i_episode)
                # writer_crloss.add_scalar(tag="critic_loss", scalar_value=critic_loss,
                #                          global_step=i * num_episodes / 10 + i_episode)
                # actor_losses.append(actor_loss.detach().cpu().numpy())
                # critic_losses.append(critic_loss.detach().cpu().numpy())
                # 每个车辆的等待时间
                v_wait.append(wait_totalt)
                # 每个车辆的总时间
                v_time.append(total_time)
                # 每个车辆的总路程
                v_distance.append(redistance_return)
                # 每个车辆的路径
                for k in range(env.vehicle):
                    v_list.extend(venodes[k])
                # v_list.extend(venodes)
                # 所有车辆的等待时间
                allv_wait.append(np.sum(wait_totalt))
                # 所有车辆的总时间
                allv_time.append(np.sum(total_time))
                # 所有车辆的最短长耗时
                allv_maxt.append(np.max(total_time))
                # 所有车辆的总路程
                allv_distance.append(np.sum(redistance_return))
                #更新进度条
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                # 每个车辆的等待时间
                data1 = pd.DataFrame(np.array(v_wait))
                data1.to_csv('v_wait1.csv', mode='a', index=False, header=False)
                # totime_list1 = np.array(total_time1)
                # 每个车辆的总时间
                data2 = pd.DataFrame(np.array(v_time))
                data2.to_csv('v_time1.csv',mode='a',index=False,header=False)
                # 每个车辆的总路程
                # waittime_list1 = np.array(wait_totalt1)
                data3 = pd.DataFrame(np.array(v_distance))
                data3.to_csv('v_distance1.csv',mode='a',index=False,header=False)
                # 每个车辆的路径
                # nodes_list1 = np.array(nodes1)
                data4 = pd.DataFrame(np.array(v_list).reshape(1, -1))
                data4.to_csv('v_list1.csv',mode='a',index=False,header=False)
                # 所有车辆的等待时间
                # distance_list1 = np.array(distance_return1)
                data5 = pd.DataFrame(np.array(allv_wait))
                data5.to_csv('allv_wait1.csv',mode='a',index=False,header=False)
                # 所有车辆的总时间
                data6 = pd.DataFrame(np.array(allv_time))
                data6.to_csv('allv_time1.csv', mode='a', index=False, header=False)
                # 所有车辆的总路程
                data7 = pd.DataFrame(np.array(allv_distance))
                data7.to_csv('allv_distance1.csv', mode='a', index=False, header=False)
                # 所有车辆的总时间
                data8 = pd.DataFrame(np.array(allv_maxt))
                data8.to_csv('allv_maxt1.csv', mode='a', index=False, header=False)

    return return_list,dqn_losses,allv_time,allv_wait,allv_distance

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

#优势估计
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)