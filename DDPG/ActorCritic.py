import numpy as np
import pandas as pd
import torch.nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
# import scipy.sparse as sp
import collections
import random
#环境
from env import Vrp_Env
from network5 import *
#定义策略网络
import trainer

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#残差、输入数据
#归一化[0,1]
def data_normal_2d01(orign_data):
    dim = 2
    d_min = torch.min(orign_data, dim=dim)[0]
    for i, en in enumerate(d_min):
        for idx, j in enumerate(en):
            if j < 0:
                orign_data[i, idx, :] += torch.abs(d_min[i][idx])
                d_min = torch.min(orign_data, dim=dim)[0]
    d_max = torch.max(orign_data, dim=dim)[0]
    d_max = d_max.squeeze(0)
    dst = d_max - d_min

    if d_min.shape[1] == orign_data.shape[1]:
        d_min = d_min.unsqueeze(2)
        dst = dst.unsqueeze(2)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data, d_min).true_divide(dst)
    for i, en in enumerate(dst):
        for idx, j in enumerate(en):
            if j == 0:
                norm_data[i, idx, :] = orign_data[i, idx, :] / 2.0
    return norm_data
#加softmax转换为概率；全连接层搞复杂，加隐藏层（先放大在缩小 128 256 128） ；加两层贝叶斯卷积神经网络（杨赛塞）表征能力；A2C、DQN;画图matlab、pygame,车星星
class PolicyNet(torch.nn.Module):
    def __init__(self,action_dim):
        super(PolicyNet,self).__init__()
        self.model=Transformer(action_dim,action_dim,0,0).to(device)
        self.device = device
    def forward(self, x0, x1, x2,k):
        if x0.size(0)==4:
            x0 = x0.unsqueeze(0)
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
        env.upmask(x2,k)
        x0 = data_normal_2d01(x0)
        x1 = data_normal_2d01(x1)
        x2 = data_normal_2d01(x2)
        trg = torch.cat([x1, x2], 1)
        out = self.model(x0, x1,x2)
        out = out.view(-1, action_dim)
        mask = torch.tensor(env.mask, dtype=torch.float).to(self.device)
        # mask=x+mask.log()
        #有可能当前车辆没有可访问节点
        if torch.allclose(mask, torch.zeros_like(mask)):  # 检查掩码是否全为零
            probs = torch.zeros_like(mask)  # 返回全为零的张量作为结果
        else:
            probs = F.softmax(out + mask.log(), dim=1)  # 概率P
        return probs
#定义价值网络001112
class ValueNet(torch.nn.Module):
    def __init__(self,action_dim):
        super(ValueNet,self).__init__()
        self.model=Transformer(action_dim,action_dim,0,0).to(device)
        self.fc = torch.nn.Linear(action_dim, 1)
        self.device = device
    def forward(self,x0,x1,x2,actions):
        x0 = data_normal_2d01(x0)
        x1 = data_normal_2d01(x1)
        x2 = data_normal_2d01(x2)
        trg = torch.cat([x1, x2,actions], 1)
        out = self.model(x0, x1,x2)
        out = out.view(-1, action_dim)
        return self.fc(out)
#定义算法
class ActorCritic:
    def __init__(self,status_dim,action_dim,actor_lr,critic_lr,sigma, epochs, tau, gamma, device):
        #策略网络
        self.actor=PolicyNet(action_dim).to(device)
        # 价值网络
        self.critic = ValueNet(action_dim).to(device)

        self.target_actor = PolicyNet(action_dim).to(device)
        self.target_critic = ValueNet(action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 策略网络优化器
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        # 价值网络优化器
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma=gamma

        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim


        # self.lmbda = lmbda
        # self.epochs = epochs  # 一条序列的数据用来训练轮数
        # self.eps = eps  # PPO中截断范围的参数
        self.device = device
    def take_action(self,state,actveh):
        #判断取（负载），送货（车对应货）
        state0=state[0]
        state0 = torch.tensor(state0,dtype=torch.float).to(self.device)
        # print(state0)
        state1 = state[1]
        state1 = torch.tensor(state1, dtype=torch.float).to(self.device)
        # print(state1)
        state2 = state[2]
        state2 = torch.tensor(state2, dtype=torch.float).to(self.device)
        # print(state2)
        prob = self.actor(state0[actveh], state1[actveh].reshape(1, -1), state2, actveh)
        action_dist = torch.distributions.Categorical(prob)
        action = action_dist.sample()
        # action_dist=torch.distributions.Categorical(prob)
        # action=action_dist.sample()

        #取样返回的是该位置的索引
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self,transition_dict):
        states=transition_dict['states']
        states = np.array(states)
        states.reshape(len(states), -1)
        states0 = states[:, 0].tolist()
        states0 = torch.tensor(states0, dtype=torch.float).to(self.device)
        states1 = states[:, 1].tolist()
        states1 = torch.tensor(states1, dtype=torch.float).to(self.device)
        states2 = states[:, 2].tolist()
        states2 = torch.tensor(states2, dtype=torch.float).to(self.device)
        actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_states = transition_dict['next_states']
        next_states = np.array(next_states)
        next_states.reshape(len(next_states), -1)
        next_states0 = next_states[:, 0].tolist()
        next_states0 = torch.tensor(next_states0, dtype=torch.float).to(self.device)
        next_states1 = next_states[:, 1].tolist()
        next_states1 = torch.tensor(next_states1, dtype=torch.float).to(self.device)
        next_states2 = next_states[:, 2].tolist()
        next_states2 = torch.tensor(next_states2, dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        actveh = torch.tensor(transition_dict['actveh']).view(-1, 1).to(self.device)
        neactveh = torch.tensor(transition_dict['neactveh']).view(-1, 1).to(self.device)
        # 下一状态的价值
        nestates0_v = torch.cat([next_states0[o, t] for o, t in enumerate(actveh)], dim=0)
        nestates1_v = torch.cat([next_states1[o, t] for o, t in enumerate(actveh)], dim=0)
        # neresult_v=self.critic(nestates0_v,torch.unsqueeze(nestates1_v, dim=1),next_states2)
        # td_target=rewards+self.gamma*neresult_v

        #当前状态的价值
        restates0_v = torch.cat([states0[o, t] for o, t in enumerate(actveh)], dim=0)
        restates1_v = torch.cat([states1[o, t] for o, t in enumerate(actveh)], dim=0)
        # result_v = self.critic(restates0_v,torch.unsqueeze(restates1_v, dim=1),states2)
        # td_delta=td_target-result_v  #td_delta=td_target-self.critic(states0,states1,states2)

        #add
        next_q_values = self.target_critic(nestates0_v,torch.unsqueeze(nestates1_v, dim=1),next_states2, self.target_actor(nestates0_v,torch.unsqueeze(nestates1_v, dim=1),next_states2,neactveh))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(restates0_v,torch.unsqueeze(restates1_v, dim=1),states2,actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(restates0_v,torch.unsqueeze(restates1_v, dim=1),states2, self.actor(restates0_v,torch.unsqueeze(restates1_v, dim=1),states2,actveh)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


        # advantage = trainer.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)
        # old_log_probs = torch.log(self.actor(restates0_v,torch.unsqueeze(restates1_v, dim=1),states2, actveh).gather(1, actions)).detach()
        # for _ in range(self.epochs):
        #     # 策略求log
        #     log_probs = torch.log(self.actor(restates0_v,torch.unsqueeze(restates1_v, dim=1),states2, actveh).gather(1, actions))  # 似然概率
        #     ratio = torch.exp(log_probs - old_log_probs)
        #     surr1 = ratio * advantage
        #     surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
        #     actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
        #     # 价值均方误差损失函数
        #     critic_loss = torch.mean(F.mse_loss(self.critic(restates0_v,torch.unsqueeze(restates1_v, dim=1),states2), td_target.detach()))
        #     # 清空梯度
        #     self.actor_optimizer.zero_grad()
        #     self.critic_optimizer.zero_grad()
        #     # 计算梯度
        #     actor_loss.backward()
        #     critic_loss.backward()
        #     # 更新参数
        #     self.actor_optimizer.step()
        #     self.critic_optimizer.step()


        return actor_loss,critic_loss

    # 模型保存
    def save(self):
        torch.save(self.actor.state_dict(), 'models/net.pdparams')

    # 模型加载，只有权重
    def load(self):
        layer_state_dict = torch.load("models/net.pdparams")
        self.actor.load_state_dict(layer_state_dict)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, next_state, reward, done,actveh,neactveh):
        self.buffer.append((state, action, next_state, reward, done,actveh,neactveh))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done,actveh,neactveh = zip(*transitions)
        return state, action, next_state, reward, done,actveh,neactveh

    def size(self):
        return len(self.buffer)

#训练
actor_lr=1e-4
critic_lr=1e-3
gamma=0.99
final_gamma = 0.1
decay_rate = 0.001
sigma=0.01# 高斯噪声标准差
tau=0.005  # 软更新参数
epochs = 10
eps = 0.2
buffer_size = 10000
env= Vrp_Env()
# env.seed(0)
torch.manual_seed(0)
# dismu_dim=len(env.dismu)
# locmu_dim=len(env.locmu)
order_status_dim=len(env.order_status)
status_dim=np.stack((env.dismu[0][0],env.dismu[0][1],env.locmu[0],env.order_status), axis=0).shape[0]
# status_dim=4
#print(env.observation_space)
# print(vehicle_loc_dim)
# print(cur_load_dim)
# print(order_status_dim)
# print(action_dim)
# hidden_dim1=128
# hidden_dim2=256
action_dim=order_status_dim
# print(action_dim)
num_episodes=20000#200000
minimal_size = 1000
batch_size = 64

replay_buffer = ReplayBuffer(buffer_size)
agent=ActorCritic(status_dim,action_dim,actor_lr,critic_lr,sigma, epochs, tau, gamma, device)
# return_list=rl_utils.train_on_policy_agent(env,agent,num_episodes)
#agent.take_action(env.state)
return_list,actor_losses,critic_losses,totime_list,waittime_list,distance_list=trainer.train_off_policy_agent(env,agent,num_episodes,replay_buffer, minimal_size, batch_size)
# actor_losses=actor_losses.detach().numpy()
# critic_losses=critic_losses.detach().numpy()
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, actor_losses)
plt.plot(episodes_list, critic_losses)
plt.xlabel('Episodes')
# plt.ylabel('Returns')
plt.legend(['actor_loss','critic_loss'],loc='upper left')
plt.title('PPO on {}'.format("VRP"))
plt.show()

plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format("VRP"))
plt.show()

mv_return = trainer.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format("VRP"))
plt.show()
# return_list=np.array(return_list)
# avgResult = np.average(return_list.reshape(-1, 10), axis=1)
# data1 = pd.DataFrame(avgResult)
# data1.to_csv('average_reward.csv')
#
# totime_list=np.array(totime_list)
# data2 = pd.DataFrame(totime_list)
# data2.to_csv('totime_list.csv')
#
# waittime_list=np.array(waittime_list)
# data3 = pd.DataFrame(waittime_list)
# data3.to_csv('waittime_list.csv')
#
# nodes_list=np.array(nodes_list)
# data4 = pd.DataFrame(nodes_list)
# data4.to_csv('nodes_list.csv')
#
# distance_list=np.array(distance_list)
# data5 = pd.DataFrame(distance_list)
# data5.to_csv('distance_list.csv')
