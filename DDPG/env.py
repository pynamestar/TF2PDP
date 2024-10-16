import copy
import numpy as np
import pandas as pd
import torch
import random

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


np.random.seed(0)
"""
    计算两个点之间的欧几里得距离
"""
# a=[]
def eucli(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist
"""
    计算当前时间车辆的速度
"""
def speed(cur_time):
    # self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][1] <= t1 <
    # self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][2]\
    for i in range(Vrp_Env.speed_sets.shape[0]):
        if Vrp_Env.speed_sets[i][0]<=cur_time<Vrp_Env.speed_sets[i][1]:
            ve_speed=Vrp_Env.speed_sets[i][2]
            break
    return ve_speed
class Vrp_Env():
    speed_sets = pd.read_csv("../DDPGV530-dim256/datasets/speed.csv").values
    def __init__(self):
        self.node_sets = pd.read_csv("../DDPGV530-dim256/datasets/node1.csv").fillna(0).astype(int).values
        self.ticoor_sets = pd.read_csv("../DDPGV530-dim256/datasets/ticoor1.csv").values
        self.order_sets = pd.read_csv("../DDPGV530-dim256/datasets/order1.csv").values
        #网点数量
        self.depot_num= 1
        #货主数量
        self.huozhu_num = 10
        #客户数量
        self.customer=self.huozhu_num*2
        #所有节点数量
        self.all_num=self.depot_num+self.huozhu_num+self.customer
        #最多仓库数
        self.max_ware=4
        # 车数量
        self.vehicle = 5
        ids3 = list(range(6, 20))
        self.max_load = np.random.choice(ids3, self.vehicle)
        # 订单数量
        self.order_num=self.order_sets.shape[0]
        #回车场的车辆数
        self.venums = 0
        #订单下标id
        self.order_index=self.order_sets[:,0]
        #时间位置索引
        self.ticoor_index = self.ticoor_sets[:, 0]
        #当前订单是否可以访问,初始化为1都可以访问
        self.mask=np.ones((self.vehicle, self.order_sets.shape[0]))  #np.repeat(1, self.order_sets.shape[0])
        # 当前订单是被那个车处理,未取送(订单状态为0):100,已取未送(订单状态为1):车下标012,已取送(订单状态为2):不变100
        self.flag = 100 * np.ones((self.vehicle, self.order_sets.shape[0]))
        for k in range(self.vehicle):
            self.flag[k][self.order_sets.shape[0]-1] = k
        # 系统总时间
        self.total_time = np.zeros(self.vehicle)#0  # 大于24h
        # # 当前等待时间
        # self.wait_time = 0
        # 等待总时间
        self.wait_totalt = np.zeros( self.vehicle)#0
        # 总行程
        self.total_len = np.zeros( self.vehicle)#0
        #车辆最大载货体积
        # self.max_load=20
        #车辆行驶速度前闭后开
        # self.vehicle_speed=20.0
        # 车辆开始行驶时间
        self.start_time = np.ones(self.vehicle) * 8
        # 系统当前时间
        self.cur_time = np.ones( self.vehicle)*8#0
        #车辆初始位置在网点
        self.x = self.ticoor_sets[self.node_sets[0][0]][3]  # 记录当前智能体位置的横坐标
        self.y = self.ticoor_sets[self.node_sets[0][0]][4]  # 记录当前智能体位置的纵坐标
        #每辆车的当前位置
        self.vehicle_loc = []
        for k in range(self.vehicle):
            self.vehicle_loc.append([self.x, self.y])
        self.vehicle_loc=np.array(self.vehicle_loc)
        # 每个订单是否被处理(未处理为0)
        self.order_status = np.repeat(0, self.order_num)
        self.order_status[self.order_num-1]=2
        self.order_sta = np.repeat(0, self.order_num)
        self.order_sta[self.order_num - 1] = 2
        #车辆当前负载
        self.cur_load = np.repeat(0, self.vehicle)
        # 车辆到每个订单当前目的地的距离,初始化为1
        mu = [[] for _ in range(self.vehicle)]   # [1,9,1,2,2,2,3,3,3]
        for k in range(self.vehicle):
            for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
                if element == 0:
                    mu[k].append(self.order_sets[i][1])
                if element == 1:
                    mu[k].append(self.order_sets[i][2])
                # if element == 2:
                #     mu[k].append(50)
        # ord_mu = sorted(mu)#[1,2,2,2,3,3,3,9,12]
        # only_mu = list(set(ord_mu)) #下一步可访问多有节点[1,2,3,9,12 ]
        #所有节点仓库数[节点，仓库]
        self.pd_node_num=np.ones((self.all_num,1 ),dtype=int)
        for index, row in pd.read_csv("../DDPGV530-dim256/datasets/node1.csv").iterrows():
            num_columns = sum(row.notna())  # 计算非空值的数量
            self.pd_node_num[index] = num_columns-1
            # print(f"行 {index + 1} 有 {num_columns} 列")
        self.dismu = np.zeros((self.vehicle,self.max_ware, self.order_sets.shape[0]))
        # self.dismu2 = np.repeat(1, self.order_sets.shape[0])
        for k in range(len(mu)):
            for i, element in enumerate(mu[k]):
                case_cur_pdnum=self.pd_node_num[element][0]
                for pd_num in range(case_cur_pdnum):
                    element_idx1 = self.node_sets[element][pd_num+1]
                    element_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][3]
                    element_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][4]
                    element_loc1 = np.array([element_locx1, element_locy1])
                    element_t1 = eucli(self.vehicle_loc[k], element_loc1) / speed(self.cur_time[k])
                    self.dismu[k][pd_num][i] = element_t1
        # 车辆运载量，换成可访问订单中，每个点的装载量(负数)、卸载量(正数)
        self.locmu =  np.zeros((self.vehicle,self.order_sets.shape[0]))
        for k in range(self.vehicle):
            for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
                if element == 0:
                    if self.order_sets[i][3] <= self.max_load[k] - self.cur_load[k]:
                        self.locmu[k][i]= -self.order_sets[i][3]
                if element == 1:
                    self.locmu[k][i]= self.order_sets[i][3]
                # if element == 2:
                #     self.locmu[k][i] = 8
        # 全部订单是否全部完成
        self.done = False

        # 输入的状态是系统当前时间、车辆当前位置、订单的处理状态、车辆负载
        # self.state = (self.cur_time,self.vehicle_loc,self.order_status,self.cur_load)
        # 输入的状态是订单的处理状态
        # self.state = self.vehicle_loc,np.array([self.cur_load]),self.order_status
        # a.append(self.state)
        # print(self.state)
        # self.state_dim=self.state.shape[0]

    """
        环境重置
    """
    def reset(self):
        # 随机选择一个i值，范围从1到80
        # choicei = random.randint(1, 80)  # choicei = random.randint(1, 80)
        # self.node_sets = pd.read_csv(f"../VRPV530-dim256/datasets/node{choicei}.csv").fillna(0).astype(int).values
        # self.ticoor_sets = pd.read_csv(f"../VRPV530-dim256/datasets/ticoor{choicei}.csv").values
        # self.order_sets = pd.read_csv(f"../VRPV530-dim256/datasets/order{choicei}.csv").values

        # 回车场的车辆数
        self.venums = 0
        # 时间位置索引
        self.ticoor_index = self.ticoor_sets[:, 0]
        # 当前订单是否可以访问,初始化为1都可以访问
        self.mask=np.ones((self.vehicle, self.order_sets.shape[0]))
        #当前订单是被那个车处理,未取送(订单状态为0):100,已取未送(订单状态为1):车下标012,已取送(订单状态为2):不变100
        self.flag = 100 * np.ones((self.vehicle, self.order_sets.shape[0]))
        for k in range(self.vehicle):
            self.flag[k][self.order_sets.shape[0]-1] = k
        # 车辆到每个订单当前目的地的距离,初始化为1
        # self.dismu = np.ones((2, self.order_sets.shape[0]))
        # # 车辆运载量，换成可访问订单中，每个点的装载量(负数)、卸载量(正数)
        # self.locmu = np.repeat(0, self.order_sets.shape[0])
        # self.locmu =np.zeros((1, self.order_sets.shape[0]))
        # 系统总时间
        self.total_time = np.zeros( self.vehicle)#0 #大于24h
        # # 当前等待时间
        # self.wait_time = 0
        # 等待总时间
        self.wait_totalt = np.zeros( self.vehicle)#0
        # 总行程
        self.total_len = np.zeros( self.vehicle)#0
        # 车辆开始行驶时间
        self.start_time = np.ones(self.vehicle) * 8
        # 系统当前时间
        self.cur_time = np.ones( self.vehicle)*8#0 #一天之内的时间
        # 车辆初始位置在网点
        self.x = self.ticoor_sets[self.node_sets[0][0]][3]  # 记录当前智能体位置的横坐标
        self.y = self.ticoor_sets[self.node_sets[0][0]][4]  # 记录当前智能体位置的纵坐标
        self.vehicle_loc = []
        for k in range(self.vehicle):
            self.vehicle_loc.append([self.x, self.y])
        self.vehicle_loc = np.array(self.vehicle_loc)
        # 每个订单是否被处理(未处理为0)
        self.order_status = np.repeat(0, self.order_num)
        self.order_status[self.order_num - 1] = 2
        self.order_sta = np.repeat(0, self.order_num)
        self.order_sta[self.order_num - 1] = 2
        # 车辆当前负载
        self.cur_load = np.repeat(0, self.vehicle)
        # 车辆到每个订单当前目的地的距离,初始化为1
        mu = [[] for _ in range(self.vehicle)]  # [1,9,1,2,2,2,3,3,3]
        for k in range(self.vehicle):
            for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
                if element == 0:
                    mu[k].append(self.order_sets[i][1])
                if element == 1:
                    mu[k].append(self.order_sets[i][2])

        # 所有节点仓库数[节点，仓库]
        # self.pd_node_num = np.ones((self.all_num, 1), dtype=int)
        # for index, row in pd.read_csv(f"../VRPV530-dim256/datasets/node{choicei}.csv").iterrows():
        #     num_columns = sum(row.notna())  # 计算非空值的数量
        #     self.pd_node_num[index] = num_columns - 1
            # print(f"行 {index + 1} 有 {num_columns} 列")

        self.dismu = np.zeros((self.vehicle, self.max_ware, self.order_sets.shape[0]))
        # self.dismu2 = np.repeat(1, self.order_sets.shape[0])
        for k in range(len(mu)):
            for i, element in enumerate(mu[k]):
                case_cur_pdnum = self.pd_node_num[element][0]
                for pd_num in range(case_cur_pdnum):
                    element_idx1 = self.node_sets[element][pd_num + 1]
                    element_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][3]
                    element_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][4]
                    element_loc1 = np.array([element_locx1, element_locy1])
                    element_t1 = eucli(self.vehicle_loc[k], element_loc1) / speed(self.cur_time[k])
                    self.dismu[k][pd_num][i] = element_t1

        # 车辆运载量，换成可访问订单中，每个点的装载量(负数)、卸载量(正数)
        self.locmu = np.zeros((self.vehicle, self.order_sets.shape[0]))
        for k in range(self.vehicle):
            for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
                if element == 0:
                    if self.order_sets[i][3] <= self.max_load[k] - self.cur_load[k]:
                        self.locmu[k][i] = -self.order_sets[i][3]
                if element == 1:
                    self.locmu[k][i] = self.order_sets[i][3]
                # if element == 2:
                #     self.locmu[k][i] = 8
        # 全部订单是否全部完成
        self.done = False

        # 输入的状态是系统当前时间、订单的处理状态、车辆当前位置、车辆负载
        # self.state = self.vehicle_loc,np.array([self.cur_load]),self.order_status
        return self.dismu,self.locmu,self.order_sta.reshape(1,self.order_num)


    """
        根据动作返回下一步的转态、奖励
    """
    def step(self,action,actveh):
        #传入的action是即将要处理的订单id
        #待送状态，去货主位置取货
        # print("self.order_status[action]",self.order_status[action])
        # distance = 0

        distance = 0
        if (self.order_status[action] == 0):
            node = self.order_sets[action][1]  # 节点
            case_cur_pdnum = self.pd_node_num[node][0]#当前节点仓库数量
            flag=0
            for pd_num in range(case_cur_pdnum):#判断是符合哪个仓库得时间窗
                node_idx = self.node_sets[node][pd_num + 1]# 第1或2、3、4个仓库
                node_locx = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][3]
                node_locy = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][4]
                node_loc = np.array([node_locx, node_locy])
                distances = eucli(self.vehicle_loc[actveh], node_loc)
                add_t = distances / speed(self.cur_time[actveh])
                t = (self.cur_time[actveh] + add_t) % 24.0  # 可能到第二天才去执行订单，从8点开始
                if (self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][1] <= t <
                        self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][2]):
                    # 符合第一个位置的时间窗，前往第一个位置
                    self.cur_time[actveh] = t
                    self.total_time[actveh] += add_t
                    self.vehicle_loc[actveh] = node_loc
                    # print("node_loc1",self.vehicle_loc)
                    self.order_status[action] = 1  # 取货完成为在运状态
                    self.order_sta = copy.deepcopy(self.order_status)
                    for k in range(self.vehicle):
                        self.flag[k][action] = actveh
                    self.cur_load[actveh] += self.order_sets[action][3]
                    reward = -add_t
                    distance = distances
                    nodes = node_idx
                    break
                else:
                    flag=flag+1 #标记不符合得仓库数量
            if flag==case_cur_pdnum:#4个仓库都不符合，等到第二天在第一个仓库位置进行访问,以时间窗为准
                node_idx1 = self.node_sets[node][1]
                node_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][3]
                node_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][4]
                node_loc1 = np.array([node_locx1, node_locy1])
                distance1 = eucli(self.vehicle_loc[actveh], node_loc1)
                add_t1 = distance1 / speed(self.cur_time[actveh])
                t1 = (self.cur_time[actveh] + add_t1) % 24.0
                if t1>=0 and  t1<=8:
                    wait_time = 8 - t1
                else:
                    wait_time = 24.0 - t1+8
                self.wait_totalt[actveh] += wait_time
                self.cur_time[actveh] = 8
                self.total_time[actveh] += add_t1 + wait_time
                self.vehicle_loc[actveh] = node_loc1
                self.order_status[action] = 1  # 取货完成为在运状态
                self.order_sta = copy.deepcopy(self.order_status)
                for k in range(self.vehicle):
                    self.flag[k][action] = actveh
                self.cur_load[actveh] += self.order_sets[action][3]
                reward = -(add_t1 + wait_time)
                distance = distance1
                nodes = node_idx1
        elif (self.order_status[action] == 1):
            node = self.order_sets[action][2]  # 节点
            if node == 0:
                node_idx = self.node_sets[node][1]
                node_locx = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][3]
                node_locy = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][4]
                node_loc = np.array([node_locx, node_locy])
                distances = eucli(self.vehicle_loc[actveh], node_loc)
                add_t = distances / speed(self.cur_time[actveh])
                t = (self.cur_time[actveh] +add_t) % 24.0  # 可能到第二天才去执行订单
                self.cur_time[actveh] = t
                self.total_time[actveh] += add_t
                self.vehicle_loc[actveh] = node_loc
                self.order_status[action] = 2  # 返回网点任务完成
                self.order_sta = copy.deepcopy(self.order_status)
                # 选过车辆不能再选择
                self.flag[actveh][action] = 100
                self.cur_load[actveh] -= self.order_sets[action][3]
                reward = -add_t
                distance = distances
                nodes = node_idx
            else:
                case_cur_pdnum = self.pd_node_num[node][0]  # 当前节点仓库数量
                flag = 0
                for pd_num in range(case_cur_pdnum):  # 判断是符合哪个仓库得时间窗
                    node_idx = self.node_sets[node][pd_num + 1]  # 第1或2、3、4个仓库
                    node_locx = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][3]
                    node_locy = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][4]
                    node_loc = np.array([node_locx, node_locy])
                    distances = eucli(self.vehicle_loc[actveh], node_loc)
                    add_t = distances / speed(self.cur_time[actveh])
                    t = (self.cur_time[actveh] + add_t) % 24.0  # 可能到第二天才去执行订单
                    if (self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][1] <= t <
                            self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx)[0][0]][2]):
                        # 符合第一个位置的时间窗，前往第一个位置
                        self.cur_time[actveh] = t
                        self.total_time[actveh] += add_t
                        self.vehicle_loc[actveh] = node_loc
                        # print("node_loc1",self.vehicle_loc)
                        self.order_status[action] = 2  # 取货完成为在运状态
                        self.order_sta = copy.deepcopy(self.order_status)
                        for k in range(self.vehicle):
                            self.flag[k][action] = actveh
                        self.cur_load[actveh] -= self.order_sets[action][3]
                        reward = -add_t
                        distance = distances
                        nodes = node_idx
                        break
                    else:
                        flag = flag + 1  # 标记不符合得仓库数量
                if flag == case_cur_pdnum:  # 4个仓库都不符合，等到第二天在第一个仓库位置进行访问,以时间窗为准
                    node_idx1 = self.node_sets[node][1]
                    node_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][3]
                    node_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == node_idx1)[0][0]][4]
                    node_loc1 = np.array([node_locx1, node_locy1])
                    distance1 = eucli(self.vehicle_loc[actveh], node_loc1)
                    add_t1 = distance1 / speed(self.cur_time[actveh])
                    t1 = (self.cur_time[actveh] + add_t1) % 24.0
                    if t1 >= 0 and t1 <= 8:
                        wait_time = 8 - t1
                    else:
                        wait_time = 24.0 - t1 + 8
                    self.wait_totalt[actveh] += wait_time
                    self.cur_time[actveh] = 8
                    self.total_time[actveh] += add_t1 + wait_time
                    self.vehicle_loc[actveh] = node_loc1
                    self.order_status[action] = 2  # 取货完成为在运状态
                    self.order_sta = copy.deepcopy(self.order_status)
                    for k in range(self.vehicle):
                        self.flag[k][action] = actveh
                    self.cur_load[actveh] -= self.order_sets[action][3]
                    reward = -(add_t1 + wait_time)
                    distance = distance1
                    nodes = node_idx1

        # venums=0是否所有订单都访问完
        if all(item == 2 for item in self.order_status) and self.venums<=self.vehicle:
            self.order_status[self.order_num-1]=1
            self.order_sta = copy.deepcopy(self.order_status)
            self.venums+=1
        if self.venums==self.vehicle+1:
            self.order_status[self.order_num - 1] == 2
            self.order_sta = copy.deepcopy(self.order_status)
            for k in range(self.vehicle):
                self.flag[k][self.order_sets.shape[0] - 1] = k
            done = True
        else:
            done = False

        # 车辆地址换成到目的地的距离，当前车辆位置到其他可访问订单目的地的距离
        mu = [[] for _ in range(self.vehicle)]  # [1,9,1,2,2,2,3,3,3]
        for k in range(self.vehicle):
            for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
                if element == 0:
                    mu[k].append(self.order_sets[i][1])
                if element == 1:
                    mu[k].append(self.order_sets[i][2])
                # if element == 2:
                #     mu[k].append(50)
        # ord_mu = sorted(mu)#[1,2,2,2,3,3,3,9,12]
        # only_mu = list(set(ord_mu)) #下一步可访问多有节点[1,2,3,9,12 ]
        self.dismu = np.zeros((self.vehicle, self.max_ware, self.order_sets.shape[0]))
        # self.dismu2 = np.repeat(1, self.order_sets.shape[0])
        for k in range(len(mu)):
            for i, element in enumerate(mu[k]):
                case_cur_pdnum = self.pd_node_num[element][0]
                for pd_num in range(case_cur_pdnum):
                    element_idx1 = self.node_sets[element][pd_num + 1]
                    element_locx1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][3]
                    element_locy1 = self.ticoor_sets[np.argwhere(self.ticoor_index == element_idx1)[0][0]][4]
                    element_loc1 = np.array([element_locx1, element_locy1])
                    element_t1 = eucli(self.vehicle_loc[k], element_loc1) / speed(self.cur_time[k])
                    self.dismu[k][pd_num][i] = element_t1
        # 车辆运载量，换成可访问订单中，每个点的装载量(负数)、卸载量(正数)
        self.locmu = np.zeros((self.vehicle, self.order_sets.shape[0]))
        for k in range(self.vehicle):
            for i, element in enumerate(self.order_status):  # order_status[0 1 0 0 0 0 0 0 0]
                if element == 0:
                    if self.order_sets[i][3] <= self.max_load[k] - self.cur_load[k]:
                        self.locmu[k][i] = -self.order_sets[i][3]
                if element == 1:
                    self.locmu[k][i] = self.order_sets[i][3]
                # if element == 2:
                #     self.locmu[k][i] = 8
        # 可选择呢车辆
        vehicles = []
        seleted_vehicles = []
        for k in range(self.vehicle):
            self.upmask(self.order_sta.reshape(1, 1, -1), k)
            mask = torch.tensor(self.mask, dtype=torch.float).to(device)
            if torch.allclose(mask, torch.zeros_like(mask)):  # 检查掩码是否全为零
                vehicles.append(-1)
            else:
                vehicles.append(k)
                seleted_vehicles.append(k)
        # 从总时间数组中选择可选择车辆的消耗时间
        selected_times = self.total_time[seleted_vehicles]
        # 找到消耗时间最短的车辆的索引
        min_time_index = np.argmin(selected_times)
        # 找到消耗时间最短的车辆在可选择车辆列表中的索引
        next_actveh = seleted_vehicles[min_time_index]

        # 更新状态信息
        next_state = self.dismu,self.locmu,self.order_sta.reshape(1,self.order_num) #self.dismu,self.locmu.reshape(1,self.order_num),self.order_sta.reshape(1,self.order_num)
        return next_state, reward, done, self.total_time, self.wait_totalt, nodes, distance,next_actveh

    """
        每一个step后更新掩码
    """
    def upmask(self,state,kin):
        self.mask = np.random.randint(1,2, size=(state.shape[0],state.shape[2]))
        for i in range(state.shape[0]):
            if type(kin) != int:
                k=kin[i]
            else:
                k=kin
            for j in range(state.shape[2]):
                if state[i][0][j]==0:
                    if self.order_sets[j][3]> self.max_load[k]-self.cur_load[k]:
                        self.mask[i][j]=0
                if state[i][0][j]==1 and self.flag[k][j]!=k: #送货必须取过相应订单的货
                    self.mask[i][j]=0
                if state[i][0][j]==2:
                    self.mask[i][j]=0
        # for i in range(state.shape[0]):
        #     flag = 1 #订单完成
        #     for j in range(state.shape[2]-1):
        #         if self.mask[i][j] == 1:
        #             flag=0 #还有订单没完成
        #     if flag==0:#还有订单没完成
        #         self.mask[i][state.shape[2]-1] = 0#网点不可访问
        #     else:#所有订单完成
        #         self.mask[i][state.shape[2]-1] = 1 #网点可以访问
        return self.mask








#所有state是在环境里的，环境要写上初始状态信息
# next_state(时间(时间取余问题)、车辆位置、订单状态、车辆负载), reward, done = env.step(action)#约束时间选择节点哪个位置
# next_action = agent.take_action(next_state)#约束车辆负载，选择哪一个订单
# 点3：t1
# 点1：x1(时间限制),x2(时间限制)
if __name__=="__main__":
    env=Vrp_Env()
    print("订单数量:",env.order_num)
    print("订单下标id:",env.order_index)
    print("订单下标id:", env.ticoor_index)
    print("当前节点是否可以取/送货:", env.mask)
    print("当前时间:", env.cur_time)
    print("当前车辆位置:", env.vehicle_loc)
    print("当前订单状态:", env.order_status)
    print("当前车辆负载:", env.cur_load)
    # print("拼接:", env.state)
    # print("拼接:", len(env.state[2]))

    # a.append(env.state)
    # print("a",a)
    # s = np.array(a)
    # print("s", s)
    # s.reshape(len(s), -1)
    # print("s", s)
    # ss = s[:, 0].tolist()
    # print("ss", ss)
    # sss = torch.tensor(ss, dtype=torch.float)
    # print(sss.shape)
    # next_state, reward, done = env.step(5)
    # print(next_state)
    # print("当前时间:", env.cur_time)
    # print("总时间:", env.total_time)
    # print("当前车辆位置:", env.vehicle_loc)
    # print("当前订单状态:", env.order_status)
    # print("当前车辆负载:", env.cur_load)
    # print(next_state)
    # print(reward)
    # print(done)
    # print("mask前:",env.mask)
    # env.upmask((np.array([2, 3]), np.array([17]), np.array([0, 0, 0, 0, 0, 0, 0, 2, 2])))
    # print("mask后:", env.mask)