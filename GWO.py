import copy
from math import*
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import random
import time
import torch
import torch.nn as nn
startt=time.perf_counter()
# citys = [[12,26],[55,28],[76,81],[82,13],[42,64],[58,35],[84,92],[17,51],[87,34],[92,83],[67,63],[77,88],[24,24],[59,68],[43,10],[64,21],[19,17],[41,98],[91,44],[98,67],[25,29],[99,50],[23,94],[4,61],[20,32],[66,77],[13,57],[97,37],[57,33],[62,9],[22,85],[38,70],[37,96],[44,100],[35,11],[18,86],[33,58],[27,47],[83,27],[79,5],[80,65],[88,20],[49,56],[30,41],[89,16],[15,46],[14,74],[53,71],[93,38],[74,55],[60,97],[51,12],[40,49],[86,6],[72,66],[11,80],[5,54],[81,52],[31,73],[8,89],[95,91],[90,42],[34,79],[28,4],[47,43],[69,40],[85,53],[50,69],[3,76],[21,95],[94,31],[65,72],[78,93],[46,19],[63,1],[9,30],[100,48],[26,3],[52,18],[1,36],[10,59],[48,75],[68,62],[54,87],[16,22],[36,45],[61,78],[75,82],[7,84],[96,14],[73,2],[39,23],[2,15],[29,99],[6,90],[70,25],[45,39],[32,8],[71,7],[56,60]]
class Vrp_Env():
    #读取节点文件(节点id及其相应位置id)
    #(13,3)
    node_sets = pd.read_csv("datasets/node1.csv").fillna(0).astype(int).values
    # print(node_sets)
    # print(node_sets.shape)

    #读取时间坐标文件(每个节点开始、结束时间和x、y坐标)
    #(25，5)
    ticoor_sets = pd.read_csv("datasets/ticoor1.csv").values
    # print(ticoor_sets[2][3])

    #读取订单文件(送货节点、收货节点、货物量)
    #(9,4)
    order_sets = pd.read_csv("datasets/order1.csv")[:-1].values
    # 读取车速文件(开始时间、结束时间、车速)
    # (7，3)
    speed_sets = pd.read_csv("datasets/speed.csv").values
    # print(order_sets)
    # print(ticoor_sets[node_sets[0][0]][3])
    # print(ticoor_sets[node_sets[0][0]][4])
    def __init__(self):
        # 网点数量
        self.depot_num = 1
        # 订单数量
        self.order_num=self.order_sets.shape[0]
        # 货主数量
        self.huozhu_num = 25
        # 客户数量
        self.customer = self.huozhu_num * 2
        # 所有节点数量
        self.all_num = self.depot_num + self.huozhu_num + self.customer
        #订单下标id
        self.order_index=self.order_sets[:,0]
        #时间位置索引
        self.ticoor_index = self.ticoor_sets[:, 0]
        # 所有节点仓库数[节点，仓库]
        self.pd_node_num = np.ones((self.all_num, 1), dtype=int)
        for index, row in pd.read_csv("datasets/order1.csv").iterrows():
            num_columns = sum(row.notna())  # 计算非空值的数量
            self.pd_node_num[index] = num_columns - 1
        #车辆初始位置在网点
        self.x = self.ticoor_sets[self.node_sets[0][0]][3]  # 记录当前智能体位置的横坐标
        self.y = self.ticoor_sets[self.node_sets[0][0]][4]  # 记录当前智能体位置的纵坐标
        self.vehicle_loc = np.array([self.x, self.y])
env=Vrp_Env()

def speed(cur_time):
    for i in range(env.speed_sets.shape[0]):
        if env.speed_sets[i][0]<=cur_time<env.speed_sets[i][1]:
            ve_speed=env.speed_sets[i][2]
            break
    return ve_speed

def eucli(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist

n1=1+env.huozhu_num+env.customer
D=np.zeros([n1,n1])
node_loc=np.zeros([n1,4])
# 当前节点所有仓库与其他所有节点得所有仓库距离均值
for i in range(n1):
    sum=0
    count=0
    for j in range(n1):
        if i != j:
            case_cur_pdnum2 = env.pd_node_num[j][0]
            for pd_num2 in range(case_cur_pdnum2):  # 判断是符合哪个仓库得时间窗
                node_idx2 = env.node_sets[j][pd_num2 + 1]
                case_cur_pdnum2 = env.pd_node_num[j][0]  # 当前节点仓库数量
                node_locx2 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][3]
                node_locy2 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx2)[0][0]][4]
                node_loc2 = np.array([node_locx2, node_locy2])
                case_cur_pdnum1 = env.pd_node_num[i][0]
                for pd_num1 in range(case_cur_pdnum1):  # 判断是符合哪个仓库得时间窗
                    node_idx1 = env.node_sets[i][pd_num1 + 1]
                    case_cur_pdnum1 = env.pd_node_num[i][0]  # 当前节点仓库数量
                    node_locx1 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][3]
                    node_locy1 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][4]
                    node_loc1 = np.array([node_locx1, node_locy1])
                    sum+= eucli(node_loc1, node_loc2)
                    count+=1
            D[i, j] = sum / count
        else:
            D[i, j] = 1e-4

#计算城市间的相互距离
def distance(routs,Flag_node,veh_num):
    nodes=[]
    node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][3]
    node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][4]
    vehicle_loc = []
    for k in range(veh_num):
        vehicle_loc.append([node_locx, node_locy])
    vehicle_loc = np.array(vehicle_loc)
    nodes.append(0)
    cur_time = np.ones(veh_num)*8
    total_time = np.zeros(veh_num)
    wait_totalt = np.zeros(veh_num)
    total_len = np.zeros(veh_num)
    i=0
    j=-1
    cur_vehicle=Flag_node[i]
    for node in routs:
        case_cur_pdnum = env.pd_node_num[node][0]  # 当前节点仓库数量
        flag = 0
        for pd_num in range(case_cur_pdnum):  # 判断是符合哪个仓库得时间窗
            node_idx = env.node_sets[node][pd_num + 1]  # 第1或2、3、4个仓库
            node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx)[0][0]][3]
            node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx)[0][0]][4]
            node_loc = np.array([node_locx, node_locy])
            distances = eucli(vehicle_loc[cur_vehicle], node_loc)
            add_t = distances / speed(cur_time[cur_vehicle])
            t = (cur_time[cur_vehicle] + add_t) % 24.0  # 可能到第二天才去执行订单，从8点开始
            if (env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx)[0][0]][1] <= t <
                    env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx)[0][0]][2]):
                # 符合第一个位置的时间窗，前往第一个位置
                cur_time[cur_vehicle] = t
                total_time[cur_vehicle] += add_t
                vehicle_loc[cur_vehicle] = node_loc
                total_len[cur_vehicle] += distances
                nodes.append(node_idx)
                break
            else:
                flag = flag + 1  # 标记不符合得仓库数量
        if flag == case_cur_pdnum:  # 4个仓库都不符合，等到第二天在第一个仓库位置进行访问,以时间窗为准
            node_idx1 = env.node_sets[node][1]
            node_locx1 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][3]
            node_locy1 = env.ticoor_sets[np.argwhere(env.ticoor_index == node_idx1)[0][0]][4]
            node_loc1 = np.array([node_locx1, node_locy1])
            distance1 = eucli(vehicle_loc[cur_vehicle], node_loc1)
            add_t1 = distance1 / speed(cur_time[cur_vehicle])
            t1 = (cur_time[cur_vehicle] + add_t1) % 24.0
            if t1 >= 0 and t1 <= 8:
                wait_time = 8 - t1
            else:
                wait_time = 24.0 - t1 + 8
            wait_totalt[cur_vehicle] += wait_time
            cur_time[cur_vehicle] = 8
            total_time[cur_vehicle] += add_t1 + wait_time
            vehicle_loc[cur_vehicle] = node_loc1
            total_len[cur_vehicle] += distance1
            nodes.append(node_idx1)

        i+=1
        j+=1
        if i<len(routs):#如果换车，前一辆车回到depot
            pre_vehicle = Flag_node[j]
            cur_vehicle = Flag_node[i]
            if cur_vehicle!=pre_vehicle:
                node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][3]
                node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][4]
                depot_loc = np.array([node_locx, node_locy])
                distances = eucli(vehicle_loc[pre_vehicle], depot_loc)
                add_t = distances / speed(cur_time[pre_vehicle])
                t = (cur_time[pre_vehicle] + add_t) % 24.0  # 可能到第二天才去执行订单
                cur_time[pre_vehicle] = t
                total_time[pre_vehicle] += add_t
                vehicle_loc[pre_vehicle] = depot_loc
                total_len[pre_vehicle] += distances
                nodes.append(0)
                nodes.append(0)
    node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][3]
    node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][4]
    depot_loc = np.array([node_locx, node_locy])
    distances = eucli(vehicle_loc[cur_vehicle], depot_loc)
    add_t = distances / speed(cur_time[cur_vehicle])
    t = (cur_time[veh_num-1] + add_t) % 24.0  # 可能到第二天才去执行订单
    cur_time[veh_num-1] = t
    total_time[veh_num-1] += add_t
    vehicle_loc[veh_num-1] = depot_loc
    total_len[veh_num-1] += distances
    nodes.append(0)
    alltotal_time=np.sum(total_time)
    allwait_totalt=np.sum(wait_totalt)
    alltotal_len=np.sum(total_len)
    maxve_time=np.max(total_time)


    return alltotal_time,allwait_totalt,alltotal_len,nodes,maxve_time,total_time,wait_totalt

#D初始化为所有节点距离，两个节点所有可能取均值
n1=1+env.huozhu_num+env.customer

iter_max = 85
n = 2 * env.order_num
veh_num = 8
Route_best = np.zeros([iter_max, n + veh_num * 2])
Totimes_best = np.zeros([iter_max, 1])
Tomu_best = np.zeros([iter_max, 1])
Length_best = np.zeros([iter_max, 1])
Watimes_best = np.zeros([iter_max, 1])
Vtotime_best = np.zeros([iter_max, veh_num])
Vwatime_best = np.zeros([iter_max, veh_num])
Route_best = Route_best.astype(np.int64)

def GWO(Route_best, Totimes_best, Length_best, Watimes_best, Tomu_best):
    # 初始化参数
    population_size = 30  # 群体规模（狼的数量）
    max_iterations = iter_max  # 最大迭代次数
    n_orders = env.order_num
    # n_nodes = n_orders * 2
    # veh_num = 8  # 车辆数量

    # 初始化狼群（解的集合）
    wolves = []  # 存储狼群（解）
    flags_nodes = []  # 存储每个狼对应的车辆分配

    # 初始化狼群
    for i in range(population_size):
        state = np.zeros(n_orders, dtype=int)  # 订单状态，0：未访问，1：取货完成，2：送货完成
        routes = []  # 每辆车的路径
        flags_node = []  # 节点对应的车辆标记
        assigned_orders = np.arange(n_orders)
        np.random.shuffle(assigned_orders)
        orders_per_vehicle = n_orders // veh_num
        remainder = n_orders % veh_num
        idx = 0
        for v in range(veh_num):
            route = []
            num_orders = orders_per_vehicle + (1 if v < remainder else 0)
            vehicle_orders = assigned_orders[idx:idx+num_orders]
            idx += num_orders
            for order_idx in vehicle_orders:
                route.append(env.order_sets[order_idx][1])  # 取货点
                route.append(env.order_sets[order_idx][2])  # 送货点
                flags_node.extend([v, v])
            routes.extend(route)
        wolves.append(routes)
        flags_nodes.append(flags_node)

    # 主循环
    for iter in range(max_iterations):
        total_times = np.zeros(population_size)
        wait_times = np.zeros(population_size)
        Length = np.zeros(population_size)
        nodes_list = []
        maxve_time = np.zeros(population_size)
        vtotal_time = np.zeros((population_size, veh_num))
        vwait_time = np.zeros((population_size, veh_num))
        fitness = np.zeros(population_size)

        for i in range(population_size):
            Route = wolves[i]
            Flag_node = flags_nodes[i]
            total_times[i], wait_times[i], Length[i], nodes, maxve_time[i], vtotal_time[i], vwait_time[i] = distance(Route, Flag_node, veh_num)
            mu = total_times[i] + 20 * (np.max(vtotal_time[i]) - np.min(vtotal_time[i]))
            fitness[i] = mu
            nodes_list.append(nodes)

        # 找到alpha、beta、delta狼
        sorted_indices = np.argsort(fitness)
        alpha_idx = sorted_indices[0]
        beta_idx = sorted_indices[1]
        delta_idx = sorted_indices[2]

        alpha_wolf = wolves[alpha_idx]
        beta_wolf = wolves[beta_idx]
        delta_wolf = wolves[delta_idx]
        alpha_flags = flags_nodes[alpha_idx]
        beta_flags = flags_nodes[beta_idx]
        delta_flags = flags_nodes[delta_idx]

        # 更新狼的位置
        a = 2 - iter * (2 / max_iterations)  # a从2线性减小到0

        for i in range(population_size):
            if i in [alpha_idx, beta_idx, delta_idx]:
                continue
            X = wolves[i]
            X_flags = flags_nodes[i]
            X_new = []
            X_new_flags = []
            for pos in range(len(X)):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                r1 = np.random.rand()
                r2 = np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                r1 = np.random.rand()
                r2 = np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                # 从alpha、beta、delta狼中选择节点
                D_alpha = abs(C1 * alpha_wolf[pos] - X[pos])
                D_beta = abs(C2 * beta_wolf[pos] - X[pos])
                D_delta = abs(C3 * delta_wolf[pos] - X[pos])
                X1 = alpha_wolf[pos] - A1 * D_alpha
                X2 = beta_wolf[pos] - A2 * D_beta
                X3 = delta_wolf[pos] - A3 * D_delta
                X_new_pos = (X1 + X2 + X3) / 3

                # 由于我们的节点是整数，需要取整并处理越界情况
                X_new_pos = int(round(X_new_pos))
                if X_new_pos < 0:
                    X_new_pos = 0
                elif X_new_pos >= n1:
                    X_new_pos = n1 - 1

                X_new.append(X_new_pos)
                # 对应的车辆标记也进行更新
                X_new_flags.append(X_flags[pos])

            # 修复解，确保没有重复的节点，并满足取送货顺序
            X_new_fixed = []
            X_new_flags_fixed = []
            visited_pickups = set()
            visited_deliveries = set()
            for idx, node in enumerate(X_new):
                if node in env.order_sets[:,1]:  # 取货点
                    order_idx = np.where(env.order_sets[:,1] == node)[0][0]
                    if order_idx not in visited_pickups:
                        X_new_fixed.append(node)
                        X_new_flags_fixed.append(X_new_flags[idx])
                        visited_pickups.add(order_idx)
                elif node in env.order_sets[:,2]:  # 送货点
                    order_idx = np.where(env.order_sets[:,2] == node)[0][0]
                    if order_idx in visited_pickups and order_idx not in visited_deliveries:
                        X_new_fixed.append(node)
                        X_new_flags_fixed.append(X_new_flags[idx])
                        visited_deliveries.add(order_idx)
            # 补充遗漏的取送货点
            for order_idx in range(n_orders):
                if order_idx not in visited_pickups:
                    X_new_fixed.append(env.order_sets[order_idx][1])
                    X_new_flags_fixed.append(np.random.randint(0, veh_num))
                if order_idx not in visited_deliveries:
                    X_new_fixed.append(env.order_sets[order_idx][2])
                    X_new_flags_fixed.append(np.random.randint(0, veh_num))

            wolves[i] = X_new_fixed
            flags_nodes[i] = X_new_flags_fixed

        # 更新最佳解
        min_mu = np.min(fitness)
        min_index = np.argwhere(fitness == min_mu)[0][0]
        Tomu_best[iter] = min_mu
        Totimes_best[iter] = total_times[min_index]
        Watimes_best[iter] = wait_times[min_index]
        Length_best[iter] = Length[min_index]
        # 确保分配的数据维度一致
        min_len = min(len(Route_best[iter]), len(nodes_list[min_index]))  # 找到较小的长度进行操作

        # 如果 Route_best 的当前迭代行长度不足以容纳 nodes_list[min_index]，则扩展其大小
        if len(Route_best[iter]) < len(nodes_list[min_index]):
            # 动态扩展 Route_best 数组
            Route_best = np.resize(Route_best, (Route_best.shape[0], len(nodes_list[min_index])))
        Route_best[iter,:len(nodes_list[min_index])] = nodes_list[min_index]
        Vtotime_best[iter] = vtotal_time[min_index]
        Vwatime_best[iter] = vwait_time[min_index]

        endt = time.perf_counter()
        print("目标值", Tomu_best[iter])
        print("最短时间per", Totimes_best[iter])
        print('每辆车总时间：', Vtotime_best[iter])
        print('每辆车等待时间：', Vwatime_best[iter])
        print("运行耗时per", endt - startt)
        print(iter+1)

    return Totimes_best,Route_best,Watimes_best,Length_best,Vtotime_best,Vwatime_best,Tomu_best

# 结果显示
Totimes_best,Route_best,Watimes_best,Length_best,Vtotime_best,Vwatime_best,Tomu_best=GWO(Route_best,Totimes_best,Length_best,Watimes_best,Tomu_best)
Shortest_mu=np.min(Tomu_best)
index = np.argwhere(Tomu_best==Shortest_mu)[0][0]

Shortest_Totimes= Totimes_best[index]
Shortest_Route=Route_best[index,:]
Shortest_Watimes = Watimes_best[index]
Shortest_Length = Length_best[index]
Shortest_Vtotime=Vtotime_best[index]
Shortest_Vwatime=Vwatime_best[index]
end=time.perf_counter()
print("运行耗时CoTime", end-startt)
print("目标值", Shortest_mu)
print('最短总时间：',Shortest_Totimes)
print('最短路径：',Shortest_Route)
print('等待时间：',Shortest_Watimes)
print('总长度Length：',Shortest_Length)
print('时间AllTime：',Shortest_Vtotime)
print('等待时间WaitTime：',Shortest_Vwatime)

def extract_subarrays(arr):
    subarrays = []
    start_index = 0
    end_index = 0

    for i in range(len(arr)):
        if arr[i] == 0:
            if start_index != end_index:
                subarray = arr[start_index-1:end_index+1]
                subarrays.append(subarray)
            start_index = i + 1
            end_index = i + 1
        else:
            end_index += 1

    return subarrays

#绘制结果
plt.figure(figsize=(8,6),dpi=450)
plt.title('GWO_VRP')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(env.ticoor_sets[...,3],env.ticoor_sets[...,4],'ko',ms = 3)
xbests = [[] for _ in range(veh_num)]
subarrays = extract_subarrays(Shortest_Route)
for j, subarray in enumerate(subarrays):
    for i in subarray:
        xbests[j].append(np.argwhere(env.ticoor_index == i)[0][0])
# aa=env.ticoor_sets[xbests[5]][4]
colors = ['red', 'blue', 'green','purple','pink']
for k in range(veh_num):
    plt.plot(env.ticoor_sets[xbests[k],3],env.ticoor_sets[xbests[k],4],colors[k])
# plt.plot([citys[xbest[-1],0],citys[xbest[0],0]],[citys[xbest[-1],1],citys[xbest[0],1]],ms = 2)
plt.legend(['All Points','Route 1', 'Route 2', 'Route 3', 'Route 4', 'Route 5'])
plt.savefig('testblueline.jpg')
plt.show()
