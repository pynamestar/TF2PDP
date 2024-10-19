import copy
from math import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import random
import time
import torch
import torch.nn as nn

startt = time.perf_counter()
'''
# 初始化参数
    FoodNumber = 30  # 食物源数量（蜂群规模）
    limit = 100  # 未改进次数限制
    MaxCycle = iter_max  # 最大迭代次数

    # 新增：定义 max_load
    ids3 = list(range(6, 20))
    max_load = np.random.choice(ids3, veh_num)
    
    # 读取节点文件(节点id及其相应位置id)
    node_sets = pd.read_csv("datasets/node2.csv").fillna(0).astype(int).values

    # 读取时间坐标文件(每个节点开始、结束时间和x、y坐标)
    ticoor_sets = pd.read_csv("datasets/ticoor2.csv").values

    # 读取订单文件(送货节点、收货节点、货物量)
    order_sets = pd.read_csv("datasets/order2.csv")[:-1].values

    # 读取车速文件(开始时间、结束时间、车速)
    speed_sets = pd.read_csv("datasets/speed.csv").values
    
    iter_max = 100
    veh_num = 5
    
'''


class Vrp_Env():
    # 读取节点文件(节点id及其相应位置id)
    node_sets = pd.read_csv("datasets/node2.csv").fillna(0).astype(int).values

    # 读取时间坐标文件(每个节点开始、结束时间和x、y坐标)
    ticoor_sets = pd.read_csv("datasets/ticoor2.csv").values

    # 读取订单文件(送货节点、收货节点、货物量)
    order_sets = pd.read_csv("datasets/order2.csv")[:-1].values

    # 读取车速文件(开始时间、结束时间、车速)
    speed_sets = pd.read_csv("datasets/speed.csv").values

    def __init__(self):
        # 网点数量
        self.depot_num = 1
        # 货主数量
        self.huozhu_num = 10
        # 客户数量
        self.customer = self.huozhu_num * 2
        # 所有节点数量
        self.all_num = self.depot_num + self.huozhu_num + self.customer
        # 订单数量
        self.order_num = self.order_sets.shape[0]
        # 订单下标id
        self.order_index = self.order_sets[:, 0]
        # 时间位置索引
        self.ticoor_index = self.ticoor_sets[:, 0]
        # 所有节点仓库数[节点，仓库]
        self.pd_node_num = np.ones((self.all_num, 1), dtype=int)
        for index, row in pd.read_csv("datasets/node2.csv").iterrows():
            num_columns = sum(row.notna())  # 计算非空值的数量
            self.pd_node_num[index] = num_columns - 1
        # 车辆初始位置在网点
        self.x = self.ticoor_sets[self.node_sets[0][0]][3]  # 记录当前智能体位置的横坐标
        self.y = self.ticoor_sets[self.node_sets[0][0]][4]  # 记录当前智能体位置的纵坐标
        self.vehicle_loc = np.array([self.x, self.y])


env = Vrp_Env()


def speed(cur_time):
    for i in range(env.speed_sets.shape[0]):
        if env.speed_sets[i][0] <= cur_time < env.speed_sets[i][1]:
            ve_speed = env.speed_sets[i][2]
            break
    return ve_speed


def eucli(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.sum(np.square(a - b)))
    return dist


n1 = 1 + env.huozhu_num + env.customer
D = np.zeros([n1, n1])
node_loc = np.zeros([n1, 4])

# 当前节点所有仓库与其他所有节点得所有仓库距离均值
for i in range(n1):
    total_distance_sum = 0
    count = 0
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
                    total_distance_sum += eucli(node_loc1, node_loc2)
                    count += 1
            D[i, j] = total_distance_sum / count
        else:
            D[i, j] = 1e-4


# 计算车辆行驶时间、等待时间等
def distance(routs, Flag_node, veh_num):
    nodes = []
    node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][3]
    node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][4]
    vehicle_loc = []
    for k in range(veh_num):
        vehicle_loc.append([node_locx, node_locy])
    vehicle_loc = np.array(vehicle_loc)
    nodes.append(0)
    cur_time = np.ones(veh_num) * 8
    total_time = np.zeros(veh_num)
    wait_totalt = np.zeros(veh_num)
    total_len = np.zeros(veh_num)
    i = 0
    j = -1
    cur_vehicle = Flag_node[i]
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
                cur_time[cur_vehicle] = t
                total_time[cur_vehicle] += add_t
                vehicle_loc[cur_vehicle] = node_loc
                total_len[cur_vehicle] += distances
                nodes.append(node_idx)
                break
            else:
                flag = flag + 1  # 标记不符合得仓库数量
        if flag == case_cur_pdnum:  # 所有仓库都不符合时间窗
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
        i += 1
        j += 1
        if i < len(routs):  # 如果换车，前一辆车回到depot
            pre_vehicle = Flag_node[j]
            cur_vehicle = Flag_node[i]
            if cur_vehicle != pre_vehicle:
                node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][3]
                node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][4]
                depot_loc = np.array([node_locx, node_locy])
                distances = eucli(vehicle_loc[pre_vehicle], depot_loc)
                add_t = distances / speed(cur_time[pre_vehicle])
                t = (cur_time[pre_vehicle] + add_t) % 24.0
                cur_time[pre_vehicle] = t
                total_time[pre_vehicle] += add_t
                vehicle_loc[pre_vehicle] = depot_loc
                total_len[pre_vehicle] += distances
                nodes.append(0)
                nodes.append(0)
    node_locx = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][3]
    node_locy = env.ticoor_sets[np.argwhere(env.ticoor_index == 0)[0][0]][4]
    depot_loc = np.array([node_locx, node_locy])
    distances = eucli(vehicle_loc[pre_vehicle], depot_loc)
    add_t = distances / speed(cur_time[pre_vehicle])
    t = (cur_time[veh_num - 1] + add_t) % 24.0
    cur_time[veh_num - 1] = t
    total_time[veh_num - 1] += add_t
    vehicle_loc[veh_num - 1] = depot_loc
    total_len[veh_num - 1] += distances
    nodes.append(0)
    alltotal_time = np.sum(total_time)
    allwait_totalt = np.sum(wait_totalt)
    alltotal_len = np.sum(total_len)
    maxve_time = np.max(total_time)
    return alltotal_time, allwait_totalt, alltotal_len, nodes, maxve_time, total_time, wait_totalt


n1 = 1 + env.huozhu_num + env.customer
iter_max = 500
n = 2 * env.order_num
veh_num = 5
Route_best = np.zeros([iter_max, n + veh_num * 2])  # 到该代为止最优的路线
Totimes_best = np.zeros([iter_max, 1])  # 到该代为止最小的路径距离
Tomu_best = np.zeros([iter_max, 1])  # 到该代为止最小的路径距离
Length_best = np.zeros([iter_max, 1])
Watimes_best = np.zeros([iter_max, 1])
Vtotime_best = np.zeros([iter_max, veh_num])
Vwatime_best = np.zeros([iter_max, veh_num])
Route_best = Route_best.astype(int)


def ABC(Route_best, Totimes_best, Length_best, Watimes_best, Tomu_best):
    # 初始化参数
    FoodNumber = 50  # 食物源数量（蜂群规模）
    limit = 100  # 未改进次数限制
    D = n  # 维度（需求节点数量）
    iter = 0  # 迭代次数的初值
    MaxCycle = iter_max  # 最大迭代次数

    # 新增：定义 max_load
    ids3 = list(range(6, 20))
    max_load = np.random.choice(ids3, veh_num)

    # 初始化食物源
    Foods = np.zeros((FoodNumber, D)).astype(int)
    state_list = []
    flags_node_list = []
    fitness = np.zeros(FoodNumber)
    trial = np.zeros(FoodNumber)
    for i in range(FoodNumber):
        state = np.zeros(env.order_num, dtype=int)
        flags_node = np.ones(D) * 100
        temp = np.random.choice([i for i in range(env.order_num)], 1)[0]
        start = env.order_sets[temp][1]
        state[temp] = 1
        flags_node[0] = 0
        flags_ding = np.ones(env.order_num) * 100
        flags_ding[temp] = 0
        cur_load = np.zeros(veh_num)
        cur_load[0] = env.order_sets[temp][3]
        ve_values = [env.order_num * 2 // veh_num] * (veh_num - 1)
        remaining = env.order_num * 2 - sum(ve_values)
        ve_values.append(remaining)
        ve_values2 = ve_values[:]
        ve_values2[0] -= 1
        Foods[i, 0] = start
        j = 1
        for k in range(veh_num):
            if k == 0:
                qu_num = 1
            else:
                qu_num = 0
            while ve_values2[k] != 0:
                allow = []
                mask = np.zeros(env.order_num, dtype=int)
                veh = k
                for h in range(env.order_num):
                    if (state[h] == 0 and qu_num < ve_values[veh] // 2 and ve_values[veh] > 2) or (
                            state[h] == 0 and qu_num > ve_values[veh] / 2 and ve_values[veh] == 2):
                        if env.order_sets[h][3] <= max_load[veh] - cur_load[veh]:
                            allow.append(env.order_sets[h][1])
                            mask[h] = 1
                    if state[h] == 1 and flags_ding[h] == veh:
                        allow.append(env.order_sets[h][2])
                        mask[h] = 1
                if len(allow) == 0:
                    break
                P = np.ones(len(allow)) / len(allow)
                target_index = np.random.choice(len(allow), p=P)
                order_no = -1
                count = -1
                for o in range(env.order_num):
                    if mask[o] == 1:
                        count += 1
                    if count == target_index:
                        order_no = o
                        break
                if state[order_no] == 1:
                    state[order_no] = 2
                    cur_load[veh] -= env.order_sets[order_no][3]
                    Foods[i, j] = env.order_sets[order_no][2]
                    flags_node[j] = veh
                    ve_values2[veh] -= 1
                if state[order_no] == 0:
                    state[order_no] = 1
                    cur_load[veh] += env.order_sets[order_no][3]
                    Foods[i, j] = env.order_sets[order_no][1]
                    flags_node[j] = veh
                    qu_num += 1
                    ve_values2[veh] -= 1
                    flags_ding[order_no] = veh
                j += 1
            if j == D:
                break
        state_list.append(state)
        flags_node_list.append(flags_node)
    # 计算适应度
    obj_values = np.zeros(FoodNumber)
    for i in range(FoodNumber):
        Route = Foods[i, :].astype(int)
        Flag_node = flags_node_list[i].astype(int)
        total_times, wait_times, Length, nodes, maxve_time, vtotal_time, vwait_time = distance(Route, Flag_node,
                                                                                               veh_num)
        mu = total_times + 20 * (np.max(vtotal_time) - np.min(vtotal_time))
        obj_values[i] = mu
        fitness[i] = 1 / (1 + mu)
    # 主循环
    while iter < MaxCycle:
        # 雇佣蜂阶段
        for i in range(FoodNumber):
            k = np.random.randint(0, D)
            neighbor = np.random.randint(0, FoodNumber)
            while neighbor == i:
                neighbor = np.random.randint(0, FoodNumber)
            vi = Foods[i, :].copy()
            vi[k], vi[(k + 1) % D] = vi[(k + 1) % D], vi[k]
            # 计算新解的适应度
            state = state_list[i].copy()
            flags_node = flags_node_list[i].copy()
            Route = vi.astype(int)
            total_times, wait_times, Length, nodes, maxve_time, vtotal_time, vwait_time = distance(Route,
                                                                                                   flags_node.astype(
                                                                                                       int), veh_num)
            mu = total_times + 20 * (np.max(vtotal_time) - np.min(vtotal_time))
            obj_val = mu
            fitness_new = 1 / (1 + obj_val)
            if fitness_new > fitness[i]:
                Foods[i, :] = vi
                fitness[i] = fitness_new
                obj_values[i] = obj_val
                trial[i] = 0
            else:
                trial[i] += 1
        # 观察蜂阶段
        maxfit = np.max(fitness)
        prob = (0.9 * fitness / maxfit) + 0.1
        for i in range(FoodNumber):
            if np.random.random() < prob[i]:
                k = np.random.randint(0, D)
                neighbor = np.random.randint(0, FoodNumber)
                while neighbor == i:
                    neighbor = np.random.randint(0, FoodNumber)
                vi = Foods[i, :].copy()
                vi[k], vi[(k + 1) % D] = vi[(k + 1) % D], vi[k]
                # 计算新解的适应度
                state = state_list[i].copy()
                flags_node = flags_node_list[i].copy()
                Route = vi.astype(int)
                total_times, wait_times, Length, nodes, maxve_time, vtotal_time, vwait_time = distance(Route,
                                                                                                       flags_node.astype(
                                                                                                           int),
                                                                                                       veh_num)
                mu = total_times + 20 * (np.max(vtotal_time) - np.min(vtotal_time))
                obj_val = mu
                fitness_new = 1 / (1 + obj_val)
                if fitness_new > fitness[i]:
                    Foods[i, :] = vi
                    fitness[i] = fitness_new
                    obj_values[i] = obj_val
                    trial[i] = 0
                else:
                    trial[i] += 1
        # 侦察蜂阶段
        for i in range(FoodNumber):
            if trial[i] > limit:
                # 与初始化食物源相同的代码
                state = np.zeros(env.order_num, dtype=int)
                flags_node = np.ones(D) * 100
                temp = np.random.choice([i for i in range(env.order_num)], 1)[0]
                start = env.order_sets[temp][1]
                state[temp] = 1
                flags_node[0] = 0
                flags_ding = np.ones(env.order_num) * 100
                flags_ding[temp] = 0
                cur_load = np.zeros(veh_num)
                cur_load[0] = env.order_sets[temp][3]
                ve_values = [env.order_num * 2 // veh_num] * (veh_num - 1)
                remaining = env.order_num * 2 - sum(ve_values)
                ve_values.append(remaining)
                ve_values2 = ve_values[:]
                ve_values2[0] -= 1
                Foods[i, 0] = start
                j = 1
                for k in range(veh_num):
                    if k == 0:
                        qu_num = 1
                    else:
                        qu_num = 0
                    while ve_values2[k] != 0:
                        allow = []
                        mask = np.zeros(env.order_num, dtype=int)
                        veh = k
                        for h in range(env.order_num):
                            if (state[h] == 0 and qu_num < ve_values[veh] // 2 and ve_values[veh] > 2) or (
                                    state[h] == 0 and qu_num > ve_values[veh] / 2 and ve_values[veh] == 2):
                                if env.order_sets[h][3] <= max_load[veh] - cur_load[veh]:
                                    allow.append(env.order_sets[h][1])
                                    mask[h] = 1
                            if state[h] == 1 and flags_ding[h] == veh:
                                allow.append(env.order_sets[h][2])
                                mask[h] = 1
                        if len(allow) == 0:
                            break
                        P = np.ones(len(allow)) / len(allow)
                        target_index = np.random.choice(len(allow), p=P)
                        order_no = -1
                        count = -1
                        for o in range(env.order_num):
                            if mask[o] == 1:
                                count += 1
                            if count == target_index:
                                order_no = o
                                break
                        if state[order_no] == 1:
                            state[order_no] = 2
                            cur_load[veh] -= env.order_sets[order_no][3]
                            Foods[i, j] = env.order_sets[order_no][2]
                            flags_node[j] = veh
                            ve_values2[veh] -= 1
                        if state[order_no] == 0:
                            state[order_no] = 1
                            cur_load[veh] += env.order_sets[order_no][3]
                            Foods[i, j] = env.order_sets[order_no][1]
                            flags_node[j] = veh
                            qu_num += 1
                            ve_values2[veh] -= 1
                            flags_ding[order_no] = veh
                        j += 1
                    if j == D:
                        break
                state_list[i] = state
                flags_node_list[i] = flags_node
                Route = Foods[i, :].astype(int)
                total_times, wait_times, Length, nodes, maxve_time, vtotal_time, vwait_time = distance(Route,
                                                                                                       flags_node.astype(
                                                                                                           int),
                                                                                                       veh_num)
                mu = total_times + 20 * (np.max(vtotal_time) - np.min(vtotal_time))
                obj_values[i] = mu
                fitness[i] = 1 / (1 + mu)
                trial[i] = 0
        # 记录最优解
        min_obj_val = np.min(obj_values)
        min_index = np.argmin(obj_values)
        if iter == 0:
            Tomu_best[iter] = min_obj_val
            Totimes_best[iter] = total_times
            Watimes_best[iter] = wait_times
            Length_best[iter] = Length
            Route_best[iter, :] = nodes
            Vtotime_best[iter] = vtotal_time
            Vwatime_best[iter] = vwait_time
        else:
            if min_obj_val < Tomu_best[iter - 1]:
                Tomu_best[iter] = min_obj_val
                Totimes_best[iter] = total_times
                Watimes_best[iter] = wait_times
                Length_best[iter] = Length
                Route_best[iter, :] = nodes
                Vtotime_best[iter] = vtotal_time
                Vwatime_best[iter] = vwait_time
            else:
                Tomu_best[iter] = Tomu_best[iter - 1]
                Totimes_best[iter] = Totimes_best[iter - 1]
                Watimes_best[iter] = Watimes_best[iter - 1]
                Length_best[iter] = Length_best[iter - 1]
                Route_best[iter, :] = Route_best[iter - 1, :]
                Vtotime_best[iter] = Vtotime_best[iter - 1]
                Vwatime_best[iter] = Vwatime_best[iter - 1]
        iter += 1
        endt = time.perf_counter()
        print(f"-----------迭代轮数{iter}----------")
        print("目标值", Tomu_best[iter - 1])
        print("最短时间per", Totimes_best[iter - 1])
        print('每辆车总时间：', Vtotime_best[iter - 1])
        print('每辆车等待时间：', Vwatime_best[iter - 1])
        print("运行耗时per", endt - startt)
    return Totimes_best, Route_best, Watimes_best, Length_best, Vtotime_best, Vwatime_best, Tomu_best


# 结果显示
Totimes_best, Route_best, Watimes_best, Length_best, Vtotime_best, Vwatime_best, Tomu_best = ABC(Route_best,
                                                                                                 Totimes_best,
                                                                                                 Length_best,
                                                                                                 Watimes_best,
                                                                                                 Tomu_best)
Shortest_mu = np.min(Tomu_best)
index = np.argwhere(Tomu_best == Shortest_mu)[0][0]
Shortest_Totimes = Totimes_best[index]
Shortest_Route = Route_best[index, :]
Shortest_Watimes = Watimes_best[index]
Shortest_Length = Length_best[index]
Shortest_Vtotime = Vtotime_best[index]
Shortest_Vwatime = Vwatime_best[index]
end = time.perf_counter()
print("运行耗时CoTime", end - startt)
print("目标值", Shortest_mu)
print('最短总时间：', Shortest_Totimes)
print('最短路径：', Shortest_Route)
print('等待时间：', Shortest_Watimes)
print('总长度Length：', Shortest_Length)
print('时间AllTime：', Shortest_Vtotime)
print('等待时间WaitTime：', Shortest_Vwatime)


def extract_subarrays(arr):
    subarrays = []
    start_index = 0
    end_index = 0
    for i in range(len(arr)):
        if arr[i] == 0:
            if start_index != end_index:
                subarray = arr[start_index - 1:end_index + 1]
                subarrays.append(subarray)
            start_index = i + 1
            end_index = i + 1
        else:
            end_index += 1
    return subarrays


# 绘制结果
plt.figure(figsize=(8, 6), dpi=450)
plt.title('ABC_VRP')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(env.ticoor_sets[..., 3], env.ticoor_sets[..., 4], 'ko', ms=3)
xbests = [[] for _ in range(veh_num)]
subarrays = extract_subarrays(Shortest_Route)
for j, subarray in enumerate(subarrays):
    for i in subarray:
        xbests[j].append(np.argwhere(env.ticoor_index == i)[0][0])
colors = ['red', 'blue', 'green', 'purple', 'pink']
for k in range(veh_num):
    plt.plot(env.ticoor_sets[xbests[k], 3], env.ticoor_sets[xbests[k], 4], colors[k])
plt.legend(['All Points', 'Route 1', 'Route 2', 'Route 3', 'Route 4', 'Route 5'])
plt.savefig('testblueline.jpg')
plt.show()
