import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math
import random
import sys

# 将输出重定向到 output.txt 文件
sys.stdout = open('output.txt', 'w')
# 1. 数据准备
path = ['datasets/order2.csv', 'datasets/node2.csv', 'datasets/ticoor2.csv', 'datasets/speed.csv']
# 生成车辆信息
vehicle_num = 5
random.seed(1)  # 为了结果可重复
vehicle_capacities = [random.randint(6, 20) for _ in range(vehicle_num)]

# 读取订单信息
# 读取 CSV 文件，并将 weight 列的数据类型转换为整数
orders = pd.read_csv(path[0], names=['order_id', 'pickup_node', 'delivery_node', 'weight'], skiprows=1)
orders['weight'] = pd.to_numeric(orders['weight'], errors='coerce')  # 强制转换为数值类型，非法值转换为 NaN
# 读取数据并跳过标题行
nodes = pd.read_csv(path[1], names=['id', 'id1', 'id2', 'id3', 'id4'], skiprows=1)

# 将每一行的仓库 ID 转换为列表格式
nodes['warehouse_ids'] = nodes[['id1', 'id2', 'id3', 'id4']].apply(
    lambda x: [int(i) for i in x if pd.notna(i)], axis=1
)
# 创建最终的 DataFrame
node_warehouse_data = pd.DataFrame({
    'node': nodes['id'],
    'warehouse_ids': nodes['warehouse_ids']
})
node_warehouses = pd.DataFrame(node_warehouse_data)

# 读取 ticoor 表格数据并跳过标题行
ticoor = pd.read_csv(path[2], names=['warehouse_id', 'start_time', 'end_time', 'coor_x', 'coor_y'],
                     skiprows=1)

# 创建 warehouse_data 字典
warehouse_data = {
    'warehouse_id': ticoor['warehouse_id'].tolist(),
    'start_time': ticoor['start_time'].tolist(),
    'end_time': ticoor['end_time'].tolist(),
    'coor_x': ticoor['coor_x'].tolist(),
    'coor_y': ticoor['coor_y'].tolist()
}

# 转换为 DataFrame 格式
warehouses = pd.DataFrame(warehouse_data)

# 读取 speed 表格并设置列名
speed_data_raw = pd.read_csv(path[3], names=['start_time', 'end_time', 'veh_speed'], skiprows=1)

# 构建 speed_data 字典
speed_data = {
    'start_time': speed_data_raw['start_time'].tolist(),
    'end_time': speed_data_raw['end_time'].tolist(),
    'speed': speed_data_raw['veh_speed'].tolist()
}

# 转换为 DataFrame 格式
speeds = pd.DataFrame(speed_data)


# 5. 定义获取速度的函数
def get_speed(time):
    for idx, row in speeds.iterrows():
        if row['start_time'] <= time < row['end_time']:
            return row['speed']
    return 30  # 默认速度


# 6. 生成车辆信息
vehicles = pd.DataFrame({
    'vehicle_id': [i for i in range(vehicle_num)],
    'capacity': vehicle_capacities
})

# 7. 计算节点之间的距离
warehouse_coords = warehouses.set_index('warehouse_id')[['coor_x', 'coor_y']].to_dict('index')


def calculate_distance(i, j):
    xi, yi = warehouse_coords[i]['coor_x'], warehouse_coords[i]['coor_y']
    xj, yj = warehouse_coords[j]['coor_x'], warehouse_coords[j]['coor_y']
    return math.hypot(xi - xj, yi - yj)


distances = {}
for i in warehouses['warehouse_id']:
    for j in warehouses['warehouse_id']:
        if i != j:
            distances[i, j] = calculate_distance(i, j)

# 8. 计算大M值
max_distance = max(distances.values())
min_speed = speeds['speed'].min()
M = max_distance / min_speed + 24  # 最大行驶时间加上一天的时间，确保足够大

# 9. 建立模型
model = gp.Model('FlexiblePickupDelivery')

# 10. 定义变量
# x[i,j,k]: 车辆k是否从节点i到节点j行驶，二元变量
x = model.addVars(distances.keys(), vehicles['vehicle_id'], vtype=GRB.BINARY, name='x')

# t[i,k]: 车辆k到达节点i的时间
t = model.addVars(warehouses['warehouse_id'], vehicles['vehicle_id'], vtype=GRB.CONTINUOUS, name='t')

# q[i,k]: 车辆k在节点i的载货量
q = model.addVars(warehouses['warehouse_id'], vehicles['vehicle_id'], vtype=GRB.CONTINUOUS, name='q')

# waiting_time[i,k]: 车辆k在节点i的等待时间
waiting_time = model.addVars(warehouses['warehouse_id'], vehicles['vehicle_id'], vtype=GRB.CONTINUOUS,
                             name='waiting_time')

# order_status[o]: 订单o的状态，0：未完成，1：已取货，2：已送达
order_status = model.addVars(orders['order_id'], vtype=GRB.INTEGER, lb=0, ub=2, name='order_status')

# pickup[o,i,k]: 订单o是否由车辆k在仓库i取货
pickup = model.addVars(orders['order_id'], warehouses['warehouse_id'], vehicles['vehicle_id'], vtype=GRB.BINARY,
                       name='pickup')

# delivery[o,j,k]: 订单o是否由车辆k在仓库j送货
delivery = model.addVars(orders['order_id'], warehouses['warehouse_id'], vehicles['vehicle_id'], vtype=GRB.BINARY,
                         name='delivery')

# total_time_vars[k]: 车辆k的总运行时间
total_time_vars = model.addVars(vehicles['vehicle_id'], vtype=GRB.CONTINUOUS, name="total_time")

# max_total_time 和 min_total_time: 所有车辆中最大和最小的总运行时间
max_total_time = model.addVar(vtype=GRB.CONTINUOUS, name='max_total_time')
min_total_time = model.addVar(vtype=GRB.CONTINUOUS, name='min_total_time')

# 11. 定义目标函数
# 计算行驶时间
total_travel_time = gp.quicksum(
    (distances[i, j] / min_speed) * x[i, j, k] for i, j in distances.keys() for k in vehicles['vehicle_id'])
total_waiting_time = gp.quicksum(
    waiting_time[i, k] for i in warehouses['warehouse_id'] for k in vehicles['vehicle_id'])

# 目标函数：最小化总时间和车辆耗时差异
model.setObjective(total_travel_time + total_waiting_time + 20 * (max_total_time - min_total_time), GRB.MINIMIZE)

# 12. 添加约束条件

# 12.1 车辆的出发和返回时间
for k in vehicles['vehicle_id']:
    model.addConstr(t[0, k] >= 8)  # 车辆从车厂出发时间不早于8点
    model.addConstr(total_time_vars[k] == t[0, k] - 8)  # 车辆总运行时间

# 12.2 每个订单的取送货由一个车辆完成
for o in orders['order_id']:
    order_weight = orders.loc[orders['order_id'] == o, 'weight'].values[0]
    if order_weight == 0:
        continue  # 跳过重量为0的订单
    pickup_node = orders.loc[orders['order_id'] == o, 'pickup_node'].values[0]
    delivery_node = orders.loc[orders['order_id'] == o, 'delivery_node'].values[0]
    pickup_warehouses = node_warehouses[node_warehouses['node'] == pickup_node]['warehouse_ids'].values[0]
    delivery_warehouses = node_warehouses[node_warehouses['node'] == delivery_node]['warehouse_ids'].values[0]

    # 订单的取送货由同一辆车完成
    model.addConstr(gp.quicksum(pickup[o, i, k] for i in pickup_warehouses for k in vehicles['vehicle_id']) == 1)
    model.addConstr(gp.quicksum(delivery[o, j, k] for j in delivery_warehouses for k in vehicles['vehicle_id']) == 1)
    for k in vehicles['vehicle_id']:
        # 确保同一辆车执行取送货
        model.addConstr(gp.quicksum(pickup[o, i, k] for i in pickup_warehouses) ==
                        gp.quicksum(delivery[o, j, k] for j in delivery_warehouses))

        # 车辆访问取送货仓库时，才能进行取送货
        for i in pickup_warehouses:
            model.addConstr(pickup[o, i, k] <= gp.quicksum(x[h, i, k] for h in warehouses['warehouse_id'] if h != i))
        for j in delivery_warehouses:
            model.addConstr(delivery[o, j, k] <= gp.quicksum(x[j, h, k] for h in warehouses['warehouse_id'] if h != j))

        # 确保取送货操作只执行一次
        model.addConstr(gp.quicksum(pickup[o, i, k] for i in pickup_warehouses) <= 1)
        model.addConstr(gp.quicksum(delivery[o, j, k] for j in delivery_warehouses) <= 1)

        # 确保取货在送货之前
        for i in pickup_warehouses:
            for j in delivery_warehouses:
                model.addConstr(t[j, k] >= t[i, k] + 0.1 - M * (2 - pickup[o, i, k] - delivery[o, j, k]))

    # 更新订单状态
    # 当订单 o 被取货后，状态更新为 1
    model.addConstr(
        order_status[o] >= gp.quicksum(pickup[o, i, k] for i in pickup_warehouses for k in vehicles['vehicle_id']))
    # 当订单 o 被送货后，状态更新为 2
    model.addConstr(order_status[o] >= 2 * gp.quicksum(
        delivery[o, j, k] for j in delivery_warehouses for k in vehicles['vehicle_id']))

# 12.3 车辆容量和载货量更新约束
for k in vehicles['vehicle_id']:
    model.addConstr(q[0, k] == 0)  # 车辆从车厂出发时载货量为0
    for i in warehouses['warehouse_id']:
        # 车辆容量约束
        model.addConstr(q[i, k] <= vehicles.loc[vehicles['vehicle_id'] == k, 'capacity'].values[0])
        model.addConstr(q[i, k] >= 0)

    # 载货量更新
    for i, j in distances.keys():
        if i != j:
            model.addConstr(q[j, k] >= q[i, k] + gp.quicksum(
                orders.loc[orders['order_id'] == o, 'weight'].values[0] * (
                        gp.quicksum(pickup[o, j, k] for k in vehicles['vehicle_id']) - gp.quicksum(
                    delivery[o, j, k] for k in vehicles['vehicle_id']))
                for o in orders['order_id']) - M * (1 - x[i, j, k]))

# 12.4 时间窗约束和等待时间计算
for k in vehicles['vehicle_id']:
    for i in warehouses['warehouse_id']:
        start_time = warehouses.loc[warehouses['warehouse_id'] == i, 'start_time'].values[0]
        end_time = warehouses.loc[warehouses['warehouse_id'] == i, 'end_time'].values[0]

        # 等待时间计算
        model.addConstr(waiting_time[i, k] >= start_time - t[i, k])
        model.addConstr(waiting_time[i, k] >= 0)

        # 车辆只能在服务时间内进行操作
        model.addConstr(t[i, k] + waiting_time[i, k] >= start_time)
        model.addConstr(t[i, k] + waiting_time[i, k] <= end_time)

# 12.5 路径连续性约束
for k in vehicles['vehicle_id']:
    # 从车厂出发
    model.addConstr(gp.quicksum(x[0, j, k] for j in warehouses['warehouse_id'] if j != 0) == 1)
    # 返回车厂
    model.addConstr(gp.quicksum(x[i, 0, k] for i in warehouses['warehouse_id'] if i != 0) == 1)
    for i in warehouses['warehouse_id']:
        inflow = gp.quicksum(x[j, i, k] for j in warehouses['warehouse_id'] if j != i)
        outflow = gp.quicksum(x[i, j, k] for j in warehouses['warehouse_id'] if j != i)
        if i != 0:
            model.addConstr(inflow == outflow)

# 12.6 行驶时间和到达时间的关系
for k in vehicles['vehicle_id']:
    for i, j in distances.keys():
        travel_time = distances[i, j] / min_speed
        model.addConstr(t[j, k] >= t[i, k] + waiting_time[i, k] + travel_time - M * (1 - x[i, j, k]))

# 12.7 定义最大和最小总运行时间
model.addGenConstrMax(max_total_time, [total_time_vars[k] for k in vehicles['vehicle_id']])
model.addGenConstrMin(min_total_time, [total_time_vars[k] for k in vehicles['vehicle_id']])

# 13. 求解模型
model.optimize()

# 14. 输出结果
if model.status == GRB.OPTIMAL:
    print('最优目标值:', model.objVal)
    # 输出每辆车的路线
    for k in vehicles['vehicle_id']:
        print(f'\n车辆{k}的路线:')
        route = [0]
        while True:
            last_node = route[-1]
            found = False
            for j in warehouses['warehouse_id']:
                if last_node != j and (last_node, j) in distances.keys():
                    if x[last_node, j, k].X > 0.5:
                        route.append(j)
                        found = True
                        break
            if not found or route[-1] == 0:
                break
        print(' -> '.join(map(str, route)))
        print(f'车辆{k}的总运行时间: {total_time_vars[k].X}')
else:
    print('未找到最优解')
    if model.status == GRB.INFEASIBLE:
        print('模型不可行，计算不可行约束集（IIS）...')
        model.computeIIS()
        model.write('model.ilp')
        print('IIS 已写入 model.ilp 文件，可用于诊断问题。')
# 记得关闭文件
sys.stdout.close()
