import coptpy as cp
import pandas as pd
import math
import random
import copy
import sys

# 将输出重定向到 output.txt 文件
# sys.stdout = open('output.txt', 'w')

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
def get_speed(current_time):
    current_time = current_time % 24  # 将时间规范到24小时制
    for idx, row in speeds.iterrows():
        if row['start_time'] <= current_time < row['end_time']:
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
env = cp.Envr()
model = env.createModel("FlexiblePickupDelivery")

# 10. 定义变量
# x[i,j,k]: 车辆k是否从节点i到节点j行驶，二元变量
x = {}
for (i, j) in distances.keys():
    for k in vehicles['vehicle_id']:
        var_name = f"x_{i}_{j}_{k}"
        x[i, j, k] = model.addVar(vtype=cp.COPT.BINARY, name=var_name)

# t[i,k]: 车辆k到达节点i的时间
t = {}
for i in warehouses['warehouse_id']:
    for k in vehicles['vehicle_id']:
        var_name = f"t_{i}_{k}"
        t[i, k] = model.addVar(vtype=cp.COPT.CONTINUOUS, name=var_name)

# q[i,k]: 车辆k在节点i的载货量
q = {}
for i in warehouses['warehouse_id']:
    for k in vehicles['vehicle_id']:
        var_name = f"q_{i}_{k}"
        q[i, k] = model.addVar(vtype=cp.COPT.CONTINUOUS, name=var_name)

# waiting_time[i,k]: 车辆k在节点i的等待时间
waiting_time = {}
for i in warehouses['warehouse_id']:
    for k in vehicles['vehicle_id']:
        var_name = f"waiting_time_{i}_{k}"
        waiting_time[i, k] = model.addVar(vtype=cp.COPT.CONTINUOUS, name=var_name)

# order_status[o]: 订单o的状态，0：未完成，1：已取货，2：已送达
order_status = {}
for o in orders['order_id']:
    var_name = f"order_status_{o}"
    order_status[o] = model.addVar(vtype=cp.COPT.INTEGER, lb=0, ub=2, name=var_name)

# pickup[o,i,k]: 订单o是否由车辆k在仓库i取货
pickup = {}
for o in orders['order_id']:
    for i in warehouses['warehouse_id']:
        for k in vehicles['vehicle_id']:
            var_name = f"pickup_{o}_{i}_{k}"
            pickup[o, i, k] = model.addVar(vtype=cp.COPT.BINARY, name=var_name)

# delivery[o,j,k]: 订单o是否由车辆k在仓库j送货
delivery = {}
for o in orders['order_id']:
    for j in warehouses['warehouse_id']:
        for k in vehicles['vehicle_id']:
            var_name = f"delivery_{o}_{j}_{k}"
            delivery[o, j, k] = model.addVar(vtype=cp.COPT.BINARY, name=var_name)

# total_time_vars[k]: 车辆k的总运行时间
total_time_vars = {}
for k in vehicles['vehicle_id']:
    var_name = f"total_time_{k}"
    total_time_vars[k] = model.addVar(vtype=cp.COPT.CONTINUOUS, name=var_name)

# max_total_time 和 min_total_time: 所有车辆中最大和最小的总运行时间
max_total_time = model.addVar(vtype=cp.COPT.CONTINUOUS, name='max_total_time')
min_total_time = model.addVar(vtype=cp.COPT.CONTINUOUS, name='min_total_time')

# 11. 定义目标函数
# 计算行驶时间
total_travel_time = cp.quicksum(
    (distances[i, j] / min_speed) * x[i, j, k]
    for (i, j) in distances.keys()
    for k in vehicles['vehicle_id']
)

# 计算等待时间
total_waiting_time = cp.quicksum(
    waiting_time[i, k]
    for i in warehouses['warehouse_id']
    for k in vehicles['vehicle_id']
)

# 设置目标函数
model.setObjective(
    total_travel_time + total_waiting_time + 20 * (max_total_time - min_total_time),
    sense=cp.COPT.MINIMIZE
)

# 12. 添加约束条件

# 12.1 车辆的出发和返回时间
for k in vehicles['vehicle_id']:
    # 车辆从车厂出发时间 >=8
    model.addConstr(t[0, k] >= 8, name=f"Start_Time_Vehicle_{k}")
    # 总运行时间 = 到达车厂的时间 - 8
    model.addConstr(total_time_vars[k] == t[0, k] - 8, name=f"Total_Time_Vehicle_{k}")

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
    model.addConstr(
        cp.quicksum(pickup[o, i, k] for i in pickup_warehouses for k in vehicles['vehicle_id']) == 1,
        name=f"Pickup_Assign_Order_{o}"
    )
    model.addConstr(
        cp.quicksum(delivery[o, j, k] for j in delivery_warehouses for k in vehicles['vehicle_id']) == 1,
        name=f"Delivery_Assign_Order_{o}"
    )

    for k in vehicles['vehicle_id']:
        # 确保同一辆车执行取送货
        model.addConstr(
            cp.quicksum(pickup[o, i, k] for i in pickup_warehouses) ==
            cp.quicksum(delivery[o, j, k] for j in delivery_warehouses),
            name=f"Same_Vehicle_Order_{o}_Vehicle_{k}"
        )

        # 车辆访问取送货仓库时，才能进行取送货
        for i in pickup_warehouses:
            model.addConstr(
                pickup[o, i, k] <= cp.quicksum(x[h, i, k] for h in warehouses['warehouse_id'] if h != i),
                name=f"Pickup_Access_Order_{o}_Vehicle_{k}_Warehouse_{i}"
            )
        for j in delivery_warehouses:
            model.addConstr(
                delivery[o, j, k] <= cp.quicksum(x[j, h, k] for h in warehouses['warehouse_id'] if h != j),
                name=f"Delivery_Access_Order_{o}_Vehicle_{k}_Warehouse_{j}"
            )

        # 确保取送货操作只执行一次
        model.addConstr(
            cp.quicksum(pickup[o, i, k] for i in pickup_warehouses) <= 1,
            name=f"Single_Pickup_Order_{o}_Vehicle_{k}"
        )
        model.addConstr(
            cp.quicksum(delivery[o, j, k] for j in delivery_warehouses) <= 1,
            name=f"Single_Delivery_Order_{o}_Vehicle_{k}"
        )

        # 确保取货在送货之前
        for i in pickup_warehouses:
            for j in delivery_warehouses:
                model.addConstr(
                    t[j, k] >= t[i, k] + 0.1 - M * (2 - pickup[o, i, k] - delivery[o, j, k]),
                    name=f"Pickup_Before_Delivery_Order_{o}_Vehicle_{k}_Pickup_{i}_Delivery_{j}"
                )

    # 更新订单状态
    # 当订单 o 被取货后，状态更新为 1
    model.addConstr(
        order_status[o] >= cp.quicksum(pickup[o, i, k] for i in pickup_warehouses for k in vehicles['vehicle_id']),
        name=f"Order_Status_Pickup_{o}"
    )
    # 当订单 o 被送货后，状态更新为 2
    model.addConstr(
        order_status[o] >= 2 * cp.quicksum(
            delivery[o, j, k] for j in delivery_warehouses for k in vehicles['vehicle_id']),
        name=f"Order_Status_Delivery_{o}"
    )

# 12.3 车辆容量和载货量更新约束
for k in vehicles['vehicle_id']:
    model.addConstr(q[0, k] == 0, name=f"Initial_Load_Vehicle_{k}")  # 车辆从车厂出发时载货量为0
    for i in warehouses['warehouse_id']:
        # 车辆容量约束
        model.addConstr(
            q[i, k] <= vehicles.loc[vehicles['vehicle_id'] == k, 'capacity'].values[0],
            name=f"Capacity_Vehicle_{k}_Location_{i}"
        )
        model.addConstr(
            q[i, k] >= 0,
            name=f"NonNegative_Load_Vehicle_{k}_Location_{i}"
        )

    # 载货量更新
    for (i, j) in distances.keys():
        if i != j:
            # q[j,k] >= q[i,k] + sum(order_weight * (pickup - delivery)) - M*(1 - x[i,j,k])
            order_contrib = cp.quicksum(
                orders.loc[orders['order_id'] == o, 'weight'].values[0] * (
                        pickup[o, i, k] - delivery[o, j, k]
                )
                for o in orders['order_id']
            )
            model.addConstr(
                q[j, k] >= q[i, k] + order_contrib - M * (1 - x[i, j, k]),
                name=f"Load_Update_Vehicle_{k}_From_{i}_To_{j}"
            )

# 12.4 时间窗约束和等待时间计算
for k in vehicles['vehicle_id']:
    for i in warehouses['warehouse_id']:
        start_time = warehouses.loc[warehouses['warehouse_id'] == i, 'start_time'].values[0]
        end_time = warehouses.loc[warehouses['warehouse_id'] == i, 'end_time'].values[0]

        # 等待时间计算
        model.addConstr(
            waiting_time[i, k] >= start_time - t[i, k],
            name=f"Waiting_Time_LowerBound_Vehicle_{k}_Location_{i}"
        )
        model.addConstr(
            waiting_time[i, k] >= 0,
            name=f"Waiting_Time_NonNegative_Vehicle_{k}_Location_{i}"
        )

        # 车辆只能在服务时间内进行操作
        model.addConstr(
            t[i, k] + waiting_time[i, k] >= start_time,
            name=f"Service_Start_Time_Vehicle_{k}_Location_{i}"
        )
        model.addConstr(
            t[i, k] + waiting_time[i, k] <= end_time,
            name=f"Service_End_Time_Vehicle_{k}_Location_{i}"
        )

# 12.5 路径连续性约束
for k in vehicles['vehicle_id']:
    # 从车厂出发
    model.addConstr(
        cp.quicksum(x[0, j, k] for j in warehouses['warehouse_id'] if j != 0) == 1,
        name=f"Start_Vehicle_{k}"
    )
    # 返回车厂
    model.addConstr(
        cp.quicksum(x[i, 0, k] for i in warehouses['warehouse_id'] if i != 0) == 1,
        name=f"End_Vehicle_{k}"
    )
    for i in warehouses['warehouse_id']:
        if i == 0:
            continue
        inflow = cp.quicksum(x[j, i, k] for j in warehouses['warehouse_id'] if j != i)
        outflow = cp.quicksum(x[i, j, k] for j in warehouses['warehouse_id'] if j != i)
        model.addConstr(
            inflow == outflow,
            name=f"Flow_Conservation_Vehicle_{k}_Location_{i}"
        )

# 12.6 行驶时间和到达时间的关系
for k in vehicles['vehicle_id']:
    for (i, j) in distances.keys():
        travel_time = distances[i, j] / min_speed
        model.addConstr(
            t[j, k] >= t[i, k] + waiting_time[i, k] + travel_time - M * (1 - x[i, j, k]),
            name=f"Time_Update_Vehicle_{k}_From_{i}_To_{j}"
        )

# 12.7 定义最大和最小总运行时间
for k in vehicles['vehicle_id']:
    model.addConstr(
        max_total_time >= total_time_vars[k],
        name=f"Max_Total_Time_Vehicle_{k}"
    )
    model.addConstr(
        min_total_time <= total_time_vars[k],
        name=f"Min_Total_Time_Vehicle_{k}"
    )

# 13. 求解模型
model.solve()

# 14. 输出结果
if model.getStatus() == cp.COPT.OPTIMAL:
    print('最优目标值:', model.getObjVal())
    # 输出每辆车的路线
    for k in vehicles['vehicle_id']:
        print(f'\n车辆{k}的路线:')
        route = [0]
        while True:
            last_node = route[-1]
            found = False
            for j in warehouses['warehouse_id']:
                if last_node != j and (last_node, j) in distances.keys():
                    # 获取变量 x[last_node, j, k]
                    var = x[last_node, j, k]
                    if model.getVal(var) > 0.5:
                        route.append(j)
                        found = True
                        break
            if not found or route[-1] == 0:
                break
        print(' -> '.join(map(str, route)))
        print(f'车辆{k}的总运行时间: {model.getVal(total_time_vars[k])}')
else:
    print('未找到最优解')
    if model.getStatus() == cp.COPT.INFEASIBLE:
        print('模型不可行，计算不可行约束集（IIS）...')

# 关闭文件
# sys.stdout.close()
