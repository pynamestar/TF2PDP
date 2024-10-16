import pandas as pd
import math
import random
import copy
import time

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


# 定义获取速度的函数
def get_speed(current_time):
    current_time = current_time % 24  # 将时间规范到24小时制
    for idx, row in speeds.iterrows():
        if row['start_time'] <= current_time < row['end_time']:
            return row['speed']
    return 30  # 默认速度


vehicles = pd.DataFrame({
    'vehicle_id': [i for i in range(vehicle_num)],
    'capacity': vehicle_capacities
})

# 计算仓库之间的距离
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

# 订单状态：0-未完成，1-已取货，2-已送达
orders['status'] = 0


# 定义车辆类，添加 distance 属性用于记录行驶距离
class Vehicle:
    def __init__(self, vehicle_id, capacity):
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.route = [0]  # 从车厂出发
        self.load = 0
        self.time = 8  # 开始时间为8点
        self.current_location = 0
        self.orders = []  # 分配给该车辆的订单
        self.travel_time = 0
        self.waiting_time = 0
        self.distance = 0.0  # 新增：记录行驶距离
        self.order_sequence = []  # 记录订单的取送货顺序


# 初始化车辆列表
vehicle_list = [Vehicle(row['vehicle_id'], row['capacity']) for idx, row in vehicles.iterrows()]

# 初始化订单列表
order_list = orders.to_dict('records')


# 定义计算路线总距离的函数
def calculate_route_distance(route):
    total_distance = 0
    for i in range(len(route) - 1):
        curr_node = route[i]
        next_node = route[i + 1]
        if (curr_node, next_node) in distances:
            total_distance += distances[curr_node, next_node]
        else:
            total_distance += float('inf')  # 如果没有路径，设置为无限大
    return total_distance


# 检查路线的合法性，确保取货在送货之前
def check_route_validity(route):
    pick_deliver_status = {}
    for node in route:
        # 检查是否是取货或送货仓库
        orders_pickup = [order for order in order_list if node in get_warehouses(order['pickup_node'])]
        orders_delivery = [order for order in order_list if node in get_warehouses(order['delivery_node'])]
        if orders_pickup:
            for order in orders_pickup:
                pick_deliver_status[order['order_id']] = 'picked'
        elif orders_delivery:
            for order in orders_delivery:
                if pick_deliver_status.get(order['order_id']) != 'picked':
                    return False  # 送货在取货之前，非法
                pick_deliver_status[order['order_id']] = 'delivered'
    return True


# 获取节点对应的仓库列表
def get_warehouses(node):
    return node_warehouses[node_warehouses['node'] == node]['warehouse_ids'].values[0]


# 定义局部搜索算法，包含订单交换和2-opt优化
def local_search(vehicle_list, order_list):
    # 初始分配订单：按权重降序排序，依次分配给车辆
    unfinished_orders = [order for order in order_list if order['status'] < 2]
    unfinished_orders.sort(key=lambda x: -x['weight'])
    vehicle_index = 0
    for order in unfinished_orders:
        vehicle = vehicle_list[vehicle_index % vehicle_num]
        vehicle.orders.append(order['order_id'])
        vehicle_index += 1

    # 构建初始解
    construct_initial_solution(vehicle_list, order_list)

    # 开始局部搜索
    max_iterations = 100  # 最大迭代次数
    start_time = time.perf_counter()  # 记录局部搜索开始时间
    for iteration in range(max_iterations):
        print(f"\n第 {iteration + 1} 次迭代")
        improved = False
        # 尝试交换订单
        for v1 in vehicle_list:
            for v2 in vehicle_list:
                if v1.vehicle_id >= v2.vehicle_id:
                    continue
                for o1 in v1.orders.copy():  # 使用copy()避免迭代时修改列表
                    for o2 in v2.orders.copy():
                        # 深拷贝车辆列表，避免影响原始数据
                        new_vehicle_list = copy.deepcopy(vehicle_list)
                        new_v1 = next(v for v in new_vehicle_list if v.vehicle_id == v1.vehicle_id)
                        new_v2 = next(v for v in new_vehicle_list if v.vehicle_id == v2.vehicle_id)
                        # 交换订单
                        new_v1.orders.remove(o1)
                        new_v2.orders.remove(o2)
                        new_v1.orders.append(o2)
                        new_v2.orders.append(o1)
                        # 重建车辆路线
                        if not construct_vehicle_route(new_v1) or not construct_vehicle_route(new_v2):
                            continue  # 不可行的解，跳过
                        # 计算新的总成本
                        new_total_cost = calculate_total_cost(new_vehicle_list)
                        current_total_cost = calculate_total_cost(vehicle_list)
                        if new_total_cost < current_total_cost:
                            # 接受新解
                            vehicle_list[:] = new_vehicle_list  # 更新原始车辆列表
                            improved = True
                            print(f"交换订单 {o1} 和 {o2}，总成本降低为 {new_total_cost:.2f}")
                            break  # 退出当前循环，进入下一次迭代
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            print("无法进一步改进，结束局部搜索")
            break
    end_time = time.perf_counter()  # 记录局部搜索结束时间
    solution_time = end_time - start_time  # 计算求解时间

    # 所有车辆返回车厂，并计算返回时间和距离
    for vehicle in vehicle_list:
        if vehicle.current_location != 0:
            if (vehicle.current_location, 0) in distances:
                distance = distances[vehicle.current_location, 0]
                speed_val = get_speed(vehicle.time)
                travel_time = distance / speed_val
                vehicle.time += travel_time
                vehicle.travel_time += travel_time
                vehicle.distance += distance  # 累加行驶距离
                vehicle.route.append(0)  # 返回车厂
                vehicle.current_location = 0
                if vehicle.time % 24 > 20:
                    # 需要等待到次日8点
                    waiting = 24 - (vehicle.time % 24) + 8
                    vehicle.waiting_time += waiting
                    vehicle.time += waiting
                print(
                    f"车辆 {vehicle.vehicle_id} 返回车厂，时间 {vehicle.time:.2f} 小时，总行驶距离 {vehicle.distance:.2f} 单位")
            else:
                print(f"车辆 {vehicle.vehicle_id} 无法返回车厂，地点 {vehicle.current_location} 与车厂之间无路径。")

    return solution_time  # 返回求解时间


# 构建初始解
def construct_initial_solution(vehicle_list, order_list):
    for vehicle in vehicle_list:
        construct_vehicle_route(vehicle)


# 构建车辆的路线
def construct_vehicle_route(vehicle):
    vehicle.route = [0]
    vehicle.load = 0
    vehicle.time = 8
    vehicle.current_location = 0
    vehicle.travel_time = 0
    vehicle.waiting_time = 0
    vehicle.distance = 0.0  # 初始化行驶距离
    vehicle.order_sequence = []
    # 按订单顺序处理
    for order_id in vehicle.orders:
        order = next(order for order in order_list if order['order_id'] == order_id)
        weight = order['weight']
        # 取货
        pickup_success = process_order_stop(vehicle, order, 'pickup')
        if not pickup_success:
            return False  # 无法构建可行路线
        # 送货
        delivery_success = process_order_stop(vehicle, order, 'delivery')
        if not delivery_success:
            return False  # 无法构建可行路线
    return True


# 处理订单的取货或送货操作
def process_order_stop(vehicle, order, operation):
    node = order['pickup_node'] if operation == 'pickup' else order['delivery_node']
    warehouses_list = get_warehouses(node)
    feasible_warehouses = []
    for wh in warehouses_list:
        start_time = warehouses.loc[warehouses['warehouse_id'] == wh, 'start_time'].values[0]
        end_time = warehouses.loc[warehouses['warehouse_id'] == wh, 'end_time'].values[0]
        if (vehicle.current_location, wh) not in distances:
            continue
        distance = distances[vehicle.current_location, wh]
        speed_val = get_speed(vehicle.time)
        travel_time = distance / speed_val
        arrival_time = vehicle.time + travel_time
        # 考虑等待时间
        if arrival_time % 24 > end_time:
            # 超过服务时间，等待到次日
            waiting = 24 - (arrival_time % 24) + start_time
            arrival_time += waiting
        elif arrival_time % 24 < start_time:
            # 需要等待
            waiting = start_time - (arrival_time % 24)
            arrival_time += waiting
        else:
            waiting = 0
        if start_time <= arrival_time % 24 <= end_time:
            feasible_warehouses.append((wh, arrival_time, travel_time, waiting))
    if not feasible_warehouses:
        return False  # 无可行仓库
    # 选择最早到达的仓库
    wh, arrival_time, travel_time, waiting = min(feasible_warehouses, key=lambda x: x[1])
    # 更新车辆状态
    vehicle.route.append(wh)
    vehicle.time = arrival_time
    vehicle.current_location = wh
    if operation == 'pickup':
        if vehicle.load + order['weight'] > vehicle.capacity:
            return False  # 超过车辆容量
        vehicle.load += order['weight']
    else:
        vehicle.load -= order['weight']
    vehicle.travel_time += travel_time
    vehicle.distance += distance  # 累加行驶距离
    vehicle.waiting_time += waiting
    if vehicle.time % 24 > 20:
        # 需要等待到次日8点
        waiting = 24 - (vehicle.time % 24) + 8
        vehicle.waiting_time += waiting
        vehicle.time += waiting
    vehicle.order_sequence.append((order['order_id'], operation))
    return True


# 计算总成本
def calculate_total_cost(vehicle_list):
    vehicle_times = [vehicle.travel_time + vehicle.waiting_time for vehicle in vehicle_list]
    total_travel_time = sum([vehicle.travel_time for vehicle in vehicle_list])
    total_waiting_time = sum([vehicle.waiting_time for vehicle in vehicle_list])
    max_time = max(vehicle_times)
    min_time = min(vehicle_times)
    total_cost = total_travel_time + total_waiting_time + 20 * (max_time - min_time)
    return total_cost


# 执行局部搜索算法
solution_time = local_search(vehicle_list, order_list)

# 计算目标函数值
total_cost = calculate_total_cost(vehicle_list)
total_travel_time = sum([vehicle.travel_time for vehicle in vehicle_list])
total_waiting_time = sum([vehicle.waiting_time for vehicle in vehicle_list])
vehicle_times = [vehicle.travel_time + vehicle.waiting_time for vehicle in vehicle_list]
max_time = max(vehicle_times)
min_time = min(vehicle_times)
max_wait_time = max([vehicle.waiting_time for vehicle in vehicle_list])

total_distance = sum([vehicle.distance for vehicle in vehicle_list])

# 输出结果，包括每辆车的行驶路径和行驶距离
print('\n最终结果:')
for vehicle in vehicle_list:
    print(f'\n车辆 {vehicle.vehicle_id} 的订单:')
    print(vehicle.orders)
    print('路线:', vehicle.route)
    print(f'车辆总耗时: {vehicle.travel_time + vehicle.waiting_time:.2f} 小时 '
          f'(行驶时间: {vehicle.travel_time:.2f} 小时, 等待时间: {vehicle.waiting_time:.2f} 小时)')
    print(f'车辆总行驶距离: {vehicle.distance:.2f} 单位')

print(f'\n最小化的总成本: {total_cost:.2f}')
print(f'总行驶时间: {total_travel_time:.2f} 小时')
print(f'总等待时间: {total_waiting_time:.2f} 小时')
print(f'总行驶距离: {total_distance:.2f} 单位')
print(f'车辆最大耗时与最小耗时之差: {max_time - min_time:.2f} 小时')
print(f'最长等待时间（WaitTime）: {max_wait_time:.2f} 小时')
print(f'总求解时间: {solution_time:.2f} 秒')
