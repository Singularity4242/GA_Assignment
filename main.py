import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms

DATA_PATH = 'VRP.csv'       # CSV数据文件路径
MAX_CAPACITY = 200      #车辆最大容量

# 遗传算法参数
POPULATION_SIZE = 100               # 种群大小
MAX_GENERATIONS = 1000               # 最大进化代数
CROSSOVER_PROB = 0.8               # 交叉概率
MUTATION_PROB = 0.15                # 变异概率
EARLY_STOP_PATIENCE = 50            # 早停耐心值（连续多少代无改进则停止）
CONVERGENCE_THRESHOLD = 0.001       # 收敛阈值（改进比例小于此值认为无显著改进）
MIN_GENERATIONS = 50                # 最小进化代数

class VRP_GA_Solver:
    # 为单辆车（容量200）寻找最优路线，服务100个客户
    def __init__(self):
        self.data_path = DATA_PATH
        self.max_capacity = MAX_CAPACITY
        self.customers = None
        self.depot = None
        self.distance_matrix = None
        # self.main_depot = None

        self._load_data()    # 加载数据
        self._get_distance_matrix()     #计算距离矩阵
        self._setup_ga()    # 设置遗传算法


    def _load_data(self):
        print("——————加载VRP数据——————")
        data = pd.read_csv(self.data_path)
        # 分离客户和仓库
        self.customers = data[data['CUST_OR_DEPOT'] == 'CUSTOMER'].copy()
        self.depot = data[(data['CUST_OR_DEPOT'] == 'DEPOT') & (data['NO'] == 0)].iloc[0]
        # print(f"找到 {len(self.customers)} 个客户")
        # print(self.depot)
        # print(f"起点仓库位置: ({self.depot['XCOORD']}, {self.depot['YCOORD']})")

    # 计算每个点之间的距离形成矩阵形式矩阵——客户之间 + 客户与仓库0之间
    def _get_distance_matrix(self):
        print("——————计算距离矩阵——————")
        n_customers = len(self.customers)
        n_depots = len(self.depot)
        self.distance_matrix = np.zeros((n_customers + 1, n_customers + 1))     # 初始化距离矩阵

        # 所有点的坐标（最后一个点是仓库）
        points = []
        for i, customer in self.customers.iterrows():
            points.append((customer['XCOORD'], customer['YCOORD']))
        points.append((self.depot['XCOORD'], self.depot['YCOORD']))

        # print(points)

        # 计算所有点对之间的距离
        for i in range(len(points)):
            for j in range(len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                self.distance_matrix[i][j] = distance
        # print(self.distance_matrix)

    def _setup_ga(self):
        #----设置遗传算法参数和操作----
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     #最小化问题，最小适应度
        creator.create("Individual", list, fitness=creator.FitnessMin)         #一个个体（染色体）表示一个可能得解，每个解有一个适应度属性
        self.toolbox = base.Toolbox()

        n_customers = len(self.customers)

        def create_individual():
            # 创建客户访问顺序的随机排列
            customer_order = random.sample(range(n_customers), n_customers)
            # 为每个客户随机分配仓库（0-4）
            depot_assignments = [random.randint(0, 4) for _ in range(n_customers)]
            # 组合成完整个体
            return customer_order + depot_assignments

        self.toolbox.register("indices", random.sample, range(n_customers), n_customers)  # 生成客户索引的随机排列
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)  # 生成染色体序列
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)  # 调用initRepeat, 输入上面生成的染色体序列，输出一个由Individual组成的list
        self.toolbox.register("evaluate", self._evaluate_route)  # 评估函数
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # 选择：锦标赛选择
        self.toolbox.register("mate", tools.cxOrdered)  # 交叉：顺序交叉
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # 变异算子：交换变异

    def _evaluate_route(self, individual):
        #----适应度计算（以总距离作为适应度，越小越优----
        total_distance = 0
        current_load = 0

        # 从仓库0出发到第一个客户
        total_distance += self.distance_matrix[-1][individual[0]] # python中索引-1可以表示最后一个元素
        current_load += self.customers.iloc[individual[0]]['DEMAND']

        for i in range(1, len(individual)):
            current_customer_idx = individual[i]
            prev_customer_idx = individual[i - 1]
            customer_demand = self.customers.iloc[current_customer_idx]['DEMAND']

            # 检查加入这个客户是否会超载
            if current_load + customer_demand <= self.max_capacity:
                # 可以继续当前路线，累加距离和负载
                total_distance += self.distance_matrix[prev_customer_idx][current_customer_idx]
                current_load += customer_demand
            else:
                # 返回仓库并开始新路线
                total_distance += self.distance_matrix[prev_customer_idx][-1]
                total_distance += self.distance_matrix[-1][current_customer_idx]    # 再从仓库出发到当前客户
                current_load = customer_demand      # 重置负载为当前客户需求

        total_distance += self.distance_matrix[individual[-1]][-1]  # 最后返回仓库
        return total_distance,

    # def analyze_route(self, individual):
    #     #----分析路径，返回子路径信息和总距离----
    #     sub_routes = []  # 存储每个子路径的信息
    #     current_route = []  # 当前子路径的客户索引
    #     total_distance = 0
    #     current_load = 0
    #
    #     # 从仓库出发到第一个客户
    #     first_customer_idx = individual[0]
    #     first_customer_no = self.customers.iloc[first_customer_idx]['NO']
    #     total_distance += self.distance_matrix[-1][first_customer_idx]
    #     current_load += self.customers.iloc[first_customer_idx]['DEMAND']
    #     current_route.append(first_customer_no)
    #
    #     sub_routes.append({
    #         'customers': current_route.copy(),
    #         'load': current_load,
    #         'distance_so_far': total_distance
    #     })
    #
    #     for i in range(1, len(individual)):
    #         current_customer_idx = individual[i]
    #         prev_customer_idx = individual[i - 1]
    #         customer_demand = self.customers.iloc[current_customer_idx]['DEMAND']
    #         customer_no = self.customers.iloc[current_customer_idx]['NO']
    #
    #         # 检查加入这个客户是否会超载
    #         if current_load + customer_demand <= self.max_capacity:
    #             # 可以继续当前路线
    #             total_distance += self.distance_matrix[prev_customer_idx][current_customer_idx]
    #             current_load += customer_demand
    #             current_route.append(customer_no)
    #             # 更新当前子路径信息
    #             sub_routes[-1] = {
    #                 'customers': current_route.copy(),
    #                 'load': current_load,
    #                 'distance_so_far': total_distance
    #             }
    #         else:
    #             # 需要返回仓库并开始新路线
    #             total_distance += self.distance_matrix[prev_customer_idx][-1]  # 返回仓库
    #             total_distance += self.distance_matrix[-1][current_customer_idx]  # 从仓库到当前客户
    #             current_load = customer_demand  # 重置负载
    #
    #             # 开始新的子路径
    #             current_route = [customer_no]
    #             sub_routes.append({
    #                 'customers': current_route.copy(),
    #                 'load': current_load,
    #                 'distance_so_far': total_distance
    #             })
    #
    #     # 最后返回仓库
    #     total_distance += self.distance_matrix[individual[-1]][-1]
    #
    #     return sub_routes, total_distance

    def solve(self):
        print("----开始求解VRP----")
        population = self.toolbox.population(n=POPULATION_SIZE)     # 创建初始种群

        # 评估初始种群适应度
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 记录进化过程
        best_fitness = []
        no_improvement_count = 0
        best_so_far = float('inf')

        # 进化循环
        for gen in range(MAX_GENERATIONS):
            # 选择下一代
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # 交叉操作
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_PROB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 变异操作
            for mutant in offspring:
                if random.random() < MUTATION_PROB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 评估新个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 更新种群
            population[:] = offspring

            # 获取当前最优解
            best_ind = tools.selBest(population, 1)[0]
            current_best_fitness = best_ind.fitness.values[0]
            best_fitness.append(current_best_fitness)

            # 早停机制检查（只在达到最小代数后检查）
            if gen >= MIN_GENERATIONS:
                improvement_ratio = abs(best_so_far - current_best_fitness) / (best_so_far + 1e-8)

                if current_best_fitness < best_so_far and improvement_ratio > CONVERGENCE_THRESHOLD:
                    # 有显著改进
                    best_so_far = current_best_fitness
                    no_improvement_count = 0
                else:
                    # 没有显著改进
                    no_improvement_count += 1

                # 检查是否早停
                if no_improvement_count >= EARLY_STOP_PATIENCE:
                    improvement_percent = ((best_fitness[0] - best_so_far) / best_fitness[0]) * 100
                    print(f"早停触发！连续 {EARLY_STOP_PATIENCE} 代没有显著改进")
                    print(f"在第 {gen} 代停止进化")
                    print(f"总改进: {improvement_percent:.1f}% "
                          f"({best_fitness[0]:.1f} → {best_so_far:.1f})")
                    break
            else:
                # 更新最优值但不检查早停
                if current_best_fitness < best_so_far:
                    best_so_far = current_best_fitness
                    no_improvement_count = 0

            # 打印进度
            if gen % 20 == 0:
                status = "探索中" if gen < MIN_GENERATIONS else f"无改进:{no_improvement_count}"
                print(f"第 {gen:3d} 代 | 最优距离: {current_best_fitness:8.2f} | 状态: {status}")

        # 获取最终最优解
        best_solution = tools.selBest(population, 1)[0]
        best_distance = best_solution.fitness.values[0]

        print(f"求解完成！")
        print(f"最终最优路径距离: {best_distance:.2f}")
        print(f"总进化代数: {len(best_fitness)}/{MAX_GENERATIONS}")

        # 计算改进百分比
        initial_best = best_fitness[0] if best_fitness else best_distance
        improvement = ((initial_best - best_distance) / initial_best) * 100
        print(f"相对初始解的改进: {improvement:.1f}%")

        return best_solution, best_fitness

    def visualize_route(self, solution, save_path=None):
        # ----可视化最优路径----
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制仓库
        depot_x, depot_y = self.depot['XCOORD'], self.depot['YCOORD']
        ax.scatter(depot_x, depot_y, c='red', s=200, marker='s', label='depot', edgecolors='black')

        # 绘制客户点
        customer_x = self.customers['XCOORD'].values
        customer_y = self.customers['YCOORD'].values
        ax.scatter(customer_x, customer_y, c='blue', s=50, alpha=0.7, label='客户')

        # 绘制路径
        route_x = [depot_x]
        route_y = [depot_y]

        # 从仓库到第一个客户
        first_customer_idx = solution[0]
        first_customer = self.customers.iloc[first_customer_idx]
        route_x.extend([depot_x, first_customer['XCOORD']])
        route_y.extend([depot_y, first_customer['YCOORD']])

        # 客户之间的路径
        for i in range(len(solution)):
            current_customer = self.customers.iloc[solution[i]]
            route_x.append(current_customer['XCOORD'])
            route_y.append(current_customer['YCOORD'])

        # 最后一个客户返回仓库
        route_x.extend([route_x[-1], depot_x])
        route_y.extend([route_y[-1], depot_y])

        # 绘制路径线
        ax.plot(route_x, route_y, 'g-', alpha=0.6, linewidth=2, label='车辆路径')
        ax.plot(route_x, route_y, 'go', alpha=0.6, markersize=4)

        # 美化图形
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_title('VRP visualization of best route')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加距离信息
        distance = self._evaluate_route(solution)[0]
        ax.text(0.02, 0.98, f'total distance: {distance:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"路径图已保存至: {save_path}")

        plt.show()

    def plot_evolution(self, fitness_history):
        # ----绘制进化过程----
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, 'b-', linewidth=2)
        plt.xlabel('evolution times')#进化代数
        plt.ylabel('best route distance')#最优路径距离
        plt.title('GA process')#遗传算法进化过程
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def main():
    # 创建求解器实例
    solver = VRP_GA_Solver()
    #求解VRP问题
    best_solution, fitness_history = solver.solve()

    # 可视化结果
    solver.visualize_route(best_solution, save_path='optimal_route.png')
    solver.plot_evolution(fitness_history)
    # 打印详细路径信息
    print("\n" + "=" * 50)
    print("最优路径详情:")
    print("=" * 50)

    # 使用新的分析方法
    sub_routes, total_distance = solver.analyze_route(best_solution)

    print(f"总行驶距离: {total_distance:.2f}")
    print(f"总行程数: {len(sub_routes)}")
    print()

    for i, route_info in enumerate(sub_routes, 1):
        customers = route_info['customers']
        load = route_info['load']
        # distance_so_far = route_info['distance_so_far']
        # print(f"行程 {i}:")
        # print(f"  客户序列: {customers}")
        # print(f"  本行程负载: {load}")
        # print(f"  累计距离: {distance_so_far:.2f}")

        # 打印本行程的详细客户信息
        current_load = 0
        for customer_no in customers:
            # 找到对应的客户数据
            customer_data = solver.customers[solver.customers['NO'] == customer_no].iloc[0]
            current_load += customer_data['DEMAND']
            # print(f"    客户 {customer_no}: 需求={customer_data['DEMAND']}, 累计负载={current_load}")

        # print(f"  本行程结束，返回仓库")
        # print()

    # 验证总负载
    # total_demand = solver.customers['DEMAND'].sum()
    # print(f"验证: 总需求={total_demand}, 各行程负载和={sum(route['load'] for route in sub_routes)}")


if __name__ == "__main__":
    print("单车单仓库")
    main()