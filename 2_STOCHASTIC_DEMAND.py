import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from config import *
from data_process import data_processor
from visualizer import VRPVisualizer
from utils import EarlyStopper

class VRP_GA_Solver:

    def __init__(self):
        self.max_capacity = MAX_CAPACITY

        # 使用数据加载器
        self.data_processor = data_processor
        self.data_processor.load_data()
        self.data_processor.compute_distance_matrix()

        # 获取数据
        self.customers = self.data_processor.get_customers()
        self.depots = self.data_processor.get_depots()
        self.main_depot = self.data_processor.get_main_depot()
        self.distance_matrix = self.data_processor.get_distance_matrix()
        self.depot_indices = self.data_processor.get_depot_indices()

        # 初始化可视化器和早停器
        self.visualizer = VRPVisualizer(self.data_processor)
        self.early_stopper = EarlyStopper(
            patience=EARLY_STOP_PATIENCE,
            threshold=CONVERGENCE_THRESHOLD,
            min_generations=MIN_GENERATIONS
        )
        # 设置遗传算法
        self._setup_ga()

    def _setup_ga(self):
        #----设置遗传算法参数和操作----
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     #最小化问题，最小适应度
        creator.create("Individual", list, fitness=creator.FitnessMin)         #一个个体（染色体）表示一个可能得解，每个解有一个适应度属性
        self.toolbox = base.Toolbox()

        n_customers = len(self.customers)

        #染色体编码方式
        def create_individual():
            customer_order = random.sample(range(n_customers), n_customers)     #客户顺序
            depot_assignments = [random.randint(0, 4) for _ in range(n_customers)]  #n个仓库顺序
            return customer_order + depot_assignments #前n为客户顺序，后n为仓库顺序

        self.toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_route)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self._custom_crossover)
        self.toolbox.register("mutate", self._custom_mutation)

    def _generate_demand(self, mean_demand):
        #-----正态分布范围内生成需求-------
        std_dev = 0.2 * mean_demand     #标准差
        random_demand = np.random.normal(mean_demand, std_dev)      # 生成正态分布随机数
        random_demand = max(1, int(round(random_demand)))       # 截断为正整数
        return random_demand


    def _evaluate_single_sample(self, individual):
        # 适应度评估（单次随机需求）
        n_customers = len(self.customers)
        customer_order = individual[:n_customers]
        depot_assignments = individual[n_customers:]

        total_distance = 0
        current_load = 0
        depot_0_idx = self.depot_indices[0]
        current_position = depot_0_idx

        for i, customer_idx in enumerate(customer_order):
            # 生成随机需求
            mean_demand = self.customers.iloc[customer_idx]['DEMAND']
            actual_demand = self._generate_demand(mean_demand)

            assigned_depot_no = depot_assignments[customer_idx]
            assigned_depot_idx = self.depot_indices[assigned_depot_no]

            # 检查容量约束
            if current_load + actual_demand > self.max_capacity:
                # 返回主仓库清空
                total_distance += self.distance_matrix[current_position][depot_0_idx]
                current_load = 0
                current_position = depot_0_idx

            # 如果当前不在客户分配的仓库，先去该仓库
            if current_position != assigned_depot_idx:
                total_distance += self.distance_matrix[current_position][assigned_depot_idx]
                current_position = assigned_depot_idx

            # 从仓库到客户
            total_distance += self.distance_matrix[current_position][customer_idx]
            current_load += actual_demand
            current_position = customer_idx

        # 最后返回主仓库
        total_distance += self.distance_matrix[current_position][depot_0_idx]

        return total_distance

    def _evaluate_route(self, individual, num_samples=10):
        # 随机需求下的适应度评估，通过多次采样求期望距离
        total_samples_distance = 0

        for sample in range(num_samples):
            sample_distance= self._evaluate_single_sample(individual)
            total_samples_distance += sample_distance
        distance = total_samples_distance/num_samples
        return distance,


    def _custom_crossover(self, ind1, ind2):
        #自定义交叉操作
        n_customers = len(self.customers)

        customer1 = ind1[:n_customers]
        customer2 = ind2[:n_customers]

        depot1 = ind1[n_customers:]
        depot2 = ind2[n_customers:]

        # 客户-顺序交叉
        child1_customer, child2_customer = tools.cxOrdered(customer1, customer2)

        # 仓库-分配交叉
        for i in range(len(depot1)):
            if random.random() < 0.5:
                depot1[i], depot2[i] = depot2[i], depot1[i]

        # 组合
        ind1[:] = child1_customer + depot1
        ind2[:] = child2_customer + depot2
        return ind1, ind2

    def _custom_mutation(self, individual):
        #自定义变异操作
        n_customers = len(self.customers)

        # 客户-顺序变异：交换两个客户位置
        if random.random() < 0.1:
            idx1, idx2 = random.sample(range(n_customers), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        # 仓库-分配变异：随机改变一个客户的仓库分配
        if random.random() < 0.1:
            idx = random.randint(0, n_customers - 1)
            individual[n_customers + idx] = random.randint(0, 4)

        return individual,

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
            should_stop, stop_info = self.early_stopper.should_stop(gen, current_best_fitness, best_fitness)

            if should_stop:
                print(f"早停触发！连续 {stop_info['patience']} 代没有显著改进")
                print(f"在第 {gen} 代停止进化")
                print(f"总改进: {stop_info['improvement_percent']:.1f}% "
                      f"({best_fitness[0]:.1f} → {stop_info['best_fitness']:.1f})")
                break

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
        self.visualizer.visualize_route(solution, save_path)

    def plot_evolution(self, fitness_history, title="GAProcess"):
        self.visualizer.plot_evolution(fitness_history, title)


def main():
    # 创建求解器实例
    solver = VRP_GA_Solver()

    # 求解VRP问题
    best_solution, fitness_history = solver.solve()

    # 可视化结果
    solver.visualize_route(best_solution, save_path='optimal_route.png')
    solver.plot_evolution(fitness_history)


if __name__ == "__main__":
    print("ass2")
    main()