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
    # 为单辆车（容量200）寻找最优路线，服务100个客户
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

        def create_individual():
            customer_order = random.sample(range(n_customers), n_customers)  # 客户顺序
            depot_assignments = [random.randint(0, 4) for _ in range(n_customers)]  # 仓库顺序
            return customer_order + depot_assignments

        self.toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_route)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self._custom_crossover)
        self.toolbox.register("mutate", self._custom_mutation)

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


def main():
    # 创建求解器实例
    solver = VRP_GA_Solver()
    #求解VRP问题
    best_solution, fitness_history = solver.solve()

    # 可视化结果
    solver.visualize_route(best_solution, save_path='optimal_route.png')
    solver.plot_evolution(fitness_history)


if __name__ == "__main__":
    print("task3")
    main()