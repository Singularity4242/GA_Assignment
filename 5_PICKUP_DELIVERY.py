import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from config import *
from data_process import data_processor
from visualizer import VRPVisualizer

class VRP_GA_Solver:
    # 1辆车 1个仓库 取货送货
    def __init__(self):
        self.max_capacity = MAX_CAPACITY
        self.data_processor = data_processor
        self.data_processor.load_data()
        self.customers = self.data_processor.get_task5_customer()
        self.dist_matrix = self.data_processor.cul_dist_matrix()
        self.depots_idx = self.data_processor.get_depot_indices()   #depots[0] =100,[1] = 101
        self.visualizer = VRPVisualizer(self.data_processor)
        self._setup_ga()

    def _setup_ga(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     #最小化问题
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_route)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)

    def _create_individual(self):
        n_customers = len(self.customers)
        customer_order = random.sample(range(n_customers), n_customers)  # 客户顺序
        return customer_order

    #适应度计算
    def _evaluate_route(self, individual):
        n_customers = len(self.customers)
        cust_order = individual[:n_customers]
        total_distance = 0
        cur_load = 0
        cur_position = self.depots_idx[0]

        for i, cust_idx in enumerate(cust_order):
            cust_demand = self.customers.iloc[cust_idx]['DEMAND']
            #操作后的总容量大于200，返回
            if (abs(cur_load + cust_demand) > self.max_capacity):
                total_distance += self.dist_matrix[cur_position][self.depots_idx[0]]
                cur_position = self.depots_idx[0]
                cur_load = 0

            # 从当前仓库到客户
            total_distance += self.dist_matrix[cur_position][cust_idx]
            cur_load += cust_demand
            cur_position = cust_idx

        # 返回主仓库
        total_distance += self.dist_matrix[cur_position][self.depots_idx[0]]

        return total_distance,


    def solve(self):
        population = self.toolbox.population(n=POPULATION_SIZE)

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        best_fitness = []

        # 进化
        for gen in range(MAX_GENERATIONS):
            # 选择
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_PROB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            # 变异
            for mutant in offspring:
                if random.random() < MUTATION_PROB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # 评估
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # 更新
            population[:] = offspring

            best_ind = tools.selBest(population, 1)[0]
            current_best_fitness = best_ind.fitness.values[0]
            best_fitness.append(current_best_fitness)

            if gen % 20 == 0:
                print(f"{gen:3d} genration | best distance: {current_best_fitness:8.2f}")

        # 获取最终最优解
        best_solution = tools.selBest(population, 1)[0]
        best_distance = best_solution.fitness.values[0]
        print(f"final total distance: {best_distance:.2f}")
        # 计算改进百分比
        initial_best = best_fitness[0] if best_fitness else best_distance
        improvement = ((initial_best - best_distance) / initial_best) * 100
        print(f"Percent improvement: {improvement:.1f}%")

        return best_solution, best_fitness, best_distance

    def visualize(self, solution, best_distance):
        self.visualizer.visualize_route_task5(solution, best_distance)

    def evo_process(self, fitness_history, title="GAProcess"):
        self.visualizer.visualize_process(fitness_history, title)


def main():
    solver = VRP_GA_Solver()
    print(solver.customers)
    best_solution, fitness_history, best_distance = solver.solve()

    solver.visualize(best_solution, best_distance)
    solver.evo_process(fitness_history)


if __name__ == "__main__":
    print("---task 5 running----")
    main()