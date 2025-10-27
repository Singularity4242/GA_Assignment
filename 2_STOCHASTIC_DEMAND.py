import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from config import *
from data_process import data_processor
from visualizer import VRPVisualizer

class VRP_GA_Solver:

    def __init__(self):
        self.max_capacity = MAX_CAPACITY
        self.data_processor = data_processor
        self.data_processor.load_data()

        self.customers = self.data_processor.get_customers()
        self.depots = self.data_processor.get_depots()
        self.main_depot = self.data_processor.get_main_depot()
        self.dist_matrix = self.data_processor.cul_dist_matrix()
        self.depot_idx = self.data_processor.get_depot_indices()
        self.visualizer = VRPVisualizer(self.data_processor)

        self._setup_ga()

    def _setup_ga(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     #最小化问题，最小适应度
        creator.create("Individual", list, fitness=creator.FitnessMin)         #一个个体（染色体）表示一个可能得解，每个解有一个适应度属性
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_route)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self._custom_crossover)
        self.toolbox.register("mutate", self._custom_mutation)

    def _create_individual(self):
        # 染色体编码方式
        n_customers = len(self.customers)
        customer_order = random.sample(range(n_customers), n_customers)  # 客户顺序
        depot_assignments = [random.randint(0, 4) for _ in range(n_customers)]  # n个仓库顺序
        return customer_order + depot_assignments  # 前n为客户顺序，后n为仓库顺序

    def _generate_demand(self, mean_demand):
        #生成需求
        std_dev = 0.2 * mean_demand
        random_demand = np.random.normal(mean_demand, std_dev)
        random_demand = max(1, int(round(random_demand)))
        return random_demand


    def _evaluate_route(self, individual):
        n_customers = len(self.customers)
        cust_order = individual[:n_customers]
        cust_depots_order = individual[n_customers:]

        total_distance = 0
        cur_load = 0
        cur_depot_idx = self.depot_idx[0]
        cur_position = self.depot_idx[0]

        for i, customer_idx in enumerate(cust_order):
            cust_mean_demand = self.customers.iloc[customer_idx]['DEMAND']
            cust_sample_demand = self._generate_demand(cust_mean_demand)

            cust_depot_no = cust_depots_order[customer_idx]
            cust_depot_idx = self.depot_idx[cust_depot_no]

            # 如果再送会超载，或者和上一个不是同一个仓库，都需要先前往仓库
            if (cust_depot_idx != cur_depot_idx) or (cur_load + cust_sample_demand > self.max_capacity):
                total_distance += self.dist_matrix[cur_position][cust_depot_idx]
                cur_depot_idx = cust_depot_idx
                cur_position = cur_depot_idx
                cur_load = 0

            total_distance += self.dist_matrix[cur_position][customer_idx]
            cur_load += cust_sample_demand
            cur_position = customer_idx
        total_distance += self.dist_matrix[cur_position][self.depot_idx[0]]

        return total_distance,

    def _custom_crossover(self, ind1, ind2):
        n_customers = len(self.customers)
        customer1 = ind1[:n_customers]
        customer2 = ind2[:n_customers]
        depot1 = ind1[n_customers:]
        depot2 = ind2[n_customers:]

        child1_customer, child2_customer = tools.cxOrdered(customer1, customer2)
        for i in range(len(depot1)):
            if random.random() < 0.5:
                depot1[i], depot2[i] = depot2[i], depot1[i]

        ind1[:] = child1_customer + depot1
        ind2[:] = child2_customer + depot2
        return ind1, ind2

    def _custom_mutation(self, individual):
        n_customers = len(self.customers)
        if random.random() < 0.1:
            idx1, idx2 = random.sample(range(n_customers), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        if random.random() < 0.1:
            idx = random.randint(0, n_customers - 1)
            individual[n_customers + idx] = random.randint(0, 4)

        return individual,

    def solve(self):
        population = self.toolbox.population(n=POPULATION_SIZE)
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        best_fitness = []
        no_improvement_count = 0
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
                #status = "探索中" if gen < MIN_GENERATIONS else f"无改进:{no_improvement_count}"
                print(f" {gen:3d} genration | total distance: {current_best_fitness:8.2f}")

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
        self.visualizer.visualize_route(solution, best_distance)

    def evo_process(self, fitness_history, title="GAProcess"):
        self.visualizer.visualize_process(fitness_history, title)


def main():
    solver = VRP_GA_Solver()
    best_solution, fitness_history, best_distance = solver.solve()
    solver.visualize(best_solution, best_distance)
    solver.evo_process(fitness_history)


if __name__ == "__main__":
    print("----task2 running---")
    main()