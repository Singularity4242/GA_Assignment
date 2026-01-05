import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from config import *
from data_process import data_processor
from visualizer import VRPVisualizer
#from utils import EarlyStopper

class VRP_GA_Solver:
    # 为1辆车（容量200）5个仓库寻找最优路线，服务100个客户
    def __init__(self):
        self.max_capacity = MAX_CAPACITY
        self.data_processor = data_processor
        self.data_processor.load_data()

        self.customers = self.data_processor.get_customers()
        self.dist_matrix = self.data_processor.cul_dist_matrix()
        self.depots_idx = self.data_processor.get_depot_indices()   #depots[0] =100,[1] = 101
        self.visualizer = VRPVisualizer(self.data_processor)
        # self.early_stopper = EarlyStopper(
        #     patience=EARLY_STOP_PATIENCE,
        #     threshold=CONVERGENCE_THRESHOLD,
        #     min_generations=MIN_GENERATIONS
        # )
        self._setup_ga()

    def _setup_ga(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     #最小化问题，最小适应度
        creator.create("Individual", list, fitness=creator.FitnessMin)
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
        customer_order = random.sample(range(n_customers), n_customers)
        depot_assignments = [random.randint(0, 4) for _ in range(n_customers)]
        individual = customer_order + depot_assignments
        return individual  # 前n为客户顺序，后n为仓库顺序


    #适应度计算
    def _evaluate_route(self, individual):

        n_customers = len(self.customers)
        # 前n客户顺序，后n仓库分配
        cust_order = individual[:n_customers]
        cust_depots_order = individual[n_customers:]

        total_distance = 0
        cur_load = 0
        cur_depot_idx = self.depots_idx[0]
        cur_position = self.depots_idx[0]

        for i, cust_idx in enumerate(cust_order):
            cust_depot_no = cust_depots_order[cust_idx]
            cust_depot_idx = self.depots_idx[cust_depot_no]
            cust_demand = self.customers.iloc[cust_idx]['DEMAND']
            #如果再送会超载，或者和上一个不是同一个仓库，都需要先前往仓库
            if (cust_depot_idx != cur_depot_idx) or (cur_load + cust_demand > self.max_capacity):
                total_distance += self.dist_matrix[cur_position][cust_depot_idx]
                cur_depot_idx = cust_depot_idx
                cur_position = cur_depot_idx
                cur_load = 0
            # 从当前仓库到客户
            total_distance += self.dist_matrix[cur_position][cust_idx]
            cur_load += cust_demand
            cur_position = cust_idx
        # 最后返回主仓库
        total_distance += self.dist_matrix[cur_position][self.depots_idx[0]]

        return total_distance,

    def _custom_crossover(self, ind1, ind2):
        #交叉
        n_customers = len(self.customers)
        customer1 = ind1[:n_customers]
        customer2 = ind2[:n_customers]
        depot1 = ind1[n_customers:]
        depot2 = ind2[n_customers:]
        # 客户
        child1_customer, child2_customer = tools.cxOrdered(customer1, customer2)
        # 仓库分配
        for i in range(len(depot1)):
            if random.random() < 0.5:
                depot1[i], depot2[i] = depot2[i], depot1[i]
        ind1[:] = child1_customer + depot1
        ind2[:] = child2_customer + depot2
        return ind1, ind2

    def _custom_mutation(self, individual):
        #变异
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

        # 进化循环
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
            #评估
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            #精英保留
            current_best = tools.selBest(population, 1)[0]
            current_worst = tools.selWorst(offspring, 1)[0]
            offspring.remove(current_worst)
            offspring.append(current_best)
            # 更新种群
            population[:] = offspring

            best_ind = tools.selBest(population, 1)[0]
            current_best_fitness = best_ind.fitness.values[0]
            best_fitness.append(current_best_fitness)

            # should_stop, stop_info = self.early_stopper.should_stop(gen, current_best_fitness, best_fitness)
            #
            # if should_stop:
            #     print(f"早停触发！连续 {stop_info['patience']} 代没有显著改进")
            #     print(f"在第 {gen} 代停止进化")
            #     print(f"总改进: {stop_info['improvement_percent']:.1f}% "
            #           f"({best_fitness[0]:.1f} → {stop_info['best_fitness']:.1f})")
            #     break

            if gen % 20 == 0:
                #status = "探索中" if gen < MIN_GENERATIONS else f"无改进:{no_improvement_count}"
                print(f" {gen:3d} generation | total distance: {current_best_fitness:8.2f}")
        best_solution = tools.selBest(population, 1)[0]
        best_distance = best_solution.fitness.values[0]
        print(f"Final optinal total distance: {best_distance:.2f}")

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
    print("----Task 1 running---")
    main()