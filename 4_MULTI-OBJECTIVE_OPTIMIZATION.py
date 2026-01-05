import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from config import *
from data_process import data_processor
from visualizer import VRPVisualizer

class VRP_GA_Solver:
    # 为1辆车（容量200）5个仓库寻找最优路线，服务100个客户
    def __init__(self, mode= 'weighted', weight = 1.0):
        self.max_capacity = MAX_CAPACITY
        self.data_processor = data_processor
        self.data_processor.load_data()

        self.customers = self.data_processor.get_customers()
        self.depots = self.data_processor.get_depots()
        self.main_depot = self.data_processor.get_main_depot()
        self.dist_matrix = self.data_processor.cul_dist_matrix()
        self.depots_idx = self.data_processor.get_depot_indices()
        self.visualizer = VRPVisualizer(self.data_processor)

        self.mode = mode
        self.weight = weight
        self._setup_ga()

    def _setup_ga(self):
        if self.mode == 'weighted':
            print("---weighted running---")
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     #最小化问题，最小适应度
            creator.create("Individual", list, fitness=creator.FitnessMin)         #一个个体（染色体）表示一个可能得解，每个解有一个适应度属性
        elif self.mode == 'nsgaii':
            print("---nsgaii running---")
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # 最小化问题，最小适应度
            creator.create("Individual", list, fitness=creator.FitnessMulti)  # 一个个体（染色体）表示一个可能得解，每个解有一个适应度属性

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._get_fitness)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self._custom_crossover)
        self.toolbox.register("mutate", self._custom_mutation)

    def _create_individual(self):
        # 染色体编码方式
        n_customers = len(self.customers)
        customer_order = random.sample(range(n_customers), n_customers)  # 客户顺序
        depot_assignments = [random.randint(0, 4) for _ in range(n_customers)]  # n个仓库顺序
        return customer_order + depot_assignments  # 前n为客户顺序，后n为仓库顺序


    #距离计算[适应度]
    def _get_fitness(self, individual):
        n_customers = len(self.customers)
        cust_order = individual[:n_customers]
        depot_assignments = individual[n_customers:]

        total_distance = 0
        total_efficiency = 0
        cur_load = 0
        cur_depot_idx = self.depots_idx[0]
        cur_position = self.depots_idx[0]
        di = 0

        # 按客户顺序访问，但需要根据仓库分配组织路线
        for i, cust_idx in enumerate(cust_order):
            cust_depot_no = depot_assignments[cust_idx]
            cust_depot_idx = self.depots_idx[cust_depot_no]
            cust_demand = self.customers.iloc[cust_idx]['DEMAND']
            eff_score = self.customers.iloc[cust_idx]['EFFICIENCY']

            # 如果再送会超载，或者和上一个不是同一个仓库，都需要先前往仓库
            if (cust_depot_idx != cur_depot_idx) or (cur_load + cust_demand > self.max_capacity):
                total_distance += self.dist_matrix[cur_position][cust_depot_idx]
                cur_depot_idx = cust_depot_idx
                cur_position = cur_depot_idx
                cur_load = 0
                di = 0

            # 从当前仓库到客户
            total_distance += self.dist_matrix[cur_position][cust_idx]
            di += self.dist_matrix[cur_position][cust_idx]
            cur_load += cust_demand
            cur_position = cust_idx

            #计算效率
            customer_eff = eff_score - di
            total_efficiency += customer_eff

        # 最后返回主仓库
        total_distance += self.dist_matrix[cur_position][self.depots_idx[0]]

        # 保存原始目标值到个体属性中
        individual.raw_distance = total_distance
        individual.raw_efficiency = total_efficiency

        if self.mode == 'weighted':
            fitness = self._weighted(total_distance, total_efficiency, self.weight)
            return fitness,
        elif self.mode == 'nsgaii':
            return total_distance, total_efficiency

    def _weighted(self, f1, f2, w):
        fitness = w * f1 - (1 - w) * f2
        return fitness


    def _custom_crossover(self, ind1, ind2):
        #交叉
        n_customers = len(self.customers)

        # 对客户顺序部分使用顺序交叉
        customer1 = ind1[:n_customers]
        customer2 = ind2[:n_customers]

        # 对仓库分配部分使用均匀交叉
        depot1 = ind1[n_customers:]
        depot2 = ind2[n_customers:]

        # 客户顺序交叉
        child1_customer, child2_customer = tools.cxOrdered(customer1, customer2)

        # 仓库分配交叉
        for i in range(len(depot1)):
            if random.random() < 0.5:
                depot1[i], depot2[i] = depot2[i], depot1[i]

        # 组合成新个体
        ind1[:] = child1_customer + depot1
        ind2[:] = child2_customer + depot2

        return ind1, ind2

    def _custom_mutation(self, individual):
        #变异
        n_customers = len(self.customers)

        # 客户顺序变异：交换两个客户位置
        if random.random() < 0.1:
            idx1, idx2 = random.sample(range(n_customers), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        # 仓库分配变异：随机改变一个客户的仓库分配
        if random.random() < 0.1:
            idx = random.randint(0, n_customers - 1)
            individual[n_customers + idx] = random.randint(0, 4)

        return individual,

    def solve(self):
        population = self.toolbox.population(n=POPULATION_SIZE)     # 创建初始种群

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        best_fitness = []

        for gen in range(MAX_GENERATIONS):
            # 选择
            if self.mode == 'nsgaii':
                offspring = tools.selNSGA2(population, len(population))
            else:
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

            # 评估新个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 更新种群
            population[:] = offspring

            if self.mode == 'nsgaii':
                best_ind = tools.selBest(population, 1)[0]
                current_best_fitness = best_ind.fitness.values
            else:
                best_ind = tools.selBest(population, 1)[0]
                current_best_fitness = best_ind.fitness.values[0]
            best_fitness.append(current_best_fitness)

            if gen % 10 == 0:
                if self.mode == 'weighted':
                    best_ind = tools.selBest(population, 1)[0]
                    current_distance = best_ind.raw_distance
                    current_efficiency = best_ind.raw_efficiency
                    current_fitness = best_ind.fitness.values[0]
                    print(f"{gen:3d} generation | weighted fitness: {current_fitness:8.2f} | distance: {current_distance:8.2f} | efficiency: {current_efficiency:8.2f}")
        # 获取最终最优解
        if self.mode == 'nsgaii':
           pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
           return pareto_front, best_fitness
        else:
            best_solution = tools.selBest(population, 1)[0]
            #best_solution_value = best_solution.fitness.values[0]
            #best_distance = best_solution.raw_distance
        return best_solution, best_fitness

    # def visualize_route(self, solution, best_distance):
    #     self.visualizer.visualize_route(solution, best_distance)
    #
    # def plot_evolution(self, fitness_history, title="GAProcess"):
    #     self.visualizer.visualize_process(fitness_history, title)
    #
    # def visualize_pareto_front(self, pareto_front):
    #     #可视化帕累托前沿
    #     # 提取目标值
    #     distances = [ind.fitness.values[0] for ind in pareto_front]
    #     efficiencies = [ind.fitness.values[1] for ind in pareto_front]
    #
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(distances, efficiencies, c='blue', alpha=0.7)
    #     plt.xlabel('Total Distance (f1)')
    #     plt.ylabel('Total Efficiency (f2)')
    #     plt.title('Pareto Front - NSGA-II')
    #     plt.grid(True, alpha=0.3)
    #     # 标记一些特殊解
    #     min_dist_idx = np.argmin(distances)
    #     max_eff_idx = np.argmax(efficiencies)
    #     plt.scatter(distances[min_dist_idx], efficiencies[min_dist_idx],
    #                 c='red', s=100, label='Min Distance')
    #     plt.scatter(distances[max_eff_idx], efficiencies[max_eff_idx],
    #                 c='green', s=100, label='Max Efficiency')
    #     plt.legend()
    #     plt.show()


def main():
    method = input("Choose weighted/nsgaii：")
    if method == 'weighted':
        solver = VRP_GA_Solver(mode = 'weighted', weight=0.7)
        best_solution, fitness_history = solver.solve()
        #solver.visualize_route(best_solution, best_distance)
        #solver.plot_evolution(fitness_history)
        print(f"---Final Result----")
        print(f"total distance: {best_solution.raw_distance:.2f}")
        print(f"total efficiency: {best_solution.raw_efficiency:.2f}")
        print(f"fitness: {best_solution.fitness.values[0]:.2f}")
    elif method == 'nsgaii':
        solver = VRP_GA_Solver(mode='nsgaii')
        pareto_front, best_fitness= solver.solve()
        # 多目标可视化
        #solver.visualize_pareto_front(pareto_front)
        print(f"found {len(pareto_front)} pareto solutions")
        for i, solution in enumerate(pareto_front):
            distance = solution.fitness.values[0]
            efficiency = solution.fitness.values[1]
            print(f"Solution {i + 1:2d}: Total Distance = {distance:8.2f}, Total Efficiency = {efficiency:8.2f}")
        #solver.visualize_route(pareto_front[0])



if __name__ == "__main__":
    print("---Task 4 running----")
    main()