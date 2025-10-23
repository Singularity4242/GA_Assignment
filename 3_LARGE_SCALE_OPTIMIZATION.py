import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from config import *
from data_process import data_processor
from visualizer import VRPVisualizer
#from utils import EarlyStopper
from sklearn.cluster import KMeans

class VRP_GA_Solver:
    # 使用Kmean分类然后.
    def __init__(self):
        self.max_capacity = MAX_CAPACITY
        self.data_processor = data_processor
        self.data_processor.load_data()

        self.customers = self.data_processor.get_task3_customer()
        self.depots = self.data_processor.get_depots()
        self.main_depot = self.data_processor.get_main_depot()
        self.dist_matrix = self.data_processor.cul_dist_matrix()
        self.depot_idx = self.data_processor.get_depot_indices()
        self.visualizer = VRPVisualizer(self.data_processor)
        self._cluster_customers(n_clusters=5)

        self._setup_ga()


    def _setup_ga(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_route)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self._custom_crossover)
        self.toolbox.register("mutate", self._custom_mutation)

    # 聚类
    def _cluster_customers(self, n_clusters=5):
        cust_coords = self.customers[['XCOORD','YCOORD']].values
        kmeans = KMeans(n_clusters = n_clusters, random_state= 42)
        cluster_labels = kmeans.fit_predict(cust_coords)
        self.clusters = {}      #字典，键为簇编号，值为所在该簇的客户索引数组
        self.cluster_centers = kmeans.cluster_centers_
        for i, label in enumerate(cluster_labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(i)
        return

    # 染色体编码方式
    def _create_individual(self):
        individual = []
        #簇内访问顺序
        for cluster_id in sorted(self.clusters.keys()):
            customers_in_cluster = self.clusters[cluster_id].copy()
            random.shuffle(customers_in_cluster)
            individual.extend(customers_in_cluster)

        cluster_order = list(range(len(self.clusters)))
        random.shuffle(cluster_order)
        individual.extend(cluster_order)
        #簇间访问顺序
        return individual

    def _get_cluster_of_customer(self, customer_idx):
        for cluster_id, customers in self.clusters.items():
            if customer_idx in customers:
                return cluster_id
        return -1

    def _evaluate_route(self, individual):
        total_distance = 0
        cur_load = 0
        main_depot = self.depot_idx[0]  # 主仓库索引

        #解析染色体
        n_customers = len(self.customers)
        cluster_order = individual[n_customers:]
        cust_order = individual[:n_customers]
        cur_position = main_depot

        for cluster_id in cluster_order:
            cluster_customers = [cust for cust in cust_order
                                 if self._get_cluster_of_customer(cust) == cluster_id]
            for cust_idx in cluster_customers:
                cust_demand = self.customers.iloc[cust_idx]['DEMAND']
                if cur_load + cust_demand > self.max_capacity:
                    total_distance += self.dist_matrix[cur_position][main_depot]
                    cur_position = main_depot
                    cur_load = 0

                total_distance += self.dist_matrix[cur_position][cust_idx]
                cur_load += cust_demand
                cur_position = cust_idx

        total_distance += self.dist_matrix[cur_position][main_depot]
        return total_distance,

    def _custom_crossover(self, ind1, ind2):
        n_customers = len(self.customers)

        customer1 = ind1[:n_customers]
        customer2 = ind2[:n_customers]
        child1_customer, child2_customer = tools.cxOrdered(customer1, customer2)

        cluster1 = ind1[n_customers:]
        cluster2 = ind2[n_customers:]
        child1_cluster, child2_cluster = tools.cxOrdered(cluster1, cluster2)

        ind1[:] = child1_customer + child1_cluster
        ind2[:] = child2_customer + child2_cluster
        return ind1, ind2

    def _custom_mutation(self, individual):
        n_customer = len(self.customers)
        if random.random() < 0.1:
            cluster_id = random.choice(list(self.clusters.keys()))
            cluter_customers = self.clusters[cluster_id]
            if len(cluter_customers) >= 2:
                idx1, idx2 = random.sample(cluter_customers, 2)
                pos1 = individual.index(idx1)
                pos2 = individual.index(idx2)
                individual[pos1], individual[pos2] = individual[pos2], individual[pos1]

        if random.random() < 0.1:
            idx1, idx2 = random.sample(range(n_customer, len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual,

    def solve(self):
        print("----开始求解ass1:VRP----")
        population = self.toolbox.population(n=POPULATION_SIZE)  # 创建初始种群

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 记录进化过程
        best_fitness = []
        no_improvement_count = 0

        for gen in range(MAX_GENERATIONS):
            # 选择下一代
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

            # 获取当前最优解
            best_ind = tools.selBest(population, 1)[0]
            current_best_fitness = best_ind.fitness.values[0]
            best_fitness.append(current_best_fitness)

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

        return best_solution, best_fitness, best_distance

    def visualize_route(self, solution, best_distance):
        self.visualizer.clusters = self.clusters
        self.visualizer.cluster_centers = self.cluster_centers
        self.visualizer.visualize_route_task3(solution, self.clusters, self.cluster_centers, best_distance)

    def plot_evolution(self, fitness_history, title="GAProcess"):
        self.visualizer.visualize_process(fitness_history, title)


def main():
    solver = VRP_GA_Solver()
    best_solution, fitness_history, best_distance = solver.solve()
    solver.visualize_route(best_solution, best_distance)
    solver.plot_evolution(fitness_history)


if __name__ == "__main__":
    print("task3")
    main()