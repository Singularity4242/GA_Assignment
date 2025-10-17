import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms
from config import *
from data_process import data_processor
from visualizer import VRPVisualizer
from utils import EarlyStopper
from sklearn.cluster import KMeans

class VRP_GA_Solver:
    # 使用Kmean分类然后
    def __init__(self):
        self.max_capacity = MAX_CAPACITY

        # 使用数据加载器
        self.data_processor = data_processor
        self.data_processor.load_data()
        self.data_processor.compute_distance_matrix()

        # 获取数据
        self.customers = self.data_processor.get_task3_customer()
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
        #self._setup_ga()


    def _setup_ga(self):
        #----设置遗传算法参数和操作----
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     #最小化问题，最小适应度
        creator.create("Individual", list, fitness=creator.FitnessMin)         #一个个体（染色体）表示一个可能得解，每个解有一个适应度属性
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_route)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self._custom_crossover)
        self.toolbox.register("mutate", self._custom_mutation)

    # 聚类
    def _cluster_customers(self, n_clusters=5):
        customer_coords = self.customers[['XCOORD','YCOORD']].values
        kmeans = KMeans(n_clusters = n_clusters, random_state= 42)
        cluster_labels = kmeans.fit_predict(customer_coords)
        self.cluster = {}
        self.cluster_centers = kmeans.cluster_centers_
        for i, label in enumerate(cluster_labels):
            if label not in self.clusters:
                self.clusters[label] = []
            self.cluster[label].append(i)
        return

    # 染色体编码方式
    def _create_individual(self):
        individual = []
        #簇内访问顺序
        for cluster_id in self.clusters:
            customers_in_cluster = self.cluster[cluster_id]
            random.shuffle(customers_in_cluster)
            individual.extend(customers_in_cluster)

        cluster_order = list(range(len(self.cluters)))
        random.shuffle(cluster_order)
        individual.extend(cluster_order)
        #簇间访问顺序
        return individual

    def _get_cluster_of_customer(self, customer_idx):
        for cluster_id, customers in self.clusters.items():
            if customer_idx in customers:
                return cluster_id
        return -1

    #重写适应度计算
    def _evaluate_route(self, individual):
        total_distance = 0
        current_position = self.depot_indices[0]

        #解析染色体
        n_customers = len(self.customers)
        cluster_order = individual[n_customers:]
        customer_order = individual[:n_customers]

        for cluster_id in cluster_order:
            cluster_customers = [cust for cust in customer_order
                                 if self._get_cluster_of_customer(cust) == cluster_id]
            #for customer_idx in cluster_customers:
        #计算距离，检查容量
        return total_distance

    def solve(self):
        print("----开始求解VRP----")
        return


def main():
    # 求解器实例
    solver = VRP_GA_Solver()
    # best_solution, fitness_history = solver.solve()
    #
    # # 可视化结果
    # solver.visualize_route(best_solution, save_path='optimal_route.png')
    # solver.plot_evolution(fitness_history)


if __name__ == "__main__":
    print("task3")
    main()