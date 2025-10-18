import matplotlib.pyplot as plt
import numpy as np


class VRPVisualizer:

    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.customers = data_processor.get_customers()
        self.depots = data_processor.get_depots()
        self.distance_matrix = data_processor.get_distance_matrix()
        self.depot_indices = data_processor.get_depot_indices()

    def visualize_route(self, solution, save_path=None):
        """可视化最优路径"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制所有仓库
        for i, depot in self.depots.iterrows():
            depot_x, depot_y = depot['XCOORD'], depot['YCOORD']
            color = 'red' if depot['NO'] == 0 else 'orange'
            ax.scatter(depot_x, depot_y, c=color, s=200, marker='s',
                       label=f'depot{depot["NO"]}', edgecolors='black')
            ax.text(depot_x, depot_y + 2, f'depot{depot["NO"]}',
                    ha='center', fontsize=10, weight='bold')

        # 绘制客户点
        customer_x = self.customers['XCOORD'].values
        customer_y = self.customers['YCOORD'].values
        ax.scatter(customer_x, customer_y, c='blue', s=50, alpha=0.7, label='customer')

        # 分析路径获取实际路线
        n_customers = len(self.customers)
        customer_order = solution[:n_customers]
        depot_assignments = solution[n_customers:]

        # 绘制路线
        depot_0_idx = self.depot_indices[0]
        current_position = depot_0_idx
        route_x = [self.depots[self.depots['NO'] == 0].iloc[0]['XCOORD']]
        route_y = [self.depots[self.depots['NO'] == 0].iloc[0]['YCOORD']]

        for i, customer_idx in enumerate(customer_order):
            assigned_depot_no = depot_assignments[customer_idx]
            assigned_depot_idx = self.depot_indices[assigned_depot_no]
            customer = self.customers.iloc[customer_idx]

            # 如果不在分配的仓库，先画到仓库的路线
            if current_position != assigned_depot_idx:
                depot = self.depots[self.depots['NO'] == assigned_depot_no].iloc[0]
                route_x.extend([depot['XCOORD']])
                route_y.extend([depot['YCOORD']])
                current_position = assigned_depot_idx

            # 画到客户的路线
            route_x.extend([customer['XCOORD']])
            route_y.extend([customer['YCOORD']])
            current_position = customer_idx

        # 最后返回主仓库
        main_depot = self.depots[self.depots['NO'] == 0].iloc[0]
        route_x.extend([main_depot['XCOORD']])
        route_y.extend([main_depot['YCOORD']])

        # 绘制路径线
        ax.plot(route_x, route_y, 'g-', alpha=0.6, linewidth=2, label='route')
        ax.plot(route_x, route_y, 'go', alpha=0.6, markersize=4)

        # 美化图形
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_title('MDVRP optimal path visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加距离信息
        distance = self._calculate_route_distance(solution)
        ax.text(0.02, 0.98, f'total distance: {distance:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"路径图已保存至: {save_path}")

        plt.show()

    def _calculate_route_distance(self, solution):
        """计算路径距离（用于可视化）"""
        n_customers = len(self.customers)
        customer_order = solution[:n_customers]
        depot_assignments = solution[n_customers:]

        total_distance = 0
        current_load = 0
        depot_0_idx = self.depot_indices[0]
        current_position = depot_0_idx

        for i, customer_idx in enumerate(customer_order):
            assigned_depot_no = depot_assignments[customer_idx]
            assigned_depot_idx = self.depot_indices[assigned_depot_no]
            customer_demand = self.customers.iloc[customer_idx]['DEMAND']

            if current_load + customer_demand > 200:  # 使用默认容量
                total_distance += self.distance_matrix[current_position][depot_0_idx]
                current_load = 0
                current_position = depot_0_idx

            if current_position != assigned_depot_idx:
                total_distance += self.distance_matrix[current_position][assigned_depot_idx]
                current_position = assigned_depot_idx

            total_distance += self.distance_matrix[current_position][customer_idx]
            current_load += customer_demand
            current_position = customer_idx

        total_distance += self.distance_matrix[current_position][depot_0_idx]
        return total_distance

    def plot_evolution(self, fitness_history, title="GA evolution process"):
        """绘制进化过程"""
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, 'b-', linewidth=2)
        plt.xlabel('Evolution times')
        plt.ylabel('optimal path distance')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_comparison(self, histories, labels, title="算法性能比较"):
        """比较多个算法的进化过程"""
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (history, label) in enumerate(zip(histories, labels)):
            plt.plot(history, color=colors[i % len(colors)], linewidth=2, label=label)

        plt.xlabel('进化代数')
        plt.ylabel('最优路径距离')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_route_task3(self, solution, clusters, cluster_centers=None, save_path=None):
        """第三问可视化：保持原有风格，只添加簇中心点"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制所有仓库（保持原样）
        for i, depot in self.depots.iterrows():
            depot_x, depot_y = depot['XCOORD'], depot['YCOORD']
            color = 'red' if depot['NO'] == 0 else 'orange'
            ax.scatter(depot_x, depot_y, c=color, s=200, marker='s',
                       label=f'depot{depot["NO"]}', edgecolors='black')
            ax.text(depot_x, depot_y + 2, f'depot{depot["NO"]}',
                    ha='center', fontsize=10, weight='bold')

        # 绘制客户点（保持原样，所有客户用相同颜色）
        customer_x = self.customers['XCOORD'].values
        customer_y = self.customers['YCOORD'].values
        ax.scatter(customer_x, customer_y, c='blue', s=50, alpha=0.7, label='customer')

        # 绘制簇中心点（新增）
        if cluster_centers is not None:
            for cluster_id, center in enumerate(cluster_centers):
                center_x, center_y = center
                ax.scatter(center_x, center_y, c='green', s=300, marker='*',
                           edgecolors='black', linewidth=2, label='Cluster Center' if cluster_id == 0 else "")
                ax.text(center_x, center_y + 3, f'Cluster{cluster_id}',
                        ha='center', fontsize=9, weight='bold', style='italic')

        # 解析染色体 - 第三问的结构
        n_customers = len(self.customers)
        customer_order = solution[:n_customers]  # 所有客户顺序
        cluster_order = solution[n_customers:]  # 簇访问顺序

        # 绘制路线（保持原有逻辑，但按簇顺序访问）
        depot_0_idx = self.depot_indices[0]
        current_position = depot_0_idx
        route_x = [self.depots[self.depots['NO'] == 0].iloc[0]['XCOORD']]
        route_y = [self.depots[self.depots['NO'] == 0].iloc[0]['YCOORD']]

        # 按簇顺序访问客户
        for cluster_id in cluster_order:
            # 获取该簇的所有客户（按染色体中的顺序）
            cluster_customers = [
                cust for cust in customer_order
                if self._get_cluster_of_customer(cust, clusters) == cluster_id
            ]

            # 访问该簇内的所有客户
            for customer_idx in cluster_customers:
                customer = self.customers.iloc[customer_idx]

                # 直接画到客户的路线（单仓库，不需要经过其他仓库）
                route_x.extend([customer['XCOORD']])
                route_y.extend([customer['YCOORD']])
                current_position = customer_idx

        # 最后返回主仓库
        main_depot = self.depots[self.depots['NO'] == 0].iloc[0]
        route_x.extend([main_depot['XCOORD']])
        route_y.extend([main_depot['YCOORD']])

        # 绘制路径线（保持原样）
        ax.plot(route_x, route_y, 'g-', alpha=0.6, linewidth=2, label='route')
        ax.plot(route_x, route_y, 'go', alpha=0.6, markersize=4)

        # 美化图形
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_title('Large Scale VRP with Clustering\n(200 Customers, Single Depot)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加距离信息
        distance = self._calculate_clustered_route_distance_task3(solution, clusters)
        ax.text(0.02, 0.98, f'Total Distance: {distance:.2f}\n'
                            f'Clusters: {len(clusters)}\n'
                            f'Customers: {len(self.customers)}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"路径图已保存至: {save_path}")

        plt.show()
        return distance

    def _calculate_clustered_route_distance_task3(self, solution, clusters):
        """计算第三问聚类路径距离"""
        n_customers = len(self.customers)
        customer_order = solution[:n_customers]
        cluster_order = solution[n_customers:]

        total_distance = 0
        current_load = 0
        depot_0_idx = self.depot_indices[0]
        current_position = depot_0_idx

        # 按簇顺序访问
        for cluster_id in cluster_order:
            cluster_customers = [
                cust for cust in customer_order
                if self._get_cluster_of_customer(cust, clusters) == cluster_id
            ]

            # 访问该簇的所有客户
            for customer_idx in cluster_customers:
                customer_demand = self.customers.iloc[customer_idx]['DEMAND']

                # 容量约束检查
                if current_load + customer_demand > 200:  # 使用默认容量
                    total_distance += self.distance_matrix[current_position][depot_0_idx]
                    current_load = 0
                    current_position = depot_0_idx

                total_distance += self.distance_matrix[current_position][customer_idx]
                current_load += customer_demand
                current_position = customer_idx

        total_distance += self.distance_matrix[current_position][depot_0_idx]
        return total_distance

    def _get_cluster_of_customer(self, customer_idx, clusters):
        """辅助方法：获取客户所属簇"""
        for cluster_id, customers in clusters.items():
            if customer_idx in customers:
                return cluster_id
        return -1