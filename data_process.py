import pandas as pd
import numpy as np
from config import DATA_PATH


class DataProcessor:

    def __init__(self):
        self.data = None
        self.customers = None
        self.depots = None
        self.main_depot = None
        self.distance_matrix = None
        self.depot_indices = None

    def load_data(self):
        #加载VRP数据
        print("——————加载VRP数据——————")
        self.data = pd.read_csv(DATA_PATH)

        # 分离客户和仓库
        self.customers = self.data[self.data['CUST_OR_DEPOT'] == 'CUSTOMER'].copy()
        self.depots = self.data[self.data['CUST_OR_DEPOT'] == 'DEPOT'].copy()
        self.main_depot = self.data[
            (self.data['CUST_OR_DEPOT'] == 'DEPOT') & (self.data['NO'] == 0)
            ].iloc[0]

        # print(f"找到 {len(self.customers)} 个客户")
        # print(f"找到 {len(self.depots)} 个仓库")
        # print("仓库信息:")
        # for i, depot in self.depots.iterrows():
        #     print(f"  仓库{depot['NO']}: ({depot['XCOORD']}, {depot['YCOORD']})")
        return self.customers, self.depots, self.main_depot

    #-----计算距离矩阵------
    def compute_distance_matrix(self):
        if self.customers is None or self.depots is None:
            raise ValueError()

        print("——————计算距离矩阵——————")
        n_customers = len(self.customers)
        n_depots = len(self.depots)

        # 初始化距离矩阵
        self.distance_matrix = np.zeros((n_customers + n_depots, n_customers + n_depots))

        points = []     # 所有点的坐标（先是客户，后是仓库）
        # 添加客户坐标
        for i, customer in self.customers.iterrows():
            points.append((customer['XCOORD'], customer['YCOORD']))
        # 按NO排序仓库
        depots_sorted = self.depots.sort_values('NO')
        for i, depot in depots_sorted.iterrows():
            points.append((depot['XCOORD'], depot['YCOORD']))

        # 计算所有点对之间的距离
        for i in range(len(points)):
            for j in range(len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                self.distance_matrix[i][j] = distance

        # 仓库索引映射
        self.depot_indices = {
            int(depot_no): n_customers + i
            for i, depot_no in enumerate(depots_sorted['NO'])
        }

        print(f"仓库索引映射: {self.depot_indices}")
        return self.distance_matrix, self.depot_indices

    #用于task3复制
    def get_task3_customer(self):
        addition_customer = self.customers.copy()
        addition_customer['YCOORD'] += 150
        addition_customer['NO'] += 100
        # 合并
        self.customers = pd.concat([self.customers, addition_customer], ignore_index=True)
        return self.customers


    #-----获取客户数据-----
    def get_customers(self):
        if self.customers is None:
            raise ValueError("self.customers is None")
        return self.customers

    #---------获取仓库数据---------
    def get_depots(self):
        if self.depots is None:
            raise ValueError("depots is None")
        return self.depots

    #------------获取主仓库数据---------
    def get_main_depot(self):
        if self.main_depot is None:
            raise ValueError("main_depot is None")
        return self.main_depot

    #-------------获取距离矩阵
    def get_distance_matrix(self):
        if self.distance_matrix is None:
            raise ValueError("distance_matrix")
        return self.distance_matrix

    def get_depot_indices(self):
        """获取仓库索引映射"""
        if self.depot_indices is None:
            raise ValueError("请先调用 compute_distance_matrix() 方法计算距离矩阵")
        return self.depot_indices

    def get_customer_demands(self):
        """获取客户需求列表"""
        if self.customers is None:
            raise ValueError("请先调用 load_data() 方法加载数据")
        return self.customers['DEMAND'].values

    def get_customer_coordinates(self):
        """获取客户坐标"""
        if self.customers is None:
            raise ValueError("请先调用 load_data() 方法加载数据")
        return self.customers[['XCOORD', 'YCOORD']].values

    def get_depot_coordinates(self):
        """获取仓库坐标"""
        if self.depots is None:
            raise ValueError("请先调用 load_data() 方法加载数据")
        return self.depots[['XCOORD', 'YCOORD']].values


# 创建全局实例
data_processor = DataProcessor()

# 便捷函数
def load_vrp_data():
    return data_processor.load_data()

def get_customers():
    return data_processor.get_customers()

def get_depots():
    return data_processor.get_depots()


def get_main_depot():
    return data_processor.get_main_depot()


def compute_distances():
    return data_processor.compute_distance_matrix()


def get_distance_matrix():
    return data_processor.get_distance_matrix()


def get_depot_indices():
    return data_processor.get_depot_indices()

def main():
    load_vrp_data()
    print("1customer:")
    print(data_processor.get_customers())
    print("3customer:")
    print(data_processor.get_task3_customer())

if __name__ == "__main__":
    print("test data process")
    main()