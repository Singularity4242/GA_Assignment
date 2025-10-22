import pandas as pd
import numpy as np
from config import DATA_PATH


class DataProcessor:

    def __init__(self):
        self.data = None
        self.custs = None
        self.depots = None
        self.main_depot = None
        self.dist_matrix = None
        self.depot_indices = None

    def load_data(self):
        print("——————加载VRP数据——————")
        self.data = pd.read_csv(DATA_PATH)

        # 分离客户和仓库
        self.custs = self.data[self.data['CUST_OR_DEPOT'] == 'CUSTOMER'].copy()
        self.depots = self.data[self.data['CUST_OR_DEPOT'] == 'DEPOT'].copy()
        self.main_depot = self.data[
            (self.data['CUST_OR_DEPOT'] == 'DEPOT') & (self.data['NO'] == 0)
            ].iloc[0]

        # print(f"找到 {len(self.customers)} 个客户")
        # print(f"找到 {len(self.depots)} 个仓库")
        # print("仓库信息:")
        # for i, depot in self.depots.iterrows():
        #     print(f"  仓库{depot['NO']}: ({depot['XCOORD']}, {depot['YCOORD']})")
        return self.custs, self.depots, self.main_depot


    def cul_dist_matrix(self):
        print("——————计算距离矩阵——————")
        n_custs = len(self.custs)
        n_depots = len(self.depots)
        self.dist_matrix = np.zeros((n_custs + n_depots, n_custs + n_depots))
        points = []
        # 客户坐标
        for i, customer in self.custs.iterrows():
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
                self.dist_matrix[i][j] = distance

        # 仓库索引映射
        self.depot_indices = {
            int(depot_no): n_custs + i
            for i, depot_no in enumerate(depots_sorted['NO'])
        }

        print(f"仓库索引映射: {self.depot_indices}")
        return self.dist_matrix
            #, self.depot_indices

    #用于task3复制
    def get_task3_customer(self):
        addition_customer = self.custs.copy()
        addition_customer['YCOORD'] += 150
        addition_customer['NO'] += 100
        # 合并
        self.custs = pd.concat([self.custs, addition_customer], ignore_index=True)
        return self.custs

    def get_dist_matrix(self):
        return self.dist_matrix

    def get_customers(self):
        return self.custs

    def get_depots(self):
        return self.depots

    def get_main_depot(self):
        return self.main_depot

    def get_depot_indices(self):
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


data_processor = DataProcessor()
def get_distance_matrix():
    return data_processor.get_distance_matrix()

def main():
    data_processor.load_data()
    # print("1customer:")
    # print(data_processor.get_customers())
    # print("3customer:")
    # print(data_processor.get_task3_customer())
    data_processor.cul_dist_matrix()
    print(data_processor.get_depot_indices())

if __name__ == "__main__":
    print("test data process")
    main()