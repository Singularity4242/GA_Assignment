DATA_PATH = 'VRP.csv'       # CSV数据文件路径
MAX_CAPACITY = 200      #车辆最大容量

# 遗传算法参数
POPULATION_SIZE = 100               # 种群大小
MAX_GENERATIONS = 100               # 最大进化代数
CROSSOVER_PROB = 0.8               # 交叉概率
MUTATION_PROB = 0.15                # 变异概率
EARLY_STOP_PATIENCE = 50            # 早停耐心值（连续多少代无改进则停止）
CONVERGENCE_THRESHOLD = 0.001       # 收敛阈值（改进比例小于此值认为无显著改进）
MIN_GENERATIONS = 50                # 最小进化代数

# 随机需求参数
STOCHASTIC_SAMPLES = 20  # 随机需求采样次数