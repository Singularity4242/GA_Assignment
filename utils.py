class EarlyStopper:
    """
    早停判断器
    """

    def __init__(self, patience=50, threshold=0.001, min_generations=50):
        self.patience = patience
        self.threshold = threshold
        self.min_generations = min_generations

        self.no_improvement_count = 0
        self.best_so_far = float('inf')
        self.stagnation_start_gen = 0

    def should_stop(self, gen, current_fitness, fitness_history):
        """
        判断是否应该早停

        Args:
            gen: 当前代数
            current_fitness: 当前最优适应度
            fitness_history: 适应度历史记录

        Returns:
            bool: 是否应该停止
            dict: 停止信息
        """
        # 如果还没达到最小代数，继续运行
        if gen < self.min_generations:
            self._update_best(current_fitness)
            return False, {"reason": "below_min_generations"}

        improvement_ratio = abs(self.best_so_far - current_fitness) / (self.best_so_far + 1e-8)

        if current_fitness < self.best_so_far and improvement_ratio > self.threshold:
            # 有显著改进
            self.best_so_far = current_fitness
            self.no_improvement_count = 0
            self.stagnation_start_gen = gen
            return False, {"reason": "improving", "improvement_ratio": improvement_ratio}
        else:
            # 没有显著改进
            self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                improvement_percent = ((fitness_history[0] - self.best_so_far) / fitness_history[0]) * 100
                stop_info = {
                    "reason": "early_stop",
                    "generation": gen,
                    "patience": self.patience,
                    "best_fitness": self.best_so_far,
                    "improvement_percent": improvement_percent,
                    "stagnation_start": self.stagnation_start_gen,
                    "stagnation_length": gen - self.stagnation_start_gen
                }
                return True, stop_info

        return False, {"reason": "no_improvement", "count": self.no_improvement_count}

    def _update_best(self, current_fitness):
        """更新最优值（不检查早停）"""
        if current_fitness < self.best_so_far:
            self.best_so_far = current_fitness
            self.no_improvement_count = 0

    def get_status(self, gen):
        """获取当前状态"""
        status = "探索中" if gen < self.min_generations else f"无改进:{self.no_improvement_count}"
        return status

    def reset(self):
        """重置早停器状态"""
        self.no_improvement_count = 0
        self.best_so_far = float('inf')
        self.stagnation_start_gen = 0