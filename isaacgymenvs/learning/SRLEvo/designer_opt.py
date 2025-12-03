"""
2025 ICRA
用于外肢体形态优化的形态优化器函数（最终版）
支持 real / int / fixed 参数空间
"""

import abc
import numpy as np
import random
import wandb
from skopt import Optimizer
from skopt.space import Real, Integer


# ================================================================
# 基类：支持 param_space（real/int/fixed）
# ================================================================
class MorphologyOptimizer(abc.ABC):
    def __init__(self, base_params, param_space, evaluate_fn):
        """
        base_params: 默认参数
        param_space: {param_name: ("real"/"int"/"fixed", low, high?) }
        evaluate_fn: SRLGym 的 design_evaluate() 或 design_evaluate_with_general_model()
        """
        self.base_params = base_params
        self.param_space = param_space
        self.param_names = list(base_params.keys())

        self.evaluate_design_method = evaluate_fn

        self.best_params = base_params.copy()
        self.best_score = float('-inf')

    # -------------------------------------------------------------
    # 通用采样函数：根据参数空间采样
    # -------------------------------------------------------------
    def sample_param(self, key):
        spec = self.param_space[key]
        ptype = spec[0]

        if ptype == "real":
            _, low, high = spec
            return float(np.random.uniform(low, high))

        elif ptype == "int":
            _, low, high = spec
            return int(np.random.randint(low, high + 1))

        elif ptype == "fixed":
            return self.base_params[key]

        else:
            raise ValueError(f"Unknown param type: {ptype}")

    def clip_param(self, key, val):
        """对 GA mutate 后的值做裁剪，保持在合法范围内"""
        spec = self.param_space[key]
        ptype = spec[0]

        if ptype == "real":
            _, low, high = spec
            return float(np.clip(val, low, high))

        elif ptype == "int":
            _, low, high = spec
            return int(np.clip(int(val), low, high))

        elif ptype == "fixed":
            return self.base_params[key]

        else:
            raise ValueError(f"Unknown param type: {ptype}")

    # -------------------------------------------------------------
    # 统一的 WandB logging
    # -------------------------------------------------------------
    def wandb_log_design(self, prefix, params, score, iteration):
        info_dict = {f"{prefix}/{k}": v for k, v in params.items()}
        info_dict[f"{prefix}/best_reward"] = score
        info_dict["iteration"] = iteration
        wandb.log(info_dict)

    @abc.abstractmethod
    def optimize(self):
        pass


# ================================================================
# Genetic Algorithm (GA)
# ================================================================
class GeneticAlgorithmOptimizer(MorphologyOptimizer):
    def __init__(
        self, base_params, param_space, evaluate_fn,
        population_size=20, mutation_rate=0.3, crossover_rate=0.7,
        num_iterations=10
    ):
        super().__init__(base_params, param_space, evaluate_fn)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_iterations = num_iterations

        self.population = self.init_population()

    # -------------------------------------------------------------
    def init_population(self):
        """初始化种群：完全基于 param_space 采样"""
        population = []
        population.append(self.base_params.copy())

        for _ in range(self.population_size - 1):
            individual = {k: self.sample_param(k) for k in self.param_names}
            population.append(individual)

        return population

    # -------------------------------------------------------------
    def crossover(self, p1, p2):
        """交叉：固定参数保持不变"""
        child = {}
        for k in self.param_names:
            if self.param_space[k][0] == "fixed":
                child[k] = self.base_params[k]
            else:
                child[k] = p1[k] if random.random() < self.crossover_rate else p2[k]
        return child

    # -------------------------------------------------------------
    def mutate(self, individual):
        """变异：real/int 变异，fixed 不变"""
        for k in self.param_names:
            if self.param_space[k][0] == "fixed":
                individual[k] = self.base_params[k]
                continue

            if random.random() < self.mutation_rate:
                if self.param_space[k][0] == "real":
                    # 小范围乘性变异
                    individual[k] *= np.random.uniform(0.9, 1.1)
                elif self.param_space[k][0] == "int":
                    individual[k] += random.choice([-1, 1])

                individual[k] = self.clip_param(k, individual[k])

        return individual

    # -------------------------------------------------------------
    def select(self, population, scores):
        """轮盘赌选择"""
        min_s = min(scores)
        max_s = max(scores)

        if max_s > min_s:
            norm_scores = [(s - min_s) / (max_s - min_s) for s in scores]
        else:
            norm_scores = [1 for _ in scores]

        total = sum(norm_scores)
        pick = random.uniform(0, total)
        curr = 0

        for ind, sc in zip(population, norm_scores):
            curr += sc
            if curr > pick:
                return ind

    # -------------------------------------------------------------
    def optimize(self):
        history = []

        for it in range(self.num_iterations):
            scores = [self.evaluate_design_method(ind) for ind in self.population]

            # 找最优
            best_idx = int(np.argmax(scores))
            best_individual = self.population[best_idx]
            best_score = scores[best_idx]

            # 更新全局最优
            if best_score > self.best_score:
                self.best_score = best_score
                self.best_params = best_individual.copy()

            self.wandb_log_design("GA", self.best_params, self.best_score, it)
            history.append((self.best_params.copy(), self.best_score))

            # 生成下一代
            new_population = [best_individual.copy()]  # 保留精英
            while len(new_population) < self.population_size:
                p1 = self.select(self.population, scores)
                p2 = self.select(self.population, scores)

                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population

        return history


# ================================================================
# Genetic Algorithm v2 —— 无默认设计个体
# ================================================================
class GeneticAlgorithmOptimizer_v2(GeneticAlgorithmOptimizer):
    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {k: self.sample_param(k) for k in self.param_names}
            population.append(individual)
        return population


# ================================================================
# Bayesian Optimization
# ================================================================
class BayesianOptimizer(MorphologyOptimizer):
    def __init__(
        self, base_params, param_space, evaluate_fn,
        num_iterations=100, initial_eval_epoch=700,
        n_initial_points=5, acq_func='EI'
    ):
        super().__init__(base_params, param_space, evaluate_fn)
        self.num_iterations = num_iterations
        self.initial_eval_epoch = initial_eval_epoch
        self.n_initial_points = n_initial_points

        # BO 搜索空间
        skopt_space = []
        for k in self.param_names:
            spec = param_space[k]
            ptype = spec[0]

            if ptype == "real":
                _, low, high = spec
                skopt_space.append(Real(low, high, name=k))

            elif ptype == "int":
                _, low, high = spec
                skopt_space.append(Integer(low, high, name=k))

            elif ptype == "fixed":
                # fixed 不加入搜索空间，由 BO 不优化
                pass

        self.optimizer = Optimizer(skopt_space, acq_func=acq_func, n_initial_points=n_initial_points)
        self.opt_space_keys = [s.name for s in skopt_space]
        self.fixed_keys = [k for k in self.param_names if param_space[k][0] == "fixed"]

        self.history = []

    # -------------------------------------------------------------
    def build_param_dict(self, x_list):
        """将 BO 产生的一维参数数组映射回 dict（补 fixed 参数）"""
        d = {}
        # real/int
        for key, val in zip(self.opt_space_keys, x_list):
            d[key] = float(val)

        # fixed
        for key in self.fixed_keys:
            d[key] = self.base_params[key]

        return d

    # -------------------------------------------------------------
    def optimize(self):
        for it in range(self.num_iterations):

            if it == 0:
                x = [self.base_params[k] for k in self.opt_space_keys]
            else:
                x = self.optimizer.ask()

            params_dict = self.build_param_dict(x)

            kwarg = {"max_epoch": self.initial_eval_epoch} if it < self.n_initial_points else {}
            score = self.evaluate_design_method(params_dict, **kwarg)

            self.optimizer.tell(x, -score)  # skopt 最小化

            if score > self.best_score:
                self.best_score = score
                self.best_params = params_dict.copy()

            self.wandb_log_design("BO", self.best_params, self.best_score, it)
            self.history.append((self.best_params.copy(), self.best_score))

        return self.history


# ================================================================
# Random Search
# ================================================================
class RandomOptimizer(MorphologyOptimizer):
    def __init__(self, base_params, param_space, evaluate_fn, num_iterations=100):
        super().__init__(base_params, param_space, evaluate_fn)
        self.num_iterations = num_iterations
        self.history = []

    def random_sample(self):
        return {k: self.sample_param(k) for k in self.param_names}

    def optimize(self):
        for it in range(self.num_iterations):
            params = self.random_sample()
            score = self.evaluate_design_method(params)

            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()

            self.wandb_log_design("Random", self.best_params, self.best_score, it)
            self.history.append((self.best_params.copy(), self.best_score))

        return self.history
