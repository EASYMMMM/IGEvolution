import abc
import numpy as np
import random
import csv
import os
import time
import wandb
from skopt import Optimizer
from skopt.space import Real

class MorphologyOptimizer(abc.ABC):
    def __init__(self, base_params):
        self.base_params = base_params
        self.param_names = list(base_params.keys())
        self.num_params = len(self.param_names)
        self.best_params = base_params
        self.best_score = float('-inf')

    @abc.abstractmethod
    def sample_population(self):
        """采样一个参数群体，用于评估"""
        pass

    @abc.abstractmethod
    def update(self, population, scores):
        """基于评估值更新优化参数"""
        pass

    def evaluate(self, params):
        """评估函数，调用外部提供的评估函数"""
        # 假设你有一个名为 evaluate_robot(params) 的外部评估函数
        # return evaluate_robot(params)
        pass

    def optimize(self, num_iterations):
        """运行优化算法"""
        for i in range(num_iterations):
            population = self.sample_population()
            scores = [self.evaluate(individual) for individual in population]
            self.update(population, scores)
            print(f"Iteration {i+1}/{num_iterations}, Best Score: {self.best_score}")
        
        return self.best_params
    
class GeneticAlgorithmOptimizer(MorphologyOptimizer):
    def __init__(self,
                  base_design_params, 
                  evaluate_design_method,
                  population_size=20, 
                  mutation_rate=0.1,
                  crossover_rate=0.7,
                  num_iterations=10,
                  bounds_scale=0.3):
        super().__init__(base_design_params)
        self.population_size = population_size
        self.evaluate_design_method = evaluate_design_method
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_iterations = num_iterations
        self.bounds_scale = bounds_scale
        self.population = self.init_population()
        
         
    def init_population(self):
        """初始化种群，参数在基础参数的上下bounds_scale范围内"""
        population = []
        population.append(self.base_params.copy())
        
        for _ in range(self.population_size - 1):
            individual = {}
            for key, val in self.base_params.items():
                lower_bound = (1 - self.bounds_scale) * val
                upper_bound = (1 + self.bounds_scale) * val
                individual[key] = np.random.uniform(lower_bound, upper_bound)
            population.append(individual)
            
        return population

    def sample_population(self):
        # GA dont need this method
        population = []
        for _ in range(self.population_size):
            individual = {key: np.random.uniform(0.5 * val, 1.5 * val) for key, val in self.base_params.items()}
            population.append(individual)
        return population
    
    def mutate(self, individual):
        """个体变异，参数裁剪到基础参数的上下bounds_scale范围内"""
        for key in individual.keys():
            if random.random() < self.mutation_rate:
                individual[key] *= np.random.uniform(0.9, 1.1)
                # 裁剪参数到指定范围内
                lower_bound = (1 - self.bounds_scale) * self.base_params[key]
                upper_bound = (1 + self.bounds_scale) * self.base_params[key]
                individual[key] = np.clip(individual[key], lower_bound, upper_bound)
        return individual

    def crossover(self, parent1, parent2):
        """交叉操作生成新个体"""
        child = {}
        for key in parent1.keys():
            if random.random() < self.crossover_rate:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def update(self, population, scores):
        """基于评估值更新优化参数，保留表现最好的两个个体"""
        # 获取表现最好的两个个体
        sorted_indices = np.argsort(scores)[::-1]  # 从大到小排序
        best_individuals = [population[sorted_indices[0]], population[sorted_indices[1]]]
        best_scores = [scores[sorted_indices[0]], scores[sorted_indices[1]]]

        # 如果有更好的个体，更新最佳参数
        if best_scores[0] > self.best_score:
            self.best_score = best_scores[0]
            self.best_params = best_individuals[0]

        # 新种群包含两个精英个体
        new_population = best_individuals.copy()

        # 通过交叉和变异生成其余的个体
        while len(new_population) < self.population_size:
            parent1 = self.select(population, scores)
            parent2 = self.select(population, scores)
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(child2))

        self.population = new_population

    def select(self, population, scores):
        """标准化选择"""
        min_score = min(scores)
        max_score = max(scores)
        
        # 如果所有分数相同，则平等分配选择概率
        if max_score > min_score:
            scores = [(score - min_score) / (max_score - min_score) for score in scores]
        else:
            scores = [1 for _ in scores]  # 所有分数相同的情况，给予等同的概率
        
        total_fitness = sum(scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual, score in zip(population, scores):
            current += score
            if current > pick:
                return individual

    def optimize(self,):
        """运行优化算法并将每一代的最优设计及其评估值保存在CSV文件中"""
        best_individuals_over_time = []  # 用于保存每一代的最优设计及其评估值
      
        for i in range(self.num_iterations):
            # 评估当前种群
            scores = []
            for individual in self.population:
                score = self.evaluate_design_method(individual)
                scores.append(score)
        
            # 基于当前种群和得分更新种群
            self.update(self.population, scores)
            print(f"Iteration {i+1}/{self.num_iterations}, Best Score: {self.best_score}")
            # 保存当前代的最优设计及其评估值
            self.log_design(self.best_params.copy(),self.best_score,i)
            best_individuals_over_time.append((self.best_params.copy(), self.best_score))
                
        
        return best_individuals_over_time
    
    def log_design(self,best_params,best_reward,iteration):
        info_dict = {
                "GA/leg1_lenth" : best_params["first_leg_lenth"],
                "GA/leg1_size"  : best_params["first_leg_size"],
                "GA/leg2_lenth": best_params["second_leg_lenth"],
                "GA/leg2_size" : best_params["second_leg_size"],
                "GA/end_size"  : best_params["third_leg_size"],
                "GA/best_reward" :  best_reward,
                "GA_iteration": iteration

        }
        wandb.log(info_dict )


class BayesianOptimizer(MorphologyOptimizer):
    def __init__(self, 
                 base_design_params, 
                 evaluate_design_method,
                 num_iterations=100, 
                 n_initial_points=5, 
                 bounds_scale = 0.3,
                 acq_func='EI'):
        super().__init__(base_design_params)
        self.num_iterations = num_iterations
        self.evaluate_design_method = evaluate_design_method
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func
        self.param_names = list(base_design_params.keys())
        self.default_design_evalutate = False
        # 定义贝叶斯优化的搜索空间
        self.search_space = [Real((1 - bounds_scale) * val, (1 + bounds_scale) * val, name=key) for key, val in base_design_params.items()]
        self.optimizer = Optimizer(self.search_space, n_initial_points=n_initial_points, acq_func=acq_func)

        self.best_individuals_over_time = []  # 用于保存每一代的最优设计及其评估值

    def sample_population(self):
        pass

    def update(self, ):
        pass

    def optimize(self):

        """运行优化算法并将每一代的最优设计及其评估值保存在CSV文件中"""
        if self.num_iterations < self.n_initial_points:
            raise ValueError(f"num_iterations ({self.num_iterations}) must be greater than or equal to n_initial_points ({self.n_initial_points}).")
        for i in range(self.num_iterations):
            # 从贝叶斯优化器中获取下一组参数
            if not self.default_design_evalutate:
                params_dict = self.base_params
                next_params = [self.base_params[key] for key in self.param_names]  # 将 base_params 转换为 next_params 格式
                self.default_design_evalutate = True
            else:
                next_params = self.optimizer.ask()
                params_dict = dict(zip(self.param_names, next_params))
            kwargs = {}
            if i < self.n_initial_points:
                kwargs['max_epoch']=700
            # 评估当前参数
            score = self.evaluate_design_method(params_dict, **kwargs)

            # 将评估结果告诉贝叶斯优化器
            self.optimizer.tell(next_params, -score)  # 贝叶斯优化器最小化目标函数，因此使用负分数

            # 更新最佳参数和分数
            if score > self.best_score:
                self.best_score = score
                self.best_params = params_dict

            print(f"Iteration {i+1}/{self.num_iterations}, Best Score: {self.best_score}")

            # 保存当前代的最优设计及其评估值
            self.log_design(self.best_params.copy(), self.best_score, i)
            self.best_individuals_over_time.append((self.best_params.copy(), self.best_score))

        return self.best_individuals_over_time

    def log_design(self, best_params, best_reward, iteration):
        info_dict = {
            "BO/leg1_lenth": best_params["first_leg_lenth"],
            "BO/leg1_size": best_params["first_leg_size"],
            "BO/leg2_lenth": best_params["second_leg_lenth"],
            "BO/leg2_size": best_params["second_leg_size"],
            "BO/end_size": best_params["third_leg_size"],
            "BO/best_reward": best_reward,
            "BO_iteration": iteration
        }
        wandb.log(info_dict)


class GeneticAlgorithmOptimizer_v2(MorphologyOptimizer):
    def __init__(self,
                  base_design_params, 
                  evaluate_design_method,
                  population_size=20, 
                  mutation_rate=0.1,
                  crossover_rate=0.7,
                  num_iterations=10,
                  bounds_scale=0.3):
        super().__init__(base_design_params)
        self.population_size = population_size
        self.evaluate_design_method = evaluate_design_method
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_iterations = num_iterations
        self.bounds_scale = bounds_scale
        self.population = self.init_population()
        
         
    def init_population(self):
        """初始化种群，参数在基础参数的上下bounds_scale范围内"""
        population = []
        # population.append(self.base_params.copy()) # v2: no default design
        
        for _ in range(self.population_size - 1):
            individual = {}
            for key, val in self.base_params.items():
                lower_bound = (1 - self.bounds_scale) * val
                upper_bound = (1 + self.bounds_scale) * val
                individual[key] = np.random.uniform(lower_bound, upper_bound)
            population.append(individual)
            
        return population

    def sample_population(self):
        # GA dont need this method
        population = []
        for _ in range(self.population_size):
            individual = {key: np.random.uniform(0.5 * val, 1.5 * val) for key, val in self.base_params.items()}
            population.append(individual)
        return population
    
    def mutate(self, individual):
        """个体变异，参数裁剪到基础参数的上下bounds_scale范围内"""
        for key in individual.keys():
            if random.random() < self.mutation_rate:
                individual[key] *= np.random.uniform(0.9, 1.1)
                # 裁剪参数到指定范围内
                lower_bound = (1 - self.bounds_scale) * self.base_params[key]
                upper_bound = (1 + self.bounds_scale) * self.base_params[key]
                individual[key] = np.clip(individual[key], lower_bound, upper_bound)
        return individual

    def crossover(self, parent1, parent2):
        """交叉操作生成新个体"""
        child = {}
        for key in parent1.keys():
            if random.random() < self.crossover_rate:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
            lower_bound = (1 - self.bounds_scale) * self.base_params[key]
            upper_bound = (1 + self.bounds_scale) * self.base_params[key]
            child[key] = np.clip(child[key], lower_bound, upper_bound)
        return child

    def update(self, population, scores):
        """基于评估值更新优化参数，保留表现最好的两个个体"""
        # 获取表现最好的两个个体
        sorted_indices = np.argsort(scores)[::-1]  # 从大到小排序
        best_individuals = [population[sorted_indices[0]], population[sorted_indices[1]]]
        best_scores = [scores[sorted_indices[0]], scores[sorted_indices[1]]]

        # 如果有更好的个体，更新最佳参数
        if best_scores[0] > self.best_score:
            self.best_score = best_scores[0]
            self.best_params = best_individuals[0]

        # 新种群包含两个精英个体
        new_population = best_individuals.copy()

        # 通过交叉和变异生成其余的个体
        while len(new_population) < self.population_size:
            parent1 = self.select(population, scores)
            parent2 = self.select(population, scores)
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(child2))

        self.population = new_population

    def select(self, population, scores):
        """标准化选择"""
        min_score = min(scores)
        max_score = max(scores)
        
        # 如果所有分数相同，则平等分配选择概率
        if max_score > min_score:
            scores = [(score - min_score) / (max_score - min_score) for score in scores]
        else:
            scores = [1 for _ in scores]  # 所有分数相同的情况，给予等同的概率
        
        total_fitness = sum(scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual, score in zip(population, scores):
            current += score
            if current > pick:
                return individual

    def optimize(self,):
        """运行优化算法并将每一代的最优设计及其评估值保存在CSV文件中"""
        best_individuals_over_time = []  # 用于保存每一代的最优设计及其评估值
      
        for i in range(self.num_iterations):
            # 评估当前种群
            scores = []
            for individual in self.population:
                kwarg = {'max_epoch': 1000} if i <= 1 else {}
                score = self.evaluate_design_method(individual,**kwarg)
                scores.append(score)
        
            # 基于当前种群和得分更新种群
            self.update(self.population, scores)
            print(f"Iteration {i+1}/{self.num_iterations}, Best Score: {self.best_score}")
            # 保存当前代的最优设计及其评估值
            self.log_design(self.best_params.copy(),self.best_score,i)
            best_individuals_over_time.append((self.best_params.copy(), self.best_score))
                
        
        return best_individuals_over_time
    
    def log_design(self,best_params,best_reward,iteration):
        info_dict = {
                "GA/leg1_lenth" : best_params["first_leg_lenth"],
                "GA/leg1_size"  : best_params["first_leg_size"],
                "GA/leg2_lenth": best_params["second_leg_lenth"],
                "GA/leg2_size" : best_params["second_leg_size"],
                "GA/end_size"  : best_params["third_leg_size"],
                "GA/best_reward" :  best_reward,
                "GA_iteration": iteration

        }
        wandb.log(info_dict )


