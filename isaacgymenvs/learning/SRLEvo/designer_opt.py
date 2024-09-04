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
                 acq_func='EI'):
        super().__init__(base_design_params)
        self.num_iterations = num_iterations
        self.evaluate_design_method = evaluate_design_method
        self.n_initial_points = n_initial_points
        self.acq_func = acq_func
        self.param_names = list(base_design_params.keys())
        # 定义贝叶斯优化的搜索空间
        self.search_space = [Real((1 - 0.3) * val, (1 + 0.3) * val, name=key) for key, val in base_design_params.items()]
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
            next_params = self.optimizer.ask()
            params_dict = dict(zip(self.param_names, next_params))
            kwargs = {}
            if i < self.n_initial_points:
                kwargs['max_epoch']=20
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


class GA_Design_Optim():
    '''
    外层设计优化 Genetic Algorithm。  
    在遗传算法中，个体表示agent各个部位的优化系数。  
    例如：优化系数为1.2，表示agent该部位尺寸为标准尺寸的1.2倍。  
    将所有的优化系数螺旋编码为二进制数，作为一个个体。  
    待优化参数：（4-26） 
    {  'thigh_lenth':0.34,            # 大腿长 0.34
       'shin_lenth':0.3,              # 小腿长 0.3
       'upper_arm_lenth':0.2771,        # 大臂长 0.2771
       'lower_arm_lenth':0.2944,        # 小臂长 0.2944
       'foot_lenth':0.18,             # 脚长   0.18
    }
    params:
    decode_size: 单个数据的二进制编码长度
    POP_size: 种群大小
    crossover_rate：交叉概率
    mutation_rate: 突变概率
    n_generations: 迭代次数
    optim_bound：个体的数值边界。
    '''
    def __init__(self,
                 model,                  # 强化学习模型
                 decode_size = 24,
                 POP_size = 80 ,
                 crossover_rate = 0.6,
                 mutation_rate  = 0.01,
                 n_generations  = 10,
                 n_envs         = 8,
                 optim_bound   = [0.7, 1.3],
                 overchange_punish = 0,  # 更新幅度过大的惩罚项
                 elite_num    = 3,  # 精英策略
                 terrain_type = 'steps'
                    ):
        self.decode_size    = decode_size                      # 单个数值的编码长度
        self.POP_size       = POP_size                         # 种群大小
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate
        self.n_generations  = n_generations                    # 迭代次数
        self.optim_bound    = optim_bound
        self.overchange_punish = overchange_punish
        self.elite_num      = elite_num
        self.__origin_design_params  = {   
                                            "first_leg_lenth" : 0.40,
                                            "first_leg_size"  : 0.03,
                                            "second_leg_lenth": 0.80,
                                            "second_leg_size" : 0.03,
                                            "third_leg_size"  : 0.03, 
                                             }     
        self.DNA_size       = decode_size * len(self.__origin_design_params)
        self.n_envs         = n_envs
        self.model = model
        self.last_best_design = np.zeros(self.DNA_size,dtype=int)
        self.last_best_design[0:len(self.__origin_design_params)] = 1
        self.terrain_type = terrain_type
        self.init_controller()


    def init_controller(self,algo="sac"):
        # 初始化控制器
        RL_algo = {
            "sac": SAC,
            "td3": TD3,
            "ppo": PPO,
        }[algo]
        hyperparams = {
            "sac": dict(
                batch_size=256,
                gamma=0.98,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_starts=10000,
                buffer_size=int(5e2),
                tau=0.01,
                gradient_steps=4,
            ),
            "ppo": dict(
                batch_size=512,
                learning_rate=2.5e-4,
                policy_kwargs=dict(net_arch=({'pi':[128,128]},{'vf':[128,128]})),
                gamma=0.99
            )
        }[algo]
        env_kwargs = {'terrain_type':self.terrain_type}
        self.envs = make_vec_env(env_id = 'HumanoidCustomEnv-v0', n_envs = self.n_envs, env_kwargs = env_kwargs)
        self.last_best_params = self.__origin_design_params
        #env = gym.make('HumanoidCustomEnv-v0', terrain_type='steps')
        #self.env = env
        #print('load from:')
        #save_path = 'best_model/5e6_steps_t5_cpu8_sac_HumanoidCustomEnv-v0.zip'
        #print(save_path)   
        #self.model = SAC("MlpPolicy", env, verbose=1,  **hyperparams)
        #self.model.set_parameters(save_path)

    def Fitness_single(self, pop ):
        thigh_lenth, shin_lenth, upper_arm_lenth, lower_arm_lenth, foot_lenth = self.translateDNA(pop)
        # 评估种群中全部个体适应度
        fitness = np.zeros(self.POP_size)
        for i in range(self.POP_size):
            # 更新XML文件
            new_params = self.new_design_params(thigh_lenth[i],shin_lenth[i],upper_arm_lenth[i],lower_arm_lenth[i],foot_lenth[i])
            self.env.update_xml_model(new_params)
            # 评估每一个个体
            episode_rewards, episode_lengths = [], []
            for _ in range(8):
                obs = self.env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                episode_rewards.append(episode_reward)
            episode_rewards.remove(max(episode_rewards)) # 随机性较大，删掉一个最大值一个最小值
            episode_rewards.remove(min(episode_rewards))
            mean_reward = np.mean(episode_rewards)
            fitness[i]  = mean_reward
            print('num:',i)

        return fitness

    def Fitness(self, pop):
        # 使用向量环境评估适应度
        thigh_lenth, shin_lenth, upper_arm_lenth, lower_arm_lenth, foot_lenth = self.translateDNA(pop)
        # 评估种群中全部个体适应度
        fitness = np.zeros(self.POP_size)
        for i in range(self.POP_size):
            new_params = self.new_design_params(thigh_lenth[i],shin_lenth[i],upper_arm_lenth[i],lower_arm_lenth[i],foot_lenth[i])
            self.envs.update_xml_model(new_params)
            obs = self.envs.reset()
            episode = 0
            episode_rewards = []
            episode_reward = np.zeros(self.n_envs)
            while episode < self.n_envs:
                action, _  = self.model.predict(obs,)
                obs, rewards, dones, infos = self.envs.step(action)
                episode_reward += rewards
                for idx,done in enumerate(dones):
                    if done:
                        print(idx)
                        episode += 1
                        episode_rewards.append(episode_reward[idx])
                        episode_reward[idx] = 0
            mean_reward = np.mean(episode_rewards)
            v_ave_list = list()
            for info in infos: # 获取episode平均速度
                v_ave_list.append(info['ave_velocity']) 
            v_ave = np.mean(v_ave_list)
            if self.terrain_type == 'default':
                min_reward = 550
            if self.terrain_type == 'steps':
                min_reward = 550
            if mean_reward < min_reward:
                v_ave = 0
            f = v_ave
            if self.out_of_range(new_params, clip_range = 0.1):
                # 如果参数更新幅度过大，惩罚20fitness
                f -= self.overchange_punish
            fitness[i]  = f
            print('num:',i)
        self.best_reward.append(np.max(fitness))
        self.fitness_data.append(fitness)
        return fitness

    def new_design_params(self,p_thigh_lenth, p_shin_lenth, p_upper_arm_lenth, p_lower_arm_lenth, p_foot_lenth):
        # 由个体DNA信息得到新的设计参数
        params = {  'thigh_lenth':self.__origin_design_params['thigh_lenth']*p_thigh_lenth,
                    'shin_lenth':self.__origin_design_params['shin_lenth']*p_shin_lenth,             
                    'upper_arm_lenth':self.__origin_design_params['upper_arm_lenth']*p_upper_arm_lenth,     
                    'lower_arm_lenth':self.__origin_design_params['lower_arm_lenth']*p_lower_arm_lenth,        
                    'foot_lenth':self.__origin_design_params['foot_lenth']*p_foot_lenth       }    
        return params

    def get_fitness(self, pop):
        pred = self.Fitness(pop)
        return pred - np.min(pred)+1e-3  # 求最大值时的适应度
        # return np.max(pred) - pred + 1e-3  # 求最小值时的适应度，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]


    def translateDNA(self, pop): 
        # DNA解码：二进制编码转换为设计参数的优化系数
        # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
        design_params_size = len(self.__origin_design_params)
        thigh_lenth_pop     = pop[:,0::design_params_size]  #奇数列表示X
        shin_lenth_pop      = pop[:,1::design_params_size]  #偶数列表示y
        upper_arm_lenth_pop = pop[:,2::design_params_size]
        lower_arm_lenth_pop = pop[:,3::design_params_size]
        foot_lenth_pop      = pop[:,4::design_params_size]
        #pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
        thigh_lenth     = self.decode(thigh_lenth_pop)
        shin_lenth      = self.decode(shin_lenth_pop)
        upper_arm_lenth = self.decode(upper_arm_lenth_pop)
        lower_arm_lenth = self.decode(lower_arm_lenth_pop)
        foot_lenth      = self.decode(foot_lenth_pop)
        return thigh_lenth, shin_lenth, upper_arm_lenth, lower_arm_lenth, foot_lenth

    def translateSingleDNA(self, pop): 
        # 上面的函数不能处理单行数据 TAT 懒得优化代码了，干脆再写一个新的好了
        # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
        design_params_size = len(self.__origin_design_params)
        thigh_lenth_pop     = pop[0::design_params_size]  #奇数列表示X
        shin_lenth_pop      = pop[1::design_params_size]  #偶数列表示y
        upper_arm_lenth_pop = pop[2::design_params_size]
        lower_arm_lenth_pop = pop[3::design_params_size]
        foot_lenth_pop      = pop[4::design_params_size]
        #pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
        thigh_lenth     = self.decode(thigh_lenth_pop)
        shin_lenth      = self.decode(shin_lenth_pop)
        upper_arm_lenth = self.decode(upper_arm_lenth_pop)
        lower_arm_lenth = self.decode(lower_arm_lenth_pop)
        foot_lenth      = self.decode(foot_lenth_pop)
        return thigh_lenth, shin_lenth, upper_arm_lenth, lower_arm_lenth, foot_lenth

    def decode(self, code):
        # 二进制转十进制解码
        # 输出结果在self.optim_bound之间
        dec = code.dot(2**np.arange(self.decode_size)[::-1])/float(2**self.decode_size-1)*(self.optim_bound[1]-self.optim_bound[0])+self.optim_bound[0]
        return dec

    def crossover_and_mutation(self, pop):
        new_pop = []
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if np.random.rand() < self.crossover_rate:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pop[np.random.randint(self.POP_size)]  # 再种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=0, high=self.DNA_size * 2)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            self.mutation(child)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop


    def mutation(self, child):
        if np.random.rand() < self.mutation_rate:  # 以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, self.DNA_size)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


    def select(self, pop, fitness:np.ndarray):  # nature selection wrt pop's fitness
        # 精英策略
        idx = np.random.choice(np.arange(self.POP_size), size=self.POP_size - self.elite_num, replace=True,
                            p=(fitness) / (fitness.sum()))
        elite_idx = fitness.argsort()[-self.elite_num:]  
        new_idx = np.append(idx,elite_idx)                  
        return pop[new_idx]

    def out_of_range(self, new_p:dict, clip_range = 0.1):
        # 更新限幅
        result = False
        for key in new_p.keys():
            r = (new_p[key] - self.last_best_params[key]) / self.last_best_params[key]
            if r > clip_range or r < -clip_range:
                result = True
        return result


    def evolve(self):
        # 进化N代

        self.pop_data = list() # 存储全部数据
        self.fitness_data = list()

        #begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        pop = np.random.randint(2, size=(self.POP_size, self.DNA_size)) # 随机初始化种群
        pop[0,:] = self.last_best_design   # 向种群中添加一个曾经的最优设计
        self.best_individual = list()
        self.best_reward     = list()
        for i in range(self.n_generations):#迭代N代
            pop = np.array(self.crossover_and_mutation(pop))
            fitness = self.get_fitness(pop)
            # 保存最优个体
            self.last_best_design = pop[np.argmax(fitness)]
            self.best_individual.append(self.last_best_design)


            pop = self.select(pop, fitness) #选择生成新的种群

            self.pop_data.append(pop)
            

            #self.best_individual.append(pop[np.argmax(fitness)])


        thigh_lenth, shin_lenth, upper_arm_lenth, lower_arm_lenth, foot_lenth = self.translateSingleDNA(self.last_best_design)
        new_design_params = self.new_design_params(thigh_lenth, shin_lenth, upper_arm_lenth, lower_arm_lenth, foot_lenth)
        self.last_best_params = new_design_params
        return new_design_params
            

    def save_fig(self, fig_name:str = None):
        t,s,u,l,f = self.translateDNA(np.array(self.best_individual))
        plt.figure(1,figsize=(8, 8))
        plt.subplot(2,1,1)
        plt.plot(t,color='r')
        plt.plot(s,color='g')
        plt.plot(u,color='b')
        plt.plot(l,color='c')
        plt.plot(f,color='m')
        plt.legend(['t','s','u','l','f'])
        plt.title('design parameters')
        plt.subplot(2,1,2)
        plt.plot(self.best_reward,color='b')
        plt.title('reward')
        if fig_name == None:
            fig_name = 'GA_optim_result'
        plt.savefig('screenshot/'+fig_name+'.png', bbox_inches='tight')
        plt.show()


