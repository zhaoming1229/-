!pip install gym
!pip install stable_baselines3
%pip install 'shimmy>=2.0'
%pip install gymnasium

import gym
from stable_baselines3 import PPO
from gym import spaces
import numpy as np

# 定义环境
class MCSEnv(gym.Env):
    def __init__(self, num_mus, budget):
        super(MCSEnv, self).__init__()
        self.num_mus = num_mus
        self.budget = budget
        # 动作空间：每个MU的价格调整
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(num_mus,), dtype=np.float32)
        # 观察空间：当前价格和质量评分
        self.observation_space = spaces.Box(low=0, high=10, shape=(2*num_mus,), dtype=np.float32)
        # 初始化价格和质量
        self.prices = np.random.uniform(1, 3, num_mus)
        self.qualities = np.random.uniform(0.8, 1.0, num_mus)
    
    def step(self, action):
        # 更新价格
        self.prices += action
        self.prices = np.clip(self.prices, 0.5, 5.0)  # 保持价格在合理范围
        
        # SP优化分配
        sensing_times = self.optimize_allocation(self.prices, self.qualities)
        
        # 计算奖励：加权效用
        utility = np.sum(self.qualities * np.log1p(sensing_times))
        
        # 更新质量评分（假设质量随时间稳定或提高）
        self.qualities = np.minimum(1.0, self.qualities + 0.01)
        
        # 观察状态
        obs = np.concatenate([self.prices, self.qualities])
        
        # 判断是否结束（无限环境，设定轮次上限）
        done = False
        
        # 计算每个MU的收益（作为个体奖励）
        rewards = self.prices * sensing_times  # 简化的收益函数
        
        return obs, utility, done, {"rewards": rewards}
    
    def reset(self):
        self.prices = np.random.uniform(1, 3, self.num_mus)
        self.qualities = np.random.uniform(0.8, 1.0, self.num_mus)
        obs = np.concatenate([self.prices, self.qualities])
        return obs
    
    def optimize_allocation(self, prices, qualities):
        """
        简化的优化分配方法，考虑数据质量
        """
        # 使用简单的分配策略：优先分配给高质量MU
        budget_remaining = self.budget
        sensing_times = np.zeros(self.num_mus)
        sorted_indices = np.argsort(-qualities)  # 按质量降序排序
        for i in sorted_indices:
            max_time = budget_remaining / prices[i]
            sensing_times[i] = min(1.0, max_time)  # 每个MU最多分配1.0时间单位
            budget_remaining -= sensing_times[i] * prices[i]
            if budget_remaining <= 0:
                break
        return sensing_times

# 初始化环境
num_mus = 5
budget = 10
env = MCSEnv(num_mus=num_mus, budget=budget)

# 初始化并训练模型
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 测试模型
obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    print(f"行动: {action}, 奖励: {rewards}")
    if done:
        break
