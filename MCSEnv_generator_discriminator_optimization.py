import gym
from stable_baselines3 import PPO
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 定义强化学习环境
class MCSEnv(gym.Env):
    def __init__(self, num_mus, budget):
        super(MCSEnv, self).__init__()
        self.num_mus = num_mus
        self.budget = budget
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(num_mus,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=10, shape=(2 * num_mus,), dtype=np.float32)
        self.prices = np.random.uniform(1, 3, num_mus)
        self.qualities = np.random.uniform(0.8, 1.0, num_mus)

    def step(self, action):
        self.prices += action
        self.prices = np.clip(self.prices, 0.5, 5.0)
        sensing_times = self.optimize_allocation(self.prices, self.qualities)
        utility = np.sum(self.qualities * np.log1p(sensing_times))
        self.qualities = np.minimum(1.0, self.qualities + 0.01)
        obs = np.concatenate([self.prices, self.qualities])
        done = False
        rewards = self.prices * sensing_times
        return obs, utility, done, {"rewards": rewards}

    def reset(self):
        self.prices = np.random.uniform(1, 3, self.num_mus)
        self.qualities = np.random.uniform(0.8, 1.0, num_mus)
        obs = np.concatenate([self.prices, self.qualities])
        return obs

    def optimize_allocation(self, prices, qualities):
        budget_remaining = self.budget
        sensing_times = np.zeros(self.num_mus)
        sorted_indices = np.argsort(-qualities)
        for i in sorted_indices:
            max_time = budget_remaining / prices[i]
            sensing_times[i] = min(1.0, max_time)
            budget_remaining -= sensing_times[i] * prices[i]
            if budget_remaining <= 0:
                break
        return sensing_times


# 定义生成器（智能体）
class Generator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 超参数
num_mus = 5
budget = 10
state_dim = 2 * num_mus
action_dim = num_mus
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# 初始化环境、生成器和判别器
env = MCSEnv(num_mus, budget)
generator = Generator(state_dim, action_dim)
discriminator = Discriminator(state_dim, action_dim)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    for _ in range(batch_size):
        # 生成器更新
        generator_optimizer.zero_grad()
        action = generator(state)
        discriminator_output = discriminator(state, action)
        generator_loss = criterion(discriminator_output, torch.ones_like(discriminator_output))
        generator_loss.backward()
        generator_optimizer.step()

        # 判别器更新
        discriminator_optimizer.zero_grad()
        real_action = torch.FloatTensor(env.action_space.sample()).unsqueeze(0)
        real_discriminator_output = discriminator(state, real_action)
        real_loss = criterion(real_discriminator_output, torch.ones_like(real_discriminator_output))

        fake_discriminator_output = discriminator(state, action.detach())
        fake_loss = criterion(fake_discriminator_output, torch.zeros_like(fake_discriminator_output))

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

    print(f'Epoch {epoch + 1}, Generator Loss: {generator_loss.item()}, Discriminator Loss: {discriminator_loss.item()}')
