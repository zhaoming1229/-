import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# 定义SP（领导者）
class SensingPlatform:
    def __init__(self, budget, num_mus):
        self.budget = budget
        self.num_mus = num_mus

    def optimize_allocation(self, prices):
        """
        给定MUs的单价数组，求解SP的最优感知时间分配
        """
        # 定义优化变量：感知时间量
        sensing_times = cp.Variable(self.num_mus, nonneg=True)
        # 定义效用函数：对数型
        utility = cp.sum(cp.log1p(sensing_times))
        # 定义约束条件
        constraints = [cp.sum(cp.multiply(prices, sensing_times)) <= self.budget]
        # 定义并求解优化问题
        problem = cp.Problem(cp.Maximize(utility), constraints)
        problem.solve()
        if problem.status == cp.OPTIMAL:
            return sensing_times.value
        else:
            print("优化问题未找到最优解。")
            return None

# 定义MUs（追随者）
class MobileUser:
    def __init__(self, initial_price, id):
        self.price = initial_price
        self.id = id

    def update_price(self, purchased_time, budget):
        """
        更新MU的报价策略
        """
        # 简单策略：根据购买量调整价格
        if purchased_time < budget / 10:
            self.price *= 0.9  # 降低价格
        else:
            self.price *= 1.1  # 提高价格
        return self.price

# 初始化参数
num_mus = 5  # 移动用户数量
budget = 10  # SP的预算
initial_prices = np.random.uniform(1, 3, num_mus)  # MUs初始报价

# 初始化SP和MUs
sp = SensingPlatform(budget=budget, num_mus=num_mus)
mus = [MobileUser(initial_price=initial_prices[i], id=i) for i in range(num_mus)]

# 记录数据
price_history = [initial_prices.copy()]
allocation_history = []

# 多轮博弈
num_rounds = 20
for round in range(num_rounds):
    print(f"轮次 {round + 1}")
    
    # SP优化分配
    prices = np.array([mu.price for mu in mus])
    allocations = sp.optimize_allocation(prices)
    allocation_history.append(allocations)
    
    # 检查优化是否成功
    if allocations is None:
        print("优化失败，停止博弈。")
        break
    
    # MUs调整价格
    for i, mu in enumerate(mus):
        mu.update_price(allocations[i], budget)
    
    # 记录报价变化
    current_prices = np.array([mu.price for mu in mus])
    price_history.append(current_prices)
    
    # 打印当前轮次的价格和分配情况
    print(f"当前价格: {current_prices}")
    print(f"当前分配量: {allocations}\n")

# 将记录转换为numpy数组
price_history = np.array(price_history)
allocation_history = np.array(allocation_history)

# 绘制价格变化
plt.figure(figsize=(12, 6))
for i in range(num_mus):
    plt.plot(price_history[:, i], label=f"MU {i+1}")
plt.title("Price Dynamics of Mobile Users")
plt.xlabel("Rounds")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# 绘制分配量变化
plt.figure(figsize=(12, 6))
for i in range(num_mus):
    plt.plot(allocation_history[:, i], label=f"MU {i+1}")
plt.title("Sensing Time Allocation Dynamics")
plt.xlabel("Rounds")
plt.ylabel("Allocated Sensing Time")
plt.legend()
plt.grid(True)
plt.show()
