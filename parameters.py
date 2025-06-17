import numpy as np

# LIDAR参数
LIDAR_RANGE = 50
LIDAR_ANGLE_STEP = 0.5 / 180 * np.pi  # 0.5度的角度增量

# 机器人参数
ROBOT_RADIUS = 10

# 探索参数
EXPLORATION_RATE_THRESHOLD = 0.95  # 探索率阈值
EXPLORATION_MAX_STEP = 200  # 最大步数

# 奖励(惩罚)参数
ALPHA = 4  # 探索惩罚系数
BETA = 20  # 探索率惩罚系数

# 神经网络参数
HIDDEN_DIM = 32

# PPO参数
LEARNING_RATE = 1e-4
GAMMA = 0.9  # 折扣因子
BATCH_SIZE = 32
TRAIN_EPOCHS = 10  # 每一局游戏训练次数
ADVANTAGE_LAMBDA = 0.95
CLIP_EPSILON = 0.2  # GAE参数

# 训练参数
TRAIN_EPISODE_NUM = 1000  # 训练轮数
EVAL_INTERVAL = 10  # 评估间隔