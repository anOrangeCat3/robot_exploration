import numpy as np

# LIDAR参数
LIDAR_RANGE = 50
LIDAR_ANGLE_STEP = 0.5 / 180 * np.pi  # 0.5度的角度增量

# 机器人参数
ROBOT_RADIUS = 10

# 探索参数
EXPLORATION_RATE_THRESHOLD = 0.95  # 探索率阈值
EXPLORATION_MAX_STEP = 1000  # 最大步数

# 奖励(惩罚)参数
ALPHA = 4  # 探索惩罚系数

