import numpy as np
from skimage.transform import resize
from typing import Tuple

from parameters import EXPLORATION_RATE_THRESHOLD,EXPLORATION_MAX_STEP,ALPHA,BETA
from robot import Robot
from map import Map

class Env:
    '''
    环境类
    用于管理机器人和地图的交互

    包含：
    map: 地图类
    robot: 机器人类
    '''
    def __init__(self,
                 robot:Robot,
                 map:Map,
                 ) -> None:
        '''
        初始化环境

        属性：
        robot: Robot
            机器人
        map: Map
            地图
        '''
        self.robot = robot
        self.map = map
        self.explored_rate = 0
        self.step_count = 0
        

    def reset(self,)->np.ndarray:
        '''
        重置环境

        返回：
        obs: np.ndarray
            机器人自己的地图
        '''
        # 更新机器人自己的地图
        self.robot.explored_area = 0
        self.step_count = 0
        robot_belief_map = self.robot.reset(self.map.robot_start_position, self.map.global_map)
        # 更新探索率
        self.update_explored_rate()
        # 加上机器人自己的位置
        obs= self.mark_robot_position(robot_belief_map)
        # TODO: resize
        obs = resize(obs, (192, 256))

        return obs
    
    def step(self,
             action:Tuple[float, float]
             )->Tuple[np.ndarray, float, bool]:
        '''
        执行一步动作

        参数:
        action: Tuple[float, float]
            动作 为机器人移动的角度和距离
            
        返回:
        obs: np.ndarray
            机器人自己的地图
        reward: float
            奖励
        done: bool
            是否结束
        '''
        # 更新步数
        self.step_count += 1
        # 机器人移动
        self.robot.move(action[0], action[1])
        # 更新机器人自己的belief_map
        robot_belief_map = self.robot.update_belief_map(self.map.global_map)
        # 加上机器人自己的位置
        obs = self.mark_robot_position(robot_belief_map)
        # TODO: resize
        obs = resize(obs, (192, 256))
        # TODO: 设计奖励
        reward,done = self.calculate_reward()
        
        return obs,reward,done
    
    def calculate_terminated_truncated(self,)->bool:
        '''判断是否结束'''
        # 根据探索率判断是否结束
        terminated = self.explored_rate >= EXPLORATION_RATE_THRESHOLD
        # 达到步数上限结束
        truncated = self.step_count >= EXPLORATION_MAX_STEP

        return terminated,truncated
        

    def calculate_reward(self)->float:
        '''计算奖励'''
        # 更新探索率
        explored_rate_change=self.update_explored_rate()
        # 判断是否结束
        terminated,truncated = self.calculate_terminated_truncated()
        done = terminated or truncated
        # 计算奖励
        if terminated:
            # 完成探索，不奖励也不惩罚
            reward = 0
        elif truncated:
            # 步数超过限制, 惩罚50
            # 但是探索率如果较大，则惩罚减少
            reward = -50 + self.explored_rate * BETA
        else:
            # 正常移动，基础惩罚，但探索率提升可减少惩罚
            # 这样即使探索率提升很大，reward 也不会变为正数。
            # 可以避免探索率很高时，reward很高，导致训练不稳定。
            reward = min(0, -1 + explored_rate_change * ALPHA)

        return reward,done


    def mark_robot_position(self, robot_belief_map):
        height, width = robot_belief_map.shape

        # 假设 self.robot.position 是 (x, y)
        pos_x, pos_y = self.robot.position
        # 因为numpy默认的坐标原点和图像的坐标原点不一样，所以需要转换
        center_y = int(pos_y)
        center_x = int(pos_x)
        center_y = np.clip(center_y, 0, height-1)
        center_x = np.clip(center_x, 0, width-1)

        y_coords, x_coords = np.ogrid[:height, :width]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        circle_mask = distances <= self.robot.radius
        robot_belief_map[circle_mask] = 1

        return robot_belief_map
    

    def update_explored_rate(self)->np.ndarray:
        '''
        更新探索率
        '''
        old_explored_rate = self.explored_rate
        self.explored_rate = self.robot.explored_area / self.map.all_passable_area
        explored_rate_change = self.explored_rate - old_explored_rate
        
        return explored_rate_change
