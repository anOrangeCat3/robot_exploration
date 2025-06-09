import numpy as np
from skimage import io
from typing import Tuple

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
        

    def reset(self,)->np.ndarray:
        '''
        重置环境

        返回：
        obs: np.ndarray
            机器人自己的地图
        '''
        # 更新机器人自己的地图
        robot_belief_map = self.robot.reset(self.map.robot_start_position, self.map.global_map)
        # 加上机器人自己的位置
        obs= self.mark_robot_position(robot_belief_map)

        return obs
    
    def step(self,
             action:Tuple[float, float]
             )->Tuple[np.ndarray, float, bool, dict]:
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
        info: dict
            信息
        '''
        # 机器人移动
        self.robot.move(action[0], action[1])
        # 更新机器人自己的belief_map
        robot_belief_map = self.robot.update_belief_map(self.map.global_map)
        # 加上机器人自己的位置
        obs = self.mark_robot_position(robot_belief_map)
        # TODO: 设计奖励
        # reward = 
        # TODO: 设计是否结束
        # done = 
        return obs

    def mark_robot_position(self, robot_belief_map):
        height, width = robot_belief_map.shape

        # 假设 self.robot.position 是 (x, y)
        pos_x, pos_y = self.robot.position
        center_y = int(pos_y)
        center_x = int(pos_x)
        center_y = np.clip(center_y, 0, height-1)
        center_x = np.clip(center_x, 0, width-1)

        y_coords, x_coords = np.ogrid[:height, :width]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        circle_mask = distances <= self.robot.radius
        robot_belief_map[circle_mask] = 1
        return robot_belief_map