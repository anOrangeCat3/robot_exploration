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

    def reset(self,)->np.ndarray:
        '''
        重置环境

        返回：
        obs: np.ndarray
            机器人自己的地图
        '''
        # 更新机器人自己的地图
        obs = self.robot.reset(self.map.robot_start_position, self.map.global_map)
        
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
        self.robot.move(action[0], action[1])
        obs = self.robot.update_belief_map(self.map.global_map)
        # TODO: 设计奖励
        # reward = 
        return obs