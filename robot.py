import numpy as np
from typing import Tuple

from parameters import ROBOT_RADIUS
from sensor import Lidar


class Robot:
    '''
    Robot类
    主要用于和地图(environment)交互
    最好和agent分开, 方便训练
    即: agent负责决策(训练), robot负责和环境交互

    包含：
    position: 机器人自身位置
    belief_map: 机器人认知地图, 确认地图上的座标点是否已经探索过 
    lidar: 传感器, sensor类

    功能：(和地图交互)
    移动
    扫描
    更新自身认知地图
    '''
    def __init__(self,
                 ) -> None:
        '''
        初始化机器人

        属性：
        position: np.ndarray
            机器人位置
        lidar: Lidar
            传感器
        belief_map: np.ndarray
            机器人自己的地图
        position_history: np.ndarray
            机器人位置历史
        '''
        self.radius = ROBOT_RADIUS
        self.position = None
        self.belief_map = None
        self.lidar = Lidar()
        self.position_history = np.zeros((0, 2))  # 初始化为空数组，但保持二维结构
        self.explored_area = 0

    def reset(self,
              robot_start_position:np.ndarray,
              global_map:np.ndarray
              )->np.ndarray:
        '''
        每轮游戏开始前重置机器人

        参数:
        robot_start_position: np.ndarray
            机器人初始位置

        global_map: np.ndarray
            本次游戏全局地图

        返回：
        belief_map: np.ndarray
            机器人自己的地图
        '''
        self.position = robot_start_position  # 初始化机器人位置
        self.belief_map=np.ones_like(global_map) * 127  # 初始化机器人自己的地图, 127为未知区域
        self.position_history = robot_start_position  # 初始化为空数组   

        # 更新机器人自己的地图
        self.update_belief_map(global_map)

        return self.belief_map.copy()
    

    def update_belief_map(self,
                         global_map:np.ndarray
                         ):
        '''
        更新机器人自己的地图

        参数:
        global_map: np.ndarray
            全局地图

        返回：
        belief_map: np.ndarray
            机器人自己认知的地图
        '''
        # lidar扫描, 获取新信息, 更新belief_map
        self.belief_map=self.lidar.scan(self.position,self.belief_map.copy(),global_map)
        # 计算已经探索区域
        self.explored_area = self.calculate_explored_area()

        return self.belief_map.copy()
    
    def move_by_angle_distance(self, 
                               angle: float, 
                               distance: float,
                               ) -> np.ndarray:
        """
        根据角度和距离计算新的坐标位置
        限制了移动距离不超过雷达探测范围
        
        参数:
        angle: float
            移动角度（弧度）[0, 2π]
        distance: float
            移动距离
        
        返回:
        new_position: np.ndarray
            新的坐标位置
        """
        # 将角度转换到[0, 2π]范围内
        angle = angle % (2 * np.pi)

        # 限制距离不超过雷达探测范围
        distance = min(distance, self.lidar.lidar_range)
        
        # 计算x和y方向的位移
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)
        
        # 计算新位置
        new_position = np.round(self.position + np.array([dx, dy])).astype(int)
        
        return new_position
    
    def move(self,
             angle:float,
             distance:float
             )->None:
        '''
        移动机器人

        参数:
        angle: float
            移动角度（弧度）[0, 2π)
            0: right
            π/2: down
            π: left
            3π/2: up
        distance: float
            移动距离
        '''
        # 计算新的位置
        new_position = self.move_by_angle_distance(angle, distance)
        # 检查是否在可通行区域, 如果不在, 则找到最近的边界点
        # TODO: 考虑机器人自身的半径, 不要撞到障碍物
        end_position = self.find_nearest_boundary(new_position)
        # 移动到本次移动的终点
        self.position = end_position
        self.position_history = np.vstack((self.position_history, end_position))
    
    def find_nearest_boundary(self,
                             position: np.ndarray,  
                             ) -> np.ndarray:
        """
        找到最近的边界点

        限制了移动距离不超过雷达探测范围
        因此本函数使用belief_map来判断是否在可通行区域
        
        参数:
        position: np.ndarray
            目标位置
        
        返回:
        end_position: np.ndarray
            最近的边界点
        """
        # 获取起点和终点坐标
        x0, y0 = self.position
        x1, y1 = position
        
        # 计算方向向量
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        
        # 初始化误差
        error = dx - dy
        
        # 当前点
        x, y = x0, y0
        
        # 检查连线上的每个点
        while True:
            # 检查当前点是否在地图范围内
            if 0 <= x < self.belief_map.shape[1] and 0 <= y < self.belief_map.shape[0]:
                # 如果遇到障碍物
                if self.belief_map[y, x] == 1:
                    # 返回上一个可通行的点
                    return np.array([x - x_inc, y - y_inc])
            
            # 如果到达终点
            if x == x1 and y == y1:
                return position
            
            # 更新误差和位置
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

    def calculate_explored_area(self,
                               )->np.ndarray:
        '''
        计算已经探索区域

        return:
        explored_area: np.ndarray
            已经探索区域面积
        '''
        return np.sum(self.belief_map == 255)
        
        
