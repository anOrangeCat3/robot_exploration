import copy

from robot import Robot
from map import Map
from env import Env
from agent import PPO_Agent
from parameters import TRAIN_EPISODE_NUM, EVAL_INTERVAL

class TrainManager():
    def __init__(self,
                 env:Env,
                 agent:PPO_Agent,
                 episode_num:int=TRAIN_EPISODE_NUM,
                 eval_iters:int=EVAL_INTERVAL,
                 ) -> None:
        '''
        初始化训练管理类

        属性：
        train_env: Env
            训练环境
            
        episode_num: int
            训练轮数
        eval_iters: int
            评估间隔
        '''
        self.train_env = env
        self.eval_env = copy.deepcopy(env)
        self.agent = agent
        self.episode_num = episode_num
        self.eval_iters = eval_iters

    def train(self,)->None:
        for _ in range(self.episode_num):
            # TODO: 每一轮训练，记录一个episode的总reward用于评估
            episode_reward = self.train_episode()
            pass
    
    def train_episode(self)->None:
        '''
        一轮游戏
        '''
        # 清空episode_recorder记录的轨迹
        self.agent.episode_recorder.reset()
        # 重置环境
        obs = self.train_env.reset()
        done = False
        while not done:
            # 选择动作
            action = self.agent.get_action(obs)
            # 执行动作
            next_obs, reward, done = self.train_env.step(action)
            # 记录轨迹
            self.agent.episode_recorder.append(obs, action, reward, next_obs, done)
            # 更新状态
            obs = next_obs
        # 训练
        self.agent.train()

    def train(self)->None:
        for i in range(self.episode_num):
            self.train_episode()
            if i % EVAL_INTERVAL == 0:
                avg_reward = self.eval()
                print(f"Episode {i}, Average Reward: {avg_reward}")

    def eval(self)->None:
        obs = self.eval_env.reset()
        done = False
        test_reward = 0
        while not done:
            action = self.agent.get_action(obs)
            next_obs, reward, done = self.eval_env.step(action)
            obs = next_obs
            test_reward += reward
        
        return test_reward
        

if __name__ == "__main__":
    robot = Robot()
    map = Map('maps/map1.png')
    env = Env(robot, map)
    agent = PPO_Agent()
    train_manager = TrainManager(env, agent)
    train_manager.train()