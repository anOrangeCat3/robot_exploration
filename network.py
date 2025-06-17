import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class PPO_Network(nn.Module):
    def __init__(self,
                 obs_dim:Tuple[int, int],  # 地图大小，用于计算卷积后的特征图大小
                 action_dim:int,          # 动作空间大小
                 num_inputs:int=1,        # 灰度图，单通道
                 device:torch.device=torch.device("cuda"),  # 默认使用cuda
                 ) -> None:
        super(PPO_Network, self).__init__()
        
        # 特征提取网络
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4),  # (200-8)/4 + 1 = 49
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (49-4)/2 + 1 = 24
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (24-3)/1 + 1 = 22
            nn.ReLU(),
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(20*28*64, 512),
            nn.ReLU(),
            nn.Dropout(0.1)  # 添加dropout防止过拟合
        )
        
        # 策略网络输出层（Actor）
        self.fc_mu = nn.Linear(512, action_dim)  # 输出动作均值
        self.fc_std = nn.Linear(512, action_dim)  # 输出动作标准差
        
        # 价值网络输出层（Critic）
        self.fc_v = nn.Linear(512, 1)  # 输出状态价值

        # 将网络移动到指定设备
        self.to(device)
    
    
    def _extract_features(self, x):
        """
        提取特征
        Args:
            x: 输入状态，shape (batch_size, channels, height, width)
        Returns:
            features: 提取的特征，shape (batch_size, 512)
        """
        # 打印输入维度
        # print(f"网络输入维度: {x.shape}")
        if len(x.shape) == 2: 
            # get_action时，输入维度为(H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            # train时，输入维度为(batch_size, H, W)
            x = x.unsqueeze(1)
        
        # 通过卷积层
        x = self.conv_layers(x)
        # print(f"卷积后特征维度: {x.shape}")
        
        # 展平
        x = x.view(x.size(0), -1)
        # print(f"展平后特征维度: {x.shape}")
        
        # 通过全连接层
        x = self.fc(x)
        # print(f"全连接层后特征维度: {x.shape}")
        
        return x
    
    def pi(self, x):
        """
        策略网络(Actor)
        Args:
            x: 输入状态
        Returns:
            action_mean: 动作均值
            action_std: 动作标准差
        """
        features = self._extract_features(x)
        action_mean = self.fc_mu(features)
        # 使用更稳定的标准差计算方式
        action_std = F.softplus(self.fc_std(features)) + 1e-6
        action_std = torch.clamp(action_std, min=1e-6, max=1.0)  # 限制标准差范围
        # action_std = F.softplus(self.fc_std(features)) + 1e-5
        # print(f"动作均值维度: {action_mean.shape}")
        # print(f"动作标准差维度: {action_std.shape}")
        return action_mean, action_std
    
    def v(self, x):
        """
        价值网络(Critic)
        Args:
            x: 输入状态
        Returns:
            value: 状态价值
        """
        # print(f"价值网络输入维度: {x.shape}")
        features = self._extract_features(x)
        value = self.fc_v(features)
        # print(f"状态价值维度: {value.shape}")
        return value
