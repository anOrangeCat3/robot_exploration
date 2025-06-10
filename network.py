import torch.nn as nn
import torch.nn.functional as F

class PPO_Network(nn.Module):
    def __init__(self,
                 num_inputs:int,
                 action_dim:int,
                 ) -> None:
        super(PPO_Network, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        
        # 修改策略网络输出层，输出均值和标准差
        self.fc_mu = nn.Linear(512, action_dim)
        self.fc_std = nn.Linear(512, action_dim) # 可学习的标准差参数
        
        # 价值网络输出层
        self.fc_v= nn.Linear(512, 1)
    
    def _extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return x
    
    def pi(self, x):
        """
        Actor network (Policy network)
        Args:
            x: input state
        Returns:
            action_mean: 动作均值
            action_std: 动作标准差
        """
        features = self._extract_features(x)
        action_mean = self.fc_mu(features)
        # 使用softplus替代exp，确保标准差非负且更稳定
        action_std = F.softplus(self.fc_std(features))
        return action_mean, action_std
    
    def v(self, x):
        """
        Critic network (Value network)
        Args:
            x: input state
        Returns:
            value
        """
        features = self._extract_features(x)
        return self.fc_v(features)
