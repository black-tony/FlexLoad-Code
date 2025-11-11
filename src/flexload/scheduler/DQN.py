import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.1, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_net = DQNNet(state_dim, action_dim).to(device)
        self.target_net = DQNNet(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.batch_size = 64
        self.learn_step = 0
        self.target_update_freq = 100

    def choose_action(self, state, ava_nodes):
        if np.random.rand() < self.epsilon:
            # 探索：在可用节点中随机选
            return np.random.choice(ava_nodes)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state).detach().cpu().numpy().flatten()
            # 只选ava_nodes中的最大Q
            ava_q = [(a, q_values[a]) for a in ava_nodes]
            ava_q.sort(key=lambda x: -x[1])
            return ava_q[0][0]

    def store(self, s, a, r, s_, done):
        self.memory.push((s, a, r, s_, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_next_batch, done_batch = zip(*batch)
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        a_batch = torch.LongTensor(a_batch).unsqueeze(1).to(self.device)
        r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(self.device)
        s_next_batch = torch.FloatTensor(s_next_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        q_eval = self.policy_net(s_batch).gather(1, a_batch)
        q_next = self.target_net(s_next_batch).max(1)[0].detach().unsqueeze(1)
        q_target = r_batch + self.gamma * q_next * (1 - done_batch)
        loss = nn.MSELoss()(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

def flatten_state(s_grid, task_idx):
    # s_grid: [[deploy_state, [task_num], cpu_list, mem_list], ...]
    # 按需求拼接为一维特征
    deploy_state, task_num, cpu_list, mem_list = s_grid[task_idx]
    state = []
    for row in deploy_state:
        state.extend(row)
    state.extend(task_num[0])      # 当前master所有节点队列
    for c in cpu_list:
        state.extend(c)
    for m in mem_list:
        state.extend(m)
    # 可以加入更多特征（如任务类型等）
    return np.array(state, dtype=np.float32)



def get_act(dqn_agent, s_grid, ava_node, context=None, epsilon=0.1):
    # 返回act列表（每个任务一个动作），以及其它占位输出
    act = []
    for task_idx in range(len(s_grid)):
        if len(ava_node[task_idx]) == 0:
            # 没有可用节点，默认分配到cloud
            act.append(6)
            continue
        state = flatten_state(s_grid, task_idx)
        chosen = dqn_agent.choose_action(state, ava_node[task_idx])
        act.append(chosen)
    # 其它返回值根据你的主循环需要，可以填None或占位
    return act, None, None, None, None, None, None