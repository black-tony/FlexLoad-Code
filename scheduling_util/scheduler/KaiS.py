import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import bisect


def discount(x, gamma):
    out = np.zeros_like(x)
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out


class GraphCNN(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim, max_depth, act_fn):
        super(GraphCNN, self).__init__()
        self.prep_layers = self._init_layers(input_dim, hid_dims, output_dim, act_fn)
        self.proc_layers = self._init_layers(output_dim, hid_dims, output_dim, act_fn)
        self.agg_layers = self._init_layers(output_dim, hid_dims, output_dim, act_fn)
        self.max_depth = max_depth

    def _init_layers(self, input_dim, hid_dims, output_dim, act_fn):
        layers = []
        curr_in_dim = input_dim
        for hid_dim in hid_dims:
            layers.append(nn.Linear(curr_in_dim, hid_dim))
            curr_in_dim = hid_dim
        layers.append(nn.Linear(curr_in_dim, output_dim))
        return nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.prep_layers:
            x = F.relu(layer(x))
        for _ in range(self.max_depth):
            y = x
            for layer in self.proc_layers:
                y = F.relu(layer(y))
            for layer in self.agg_layers:
                y = F.relu(layer(y))
            x = x + y
        return x


class GraphSNN(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim, act_fn):
        super(GraphSNN, self).__init__()
        self.dag_layers = self._init_layers(input_dim, hid_dims, output_dim, act_fn)
        self.global_layers = self._init_layers(output_dim, hid_dims, output_dim, act_fn)

    def _init_layers(self, input_dim, hid_dims, output_dim, act_fn):
        layers = []
        curr_in_dim = input_dim
        for hid_dim in hid_dims:
            layers.append(nn.Linear(curr_in_dim, hid_dim))
            curr_in_dim = hid_dim
        layers.append(nn.Linear(curr_in_dim, output_dim))
        return nn.ModuleList(layers)

    def forward(self, x):
        summaries = []
        s = x
        for layer in self.dag_layers:
            s = F.relu(layer(s))
        summaries.append(s)
        for layer in self.global_layers:
            s = F.relu(layer(s))
        summaries.append(s)
        return summaries


class OrchestrateAgent(nn.Module):
    def __init__(self, node_input_dim, cluster_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, eps=1e-6, act_fn=F.leaky_relu):
        super(OrchestrateAgent, self).__init__()
        self.node_input_dim = node_input_dim
        self.cluster_input_dim = cluster_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.eps = eps
        self.act_fn = act_fn

        self.gcn = GraphCNN(node_input_dim, hid_dims, output_dim, max_depth, act_fn)
        self.gsn = GraphSNN(node_input_dim + output_dim, hid_dims, output_dim, act_fn)

        self.node_fc = nn.Sequential(
            nn.Linear(output_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.cluster_fc = nn.Sequential(
            nn.Linear(cluster_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, node_inputs, cluster_inputs):
        gcn_outputs = self.gcn(node_inputs)
        gsn_summaries = self.gsn(torch.cat([node_inputs, gcn_outputs], dim=1))

        node_outputs = self.node_fc(torch.cat([node_inputs, gcn_outputs], dim=1))
        node_outputs = F.softmax(node_outputs, dim=-1)

        expanded_state = torch.cat([cluster_inputs], dim=2)  # Expand this as needed
        cluster_outputs = self.cluster_fc(expanded_state)
        cluster_outputs = cluster_outputs.view(-1, len(self.executor_levels))
        cluster_outputs = F.softmax(cluster_outputs, dim=-1)

        return node_outputs, cluster_outputs
    def orchestrate_network(self, node_inputs, gcn_outputs, cluster_inputs, gsn_dag_summary, gsn_global_summary):
        # This method can be integrated into the `forward` method if needed
        pass

    def get_action(self, node_act_probs, cluster_act_probs):
        # Sample actions based on probabilities
        node_acts = torch.multinomial(node_act_probs, num_samples=3)
        cluster_acts = torch.multinomial(cluster_act_probs.view(-1, len(self.executor_levels)), num_samples=3)
        return node_acts, cluster_acts

    def compute_loss(self, node_act_probs, cluster_act_probs, node_act_vec, cluster_act_vec, adv, entropy_weight):
        # Compute selected probabilities
        selected_node_prob = (node_act_probs * node_act_vec).sum(dim=1, keepdim=True)
        selected_cluster_prob = (cluster_act_probs * cluster_act_vec).sum(dim=2).sum(dim=1, keepdim=True)

        # Advantage loss
        adv_loss = -torch.sum(torch.log(selected_node_prob * selected_cluster_prob + self.eps) * adv)

        # Entropy loss
        node_entropy = -(node_act_probs * torch.log(node_act_probs + self.eps)).sum()
        entropy_loss = node_entropy / (torch.log(torch.tensor(node_act_probs.size(1), dtype=torch.float32)) +
                                       torch.log(torch.tensor(len(self.executor_levels), dtype=torch.float32)))

        # Total loss
        act_loss = adv_loss + entropy_weight * entropy_loss
        return act_loss

    def update_gradients(self, optimizer, node_inputs, cluster_inputs, node_act_vec, cluster_act_vec, adv, entropy_weight):
        optimizer.zero_grad()
        node_act_probs, cluster_act_probs = self.forward(node_inputs, cluster_inputs)
        loss = self.compute_loss(node_act_probs, cluster_act_probs, node_act_vec, cluster_act_vec, adv, entropy_weight)
        loss.backward()
        optimizer.step()
        return loss.item()

def decrease_var(var, min_var, decay_rate):
    return max(var - decay_rate, min_var)


def get_piecewise_linear_fit_baseline(all_cum_rewards, all_wall_time):
    assert len(all_cum_rewards) == len(all_wall_time)
    unique_wall_time = np.unique(np.hstack(all_wall_time))
    baseline_values = {}
    for t in unique_wall_time:
        baseline = 0
        for i in range(len(all_wall_time)):
            idx = bisect.bisect_left(all_wall_time[i], t)
            if idx == 0:
                baseline += all_cum_rewards[i][idx]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            elif all_wall_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            else:
                baseline += (
                    (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) /
                    (all_wall_time[i][idx] - all_wall_time[i][idx - 1]) *
                    (t - all_wall_time[i][idx]) + all_cum_rewards[i][idx]
                )
        baseline_values[t] = baseline / float(len(all_wall_time))
    baselines = [np.array([baseline_values[t] for t in wall_time]) for wall_time in all_wall_time]
    return baselines


def train_orchestrate_agent(orchestrate_agent, exp, entropy_weight, entropy_weight_min, entropy_weight_decay):
    all_cum_reward = []
    all_rewards = exp['reward']
    batch_time = exp['wall_time']
    all_times = batch_time[1:]
    all_diff_times = np.array(batch_time[1:]) - np.array(batch_time[:-1])
    rewards = np.array([r for (r, t) in zip(all_rewards, all_diff_times)])
    cum_reward = discount(rewards, 1)
    all_cum_reward.append(cum_reward)

    baselines = get_piecewise_linear_fit_baseline(all_cum_reward, [all_times])
    batch_adv = all_cum_reward[0] - baselines[0]
    batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])

    loss = orchestrate_agent.compute_loss(exp, batch_adv, entropy_weight)
    entropy_weight = decrease_var(entropy_weight, entropy_weight_min, entropy_weight_decay)
    return entropy_weight, loss


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from copy import deepcopy


class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU()):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.fc(x))


class Estimator(nn.Module):
    def __init__(self, action_dim, state_dim, n_valid_node, scope="estimator", summaries_dir=None):
        super(Estimator, self).__init__()
        self.n_valid_node = n_valid_node
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.scope = scope
        self.T = 144

        self.value_model = nn.Sequential(
            FullyConnected(state_dim, 128),
            FullyConnected(128, 64),
            FullyConnected(64, 32),
            FullyConnected(32, 1, activation=nn.Identity())
        )

        self.policy_model = nn.Sequential(
            FullyConnected(state_dim, 128),
            FullyConnected(128, 64),
            FullyConnected(64, 32),
            FullyConnected(32, action_dim, activation=nn.Identity())
        )

    def forward_value(self, state):
        return self.value_model(state)

    def forward_policy(self, state, neighbor_mask):
        logits = self.policy_model(state) + 1
        valid_logits = logits * neighbor_mask
        softmax_prob = torch.softmax(torch.log(valid_logits + 1e-8), dim=-1)
        return softmax_prob

    def predict(self, s):
        with torch.no_grad():
            return self.forward_value(s)

    def action(self, s, ava_node, context, epsilon):
        # change s to tensor
        s = torch.tensor(s, dtype=torch.float32)
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        value_output = self.predict(s).flatten()
        action_tuple = []
        valid_prob = []

        action_choosen_mat = []
        policy_state = []
        curr_state_value = []
        next_state_ids = []

        grid_ids = [x for x in range(self.n_valid_node)]

        valid_action_mask = np.zeros((self.n_valid_node, self.action_dim))
        for i in range(len(ava_node)):
            for j in ava_node[i]:
                valid_action_mask[i][j] = 1
        curr_neighbor_mask = deepcopy(valid_action_mask)

        valid_neighbor_node_id = [[i for i in range(self.action_dim)], [i for i in range(self.action_dim)]]

        action_probs = self.forward_policy(s, torch.tensor(curr_neighbor_mask, dtype=torch.float32)).numpy()
        curr_neighbor_mask_policy = []

        for idx, grid_valid_idx in enumerate(grid_ids):
            action_prob = action_probs[idx]
            valid_prob.append(action_prob)
            if int(context[idx]) == 0:
                continue
            curr_action_indices_temp = np.random.choice(self.action_dim, int(context[idx]),
                                                        p=action_prob / np.sum(action_prob))
            curr_action_indices = [0] * self.action_dim
            for kk in curr_action_indices_temp:
                curr_action_indices[kk] += 1

            valid_neighbor_grid_id = valid_neighbor_node_id
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    end_node_id = int(valid_neighbor_node_id[grid_valid_idx][curr_action_idx])
                    action_tuple.append(end_node_id)

                    temp_a = np.zeros(self.action_dim)
                    temp_a[curr_action_idx] = 1
                    action_choosen_mat.append(temp_a)
                    policy_state.append(s[idx])
                    curr_state_value.append(value_output[idx])
                    next_state_ids.append(valid_neighbor_grid_id[grid_valid_idx][curr_action_idx])
                    curr_neighbor_mask_policy.append(curr_neighbor_mask[idx])

        return action_tuple, np.stack(valid_prob), \
               np.stack(policy_state), np.stack(action_choosen_mat), curr_state_value, \
               np.stack(curr_neighbor_mask_policy), next_state_ids

    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, gamma):
        advantage = []
        node_reward = node_reward.flatten()
        qvalue_next = self.predict(next_state).flatten()
        for idx, next_state_id in enumerate(next_state_ids):
            temp_adv = sum(node_reward) + gamma * sum(qvalue_next) - curr_state_value[idx]
            advantage.append(temp_adv)
        return advantage

    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        targets = []
        node_reward = node_reward.flatten()
        qvalue_next = self.predict(next_state).flatten()

        for idx in np.arange(self.n_valid_node):
            grid_prob = valid_prob[idx][self.valid_action_mask[idx] > 0]
            curr_grid_target = np.sum(
                grid_prob * (sum(node_reward) + gamma * sum(qvalue_next)))
            targets.append(curr_grid_target)

        return np.array(targets).reshape([-1, 1])

    def initialization(self, s, y, learning_rate):
        optimizer = optim.Adam(self.value_model.parameters(), lr=learning_rate)
        value_output = self.forward_value(s)
        loss = nn.MSELoss()(value_output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def update_policy(self, policy_state, advantage, action_choosen_mat, curr_neighbor_mask, learning_rate):
        optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        softmax_prob = self.forward_policy(policy_state, curr_neighbor_mask)
        log_softmax_prob = torch.log(softmax_prob)
        neglogprob = -log_softmax_prob * action_choosen_mat
        actor_loss = torch.mean(torch.sum(neglogprob * advantage, dim=1))
        entropy = -torch.mean(softmax_prob * log_softmax_prob)
        policy_loss = actor_loss - 0.01 * entropy

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        return policy_loss.item()

    def update_value(self, s, y, learning_rate):
        optimizer = optim.Adam(self.value_model.parameters(), lr=learning_rate)
        value_output = self.forward_value(s)
        loss = nn.MSELoss()(value_output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


class PolicyReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.neighbor_mask = []
        self.actions = []
        self.rewards = []
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r, mask):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.neighbor_mask[index:(index + new_sample_lens)] = mask

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask]
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.neighbor_mask[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.neighbor_mask = []
        self.curr_lens = 0


class ReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r, next_s):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states]
        indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_mask = self.next_states[indices]
        return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.curr_lens = 0


class ModelParametersCopier:
    def __init__(self, estimator1, estimator2):
        self.estimator1 = estimator1
        self.estimator2 = estimator2

    def copy(self):
        for param1, param2 in zip(self.estimator1.parameters(), self.estimator2.parameters()):
            param2.data.copy_(param1.data)
