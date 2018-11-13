import gym
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym import wrappers
env = gym.make("Walker2d-v2")
#env = wrappers.Monitor(env, "Walker2d-",force=True)
env.action_space.shape[0]
env.observation_space.shape[0]
seed = 333
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.set_default_tensor_type('torch.DoubleTensor')
class Memory_Buffer:
    def __init__(self):
        self.memory = []
        self.states = []
        self.rewards = []
        self.dones = []
        self.actions = []
        self.length_mem = 0
        pass

    def add_memory(self,state,action,next_state,reward,done):
        self.memory.append((state,action,next_state,reward,done))
        pass

    def clear_memory(self):
        self.memory = []
        pass

    def process_buffer(self):
        mem = list(zip(*self.memory))
        self.length_mem = len(self.memory)
        self.states = torch.Tensor(mem[0])
        self.actions = torch.Tensor(mem[1])
        self.rewards = torch.Tensor(mem[3])
        self.dones = mem[4] * 1
        pass

class Actor_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor_Network, self).__init__()
        self.xw_size = 64
        self.hw_size = 64
        self.x_layer = nn.Linear(input_dim, self.xw_size)
        self.h_layer = nn.Linear(self.xw_size, self.hw_size)
        self.mu_output = nn.Linear(self.hw_size, output_dim)
        self.log_sigma = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        x = F.tanh(self.x_layer(x))
        x = F.tanh(self.h_layer(x))
        mu = self.mu_output(x)
        sigma = torch.exp(self.log_sigma.expand_as(mu))
        return mu, sigma

class Critic_Network(nn.Module):
    def __init__(self, num_inputs):
        super(Critic_Network, self).__init__()
        self.xw_size = 64
        self.hw_size = 64
        self.x_layer = nn.Linear(num_inputs, self.xw_size)
        self.h_layer = nn.Linear(self.xw_size, self.hw_size)
        self.y_layer = nn.Linear(self.hw_size, 1)
    def forward(self, x):
        x = F.tanh(self.x_layer(x))
        x = F.tanh(self.h_layer(x))
        state_values = self.y_layer(x)
        return state_values

def normalizer(x):
    return (x - x.mean()) / (x.std())

class PPO_Agent():
    def __init__(self, input_dim, output_dim):
        self.m_buffer = Memory_Buffer()
        self.actor = Actor_Network(input_dim, output_dim)
        self.critic = Critic_Network(input_dim)
        self.discounts = 0.995
        self.tau = 0.97
        self.criterion = nn.MSELoss()
        self.targets = []
        self.advantages = []
        self.state_values = []
        self.actions_log_probablities_old = []
        self.mu_old = []
        self.sigma_old = []
        self.log_sigma_old = []
        self.sigma_old2 = []
        self.actions_old = []
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=1e-2)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=1e-4)
        self.critic_optimization_steps = 5
        self.actor_optimization_steps = 20
        self.kl_div_list = []
        self.kl_div_averages_data = []
        self.step_size = 1
        self.step_rate = 10
        pass

    def process_training_data(self):
        next = 0
        self.state_values = self.critic(Variable(self.m_buffer.states))
        # must go from the last advantage to the first in-order to have the previous advantage discount on the next (GAE)
        # GAE Paper: https://arxiv.org/pdf/1506.02438.pdf
        for i in reversed(range(self.m_buffer.length_mem)):
            discounts_mult_done = self.discounts * self.m_buffer.dones[i] #faster to just compute this once TODO: multiply the whole dones vector with discounts at once
            self.targets[i] = self.m_buffer.rewards[i] + self.targets[i + next] * discounts_mult_done # targets for the critic
            td_error = (self.m_buffer.rewards[i] + self.state_values.data[i + next] * discounts_mult_done - self.state_values.data[i]) #surprise :D
            self.advantages[i] = td_error + (self.advantages[i + next] * discounts_mult_done * self.tau) #generalized advantages for the actor
            next = 1
        self.advantages = normalizer(self.advantages)

    def critic_objective_function(self):
        values_ = self.critic(Variable(self.m_buffer.states))
        loss = self.criterion(values_, Variable(self.targets))
        return loss


    def actor_objective_function(self, clip, optimizer, first):
        mu, sigma = self.actor(Variable(self.m_buffer.states, volatile=False))
        #KL(p||q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_{1}^{2} + (\mu_1-\mu_2)^2}{2\sigma_{2}^{2}} - \frac{1}{2}
        log_sigma_two_div_sigma_one = torch.log(sigma) - self.log_sigma_old
        ratio = (self.sigma_old.pow(2) + (self.mu_old - mu).pow(2)) / (sigma.pow(2) * 2)
        kl_div = log_sigma_two_div_sigma_one + ratio - 0.5
        kl_div_sum = np.abs(np.sum(kl_div.data.numpy()))
        self.kl_div_list.append(kl_div_sum)
        #adjust clipping based on the kl rate
        adjusted_clip = 0.2*self.step_size
        if not first:
            while kl_div_sum > 0:
                adjusted_clip -= (0.02*self.step_size)/self.step_rate
                kl_div_sum -= 10/self.step_rate
        if first:
            adjusted_clip = 0.06

        if adjusted_clip < 0.005:
            adjusted_clip = 0.005

        action_log_probabilities = torch.distributions.Normal(mu, sigma).log_prob(Variable(self.m_buffer.actions)).sum(1,keepdim=True)
        ratio = torch.exp(action_log_probabilities) / torch.exp(Variable(self.actions_log_probablities_old))
        ppoclip = ratio.clamp(1 - adjusted_clip, 1 + adjusted_clip)
        surr1 = Variable(self.advantages) * ppoclip
        surr2 = Variable(self.advantages) * ratio
        action_loss = -torch.min(surr1, surr2)

        #-- vanilla_pg_loss --
        #action_loss = -(action_log_probabilities * Variable(self.advantages))

        return action_loss.mean()


    def learn(self):
        self.m_buffer.process_buffer()
        self.targets = torch.zeros(self.m_buffer.length_mem, 1)
        self.advantages = torch.zeros(self.m_buffer.length_mem, 1)
        self.process_training_data()
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=0.0007)
        for i in range(self.critic_optimization_steps):
            self.critic_optimizer.zero_grad()
            self.critic_objective_function().backward()
            nn.utils.clip_grad_norm(self.actor.parameters(), 20)
            self.critic_optimizer.step()

        self.batch_action_log_probablity_old()
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=0.0001)
        first = True
        for i in range(self.actor_optimization_steps):
            self.actor_optimizer.zero_grad()
            self.actor_objective_function(0.2, self.actor_optimizer, first).backward()
            nn.utils.clip_grad_norm(self.actor.parameters(), 2)
            self.actor_optimizer.step()
            if i == 0:
                first = False
        #used for plotting kl
        self.kl_div_averages_data.append(np.sum(self.kl_div_list) / self.actor_optimization_steps)
        self.kl_div_list = []

    def get_batch_mu_sigma(self):
        mu_old, sigma_old = self.actor(Variable(self.m_buffer.states))
        self.mu_old = mu_old
        self.log_sigma_old = self.actor.log_sigma
        self.sigma_old = sigma_old
        self.actions_old = torch.distributions.Normal(mu_old, sigma_old)

    def batch_action_log_probablity_old(self):
        self.get_batch_mu_sigma()
        self.actions_log_probablities_old = self.actions_old.log_prob(Variable(self.m_buffer.actions)).sum(1,keepdim=True).data.clone()

    def select_action(self,state):
        state = torch.from_numpy(state)
        mu, sigma = self.actor(Variable(state.unsqueeze(0)))
        action = torch.distributions.Normal(mu, sigma)
        return action.sample().data[0].numpy()


class Standardizer:
    def __init__(self):
        self.n = 0
        self.scaled_mean = 0
        self.scaled_variance = 0

    def process(self, data):
        self.scaled_mean = ((self.scaled_mean * self.n) + data) / (self.n + 1)
        self.scaled_variance = ((self.scaled_variance * self.n) + np.power((data - self.scaled_mean),2)) / (self.n + 1)
        standard_deviation = np.sqrt(self.scaled_variance + 0.000001)
        self.n += 1
        output = (data - self.scaled_mean) / standard_deviation
        return output

standardizer = Standardizer()
batch_size = 1000
steps = 0
epochs = 2000
render = False
plot_data = []
agent = PPO_Agent(env.observation_space.shape[0],env.action_space.shape[0])
for e in range(epochs):
    steps = 0
    rewards = 0
    rewards_per_batch = 0
    rewards_per_batch_counter = 0
    agent.m_buffer.clear_memory()
    while steps < batch_size:
        state = standardizer.process(env.reset())
        rewards = 0
        for i in range(100000):
            #if (e > 1500):
            #    env.render()
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = standardizer.process(next_state)
            rewards += reward
            agent.m_buffer.add_memory(state, action, next_state, reward, not done)
            if done:
                break
            state = next_state
        steps += i-1
        rewards_per_batch_counter += 1
        rewards_per_batch += rewards
    agent.learn()
    plot_data.append(rewards_per_batch / rewards_per_batch_counter)
    print('epoch: ', e, 'rewards: ', rewards_per_batch / rewards_per_batch_counter)
    pass

n = 1
t = np.arange(0, len(plot_data) * n, n)
plt.plot(t, plot_data)
plt.legend(['batch_avg_return'], loc='upper left')
plt.xlabel('Episodes (1000 timesteps)')
plt.title('Proximal Policy Optimization: adjusted clipping step_size = ' + str(agent.step_size))
#plt.title('Vanilla Actor Critic')
plt.ylabel('Average return of batch size 1000')
plt.grid(True)
plt.show()

plt.plot(t, agent.kl_div_averages_data)
plt.legend(['Kullback–Leibler divergence'], loc='upper left')
plt.xlabel('Episodes (1000 timesteps)')
plt.title('Proximal Policy Optimization: adjusted clipping step_size = ' + str(agent.step_size))
plt.ylabel('Amount of divergence between old policy and new policies')
plt.grid(True)
plt.show()
