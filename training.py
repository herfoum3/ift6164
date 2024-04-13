import torch
import ReplayBuffer
import GatheringEnv
import Agent
import numpy as np
import random
import os

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    #torch.use_deterministic_algorithms(True)

seed_everything(10)

def toTensor(states):
    states_tensors = []
    for s  in states:
        states_tensors.append(torch.tensor(s, dtype=torch.float32,device=device))
    return torch.stack(states_tensors)

def train_double_agents(env,
                        agent1, agent2,
                        episodes, batch_size, gamma):

    rates = []
    for episode in range(episodes):
        states = toTensor(env.reset()) # TODO may be just use directly a tensor for the grid
        dones = [False, False]
        agent1.q_network.train()
        agent2.q_network.train()
        while not all(dones):
            action1 = agent1.select_action(states[0])
            action2 = agent2.select_action(states[1])

            next_states, rewards, dones = env.step([action1, action2])
            next_states = toTensor(next_states)

            agent1.buffer.push(states[0], action1, rewards[0], next_states[0],dones[0])
            agent2.buffer.push(states[1], action2, rewards[1], next_states[1],dones[1])

            states = next_states

            # During training
            for agent in [agent1, agent2]:
                if len(agent.buffer) > batch_size:
                    batch = agent.buffer.sample(batch_size)
                    batch_states =[]
                    batch_actions = []
                    batch_rewards = []
                    batch_next_states = []
                    batch_dones = []
                    for elem in batch:
                        batch_states.append(elem[0])
                        batch_actions.append(elem[1])
                        batch_rewards.append(elem[2])
                        batch_next_states.append(elem[3])
                        batch_dones.append(elem[4])

                    batch_states = torch.stack(batch_states).float().to(device)
                    batch_next_states = torch.stack(batch_next_states).float().to(device)
                    batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=device)
                    batch_rewards =  torch.tensor(batch_rewards, dtype=torch.float32, device=device)
                    batch_dones = torch.tensor(batch_dones, dtype=torch.int, device=device)


                    current_q_values  = agent.q_network(batch_states).gather(1, batch_actions.unsqueeze(1))
                    max_next_q_values = agent.q_network(batch_next_states).detach().max(1)[0]
                    expected_q_values = batch_rewards + (gamma * max_next_q_values * (1 - batch_dones))

                    agent.loss = agent.loss_fn(current_q_values, expected_q_values.unsqueeze(1))

                    agent.optimizer.zero_grad()
                    agent.loss.backward()
                    agent.optimizer.step()

                    agent.update_epsilon()

        rates.append(env.beam_rate)
    return np.average(rates)


apples = 20
N_apple = 150
N_tagged = 2

Napples = [i for i in range(1, 100, 5)] # (2,20,40,80,100,150,200)
Ntaggs = [i for i in range(1, 21)]
beam_rates = np.zeros((len(Ntaggs),len(Napples)))


import time
t1 = time.time()
for i , N_apple in enumerate(Napples):
    for j, N_tagged in enumerate(Ntaggs):
        print(N_apple, '-->', N_tagged)
        env = GatheringEnv.GatheringEnv(apples=apples, N_apple=N_apple, N_tagged=N_tagged)
        agent1 = Agent.Agent(lr=0.01)
        agent2 = Agent.Agent(lr=0.01)
        beam_rate = train_double_agents(env,agent1,agent2, 10, 10, 0.99)
        beam_rates[(i,j)]=beam_rate

print('took: ', time.time() - t1)
print(beam_rates)


draw = True
#draw = False
if draw:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))

    sns.heatmap(beam_rates, annot=True, fmt=".2f", xticklabels=N_tagged, yticklabels=N_apple, cmap='viridis')
    plt.title('Beam use rate as a function of apple respawn and agent respawn times')
    plt.xlabel('N_tagged')
    plt.ylabel('N_apple')

    #plt.scatter(hit_rates, Napples, color='green', label='Apples')
    #plt.scatter(hit_rate, Ntaggs, color='red', label='Tagged')
    #plt.title('Apple respawn times as a Function of Hit Rate')
    #plt.xlabel('Hit Rate')
    #plt.ylabel('N_apple')
    #plt.legend()

    plt.savefig('heatmap3.png')
