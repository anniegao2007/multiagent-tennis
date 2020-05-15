class Agent():
    def __init__(self, state_size, action_size, device, buffer_capacity, minibatch_size, lr_actor=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.buffer = PrioritizedReplayBuffer(int(buffer_capacity), minibatch_size, self.device)
        self.noise = OrnsteinUhlenbeck(num_agents=num_agents, action_size=action_size)
        
        self.actor_local = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.copy_weights(self.actor_target, self.actor_local)
        self.optim_actor = optim.Adam(self.actor_local.parameters(), lr=lr_actor)    
    
    def copy_weights(self, target_network, source_network):
        for target_param, param_to_copy in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(param_to_copy.data)
    
    def reset(self):
        self.noise.reset()
    
    def select_action(self, state, use_local=True):  # states is shape (1, state_size)
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        if use_local:
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state)  # shape (1, action_size)
            self.actor_local.train()
            return np.clip(action.cpu().data.numpy() + self.noise.sample(), -1, 1)
        else:  # for updating priorities
            self.actor_target.eval()
            with torch.no_grad():
                action = self.actor_target(state)
            self.actor_target.train()
            return action
    
    def add_to_buffer(self, state, action, reward, next_state, done, td_error=None):
        if td_error is None:
            self.buffer.add(state, action, reward, next_state, done)
        else:     
            self.buffer.add(state, action, reward, next_state, done, td_error)
    
    def get_buffer_samples(self):
        return self.buffer.sample()

class MADDPG():
    def __init__(self, state_size, action_size, num_agents, buffer_capacity=1e6, minibatch_size=256, update_freq=1, tau=1e-3, gamma=0.99, lr_critic=1e-3):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.agents = [Agent(state_size, action_size, self.device, buffer_capacity, minibatch_size) for i in range(num_agents)]
        
        self.shared_critic_local = Critic(state_size, action_size).to(self.device)
        self.shared_critic_target = Critic(state_size, action_size).to(self.device)
        self.optim_critic = optim.Adam(self.shared_critic_local.parameters(), lr=lr_critic)
        self.copy_weights(self.shared_critic_target, self.shared_critic_local)
        
        self.minibatch_size = minibatch_size
        self.update_freq = update_freq
        self.tau = tau
        self.gamma = gamma
        self.num_steps = 0
    
    def copy_weights(self, target_network, source_network):
        for target_param, param_to_copy in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(param_to_copy.data)
            
    def get_actions(self, states):
        actions = [self.agents[i].select_action(states[i]) for i in range(self.num_agents)]
        return np.concatenate(actions)
    
    def reset(self):
        for i in range(self.num_agents):
            self.agents[i].reset()
    
    def update(self, states, actions, rewards, next_states, dones):
        # calculate td_error and pass to agent.update()
        for i in range(self.num_agents):
            s = torch.from_numpy(states[i]).unsqueeze(0).float().to(self.device)
            a = torch.from_numpy(actions[i]).unsqueeze(0).float().to(self.device)
            n_s = torch.from_numpy(next_states[i]).unsqueeze(0).float().to(self.device)
            self.shared_critic_local.eval()
            self.shared_critic_target.eval()
            with torch.no_grad():
                expected_val = self.shared_critic_local((s, a))
                selected_action = self.agents[i].select_action(next_states[i], use_local=False)
                target_q = self.shared_critic_target((n_s, selected_action))
            self.shared_critic_local.train()
            self.shared_critic_target.train()
            done = 1 if dones[i] else 0
            target_val = rewards[i] + self.gamma*target_q*(1-done)
            td_error = target_val.item() - expected_val.item()
            self.agents[i].add_to_buffer(states[i], actions[i], rewards[i], next_states[i], done, td_error)
        
        self.num_steps = (self.num_steps + 1) % self.update_freq
        if self.num_steps == 0 and self.agents[0].buffer.get_len() >= self.minibatch_size:
            self.update_local()
            
    def update_local(self):
        for i in range(self.num_agents):
            s, a, r, n_s, d, indices, is_weights = self.agents[i].get_buffer_samples()
            
            # update actor
            actions_pred = self.agents[i].actor_local(s)
            actor_loss = -self.shared_critic_local((s, actions_pred)).mean()
            self.agents[i].optim_actor.zero_grad()
            actor_loss.backward()
            self.agents[i].optim_actor.step()

            # update critic
            next_actions_target = self.agents[i].actor_target(n_s)
            future_rewards_target = self.shared_critic_target((n_s, next_actions_target))
            value_target = r + self.gamma*future_rewards_target*(1-d)
            value_local = self.shared_critic_local((s, a))

            errors = torch.abs(value_local - value_target).cpu().data.numpy()  # update priorities in buffer
            for j in range(self.minibatch_size):
                self.agents[i].buffer.update(indices[j], errors[j])

            critic_loss = F.mse_loss(value_local, value_target * is_weights).mean()
            self.optim_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.shared_critic_local.parameters(), 1)
            self.optim_critic.step()

            self.soft_update(self.agents[i].actor_target, self.agents[i].actor_local)
        self.soft_update(self.shared_critic_target, self.shared_critic_local)
            
    def soft_update(self, target_network, local_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1-self.tau)*target_param.data)