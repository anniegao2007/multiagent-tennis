class SumTree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity-1)  # parent nodes are sums of children nodes, leaf nodes have priority of data point
        self.data = np.ndarray(capacity, dtype=object)  # contain experience tuples
        self.size = 0
        self.data_write_idx = 0
        
    def add(self, data, priority):
        tree_write_idx = self.data_write_idx + self.capacity - 1
        self.data[self.data_write_idx] = data
        self.data_write_idx = (self.data_write_idx + 1) % self.capacity
        self.update(tree_write_idx, priority)
        if self.size < self.capacity:
            self.size += 1
            
    def update(self, tree_idx, priority):
        diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        parent = (tree_idx-1) // 2
        while parent >= 0:
            self.tree[parent] += diff
            parent = (parent-1) // 2
            
    def get_sample(self, value):
        tree_idx = 0
        left = 1
        right = 2
        while left < len(self.tree):
            if value <= self.tree[tree_idx]:
                tree_idx = left
            else:
                tree_idx = right
                value -= self.tree[left]
            left = 2 * tree_idx + 1
            right = left + 1
        data_idx = tree_idx - self.capacity + 1
        if data_idx >= self.size:
            data_idx = random.randrange(0, self.size, step=1)
            tree_idx = data_idx + self.capacity - 1
        return (tree_idx, self.tree[tree_idx], self.data[data_idx])
    
    def get_size(self):
        return self.size
    
    def get_total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer():
    def __init__(self, capacity, minibatch_size, device, alpha=0.7, beta=0.4, beta_inc=1e-3, noise=1e-2):
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.buffer = SumTree(capacity)
        self.device = device
        self.minibatch_size = minibatch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.noise = noise
    
    def get_priority(self, error):
        return (np.abs(error) + self.noise) ** self.alpha
    
    def add(self, s, a, r, n_s, d, error):
        data = self.experience(s, a, r, n_s, d)
        priority = self.get_priority(error)
        self.buffer.add(data, priority)
        
    def update(self, idx, error):
        priority = self.get_priority(error)
        self.buffer.update(idx, priority)
        
    def sample(self):
        segment = self.buffer.get_total_priority() / self.minibatch_size
        experiences = []
        indices = []
        priorities = []
        
        for i in range(self.minibatch_size):
            left = segment * i
            right = segment * (i+1)
            value = random.uniform(left, right)
            tree_idx, priority, exp = self.buffer.get_sample(value)
            experiences.append(exp)
            indices.append(tree_idx)
            priorities.append(priority)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([1 if e.done else 0 for e in experiences if e is not None])).float().to(self.device)
        
        sampling_probabilities = priorities / self.buffer.get_total_priority()
        is_weights = np.power(self.buffer.get_size() * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(np.vstack([weight for weight in is_weights])).float().to(self.device)
        
        self.beta = np.min([1, self.beta + self.beta_inc])
        return (states, actions, rewards, next_states, dones, indices, is_weights)
    
    def get_len(self):
        return self.buffer.get_size()