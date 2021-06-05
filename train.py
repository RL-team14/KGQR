import numpy as np
import random
import torch

from env import Simulator, Graph


def train(config):
    memory = deque(maxlen=10000)
    policy_net = Net()
    target_net = Net()
    # TODO
    # Delete all tmp functions
    def tmp_item_embed(n_hop_dict):
        '''
            n_hop_dict[1]: First hop (Head, Relation, Tail)
            n_hop_dict[2]: Second hop (Head, Relation, Tail)
            return 1 item's embedding
        '''
        return torch.rand(config.item_embed_dim)
    
    def tmp_state_embed(embeded_item_list):
        return torch.rand(config.state_embed_dim)
    
    def tmp_Q_eps_greedy(state, actions):
        epsilon = 0.3
        state = torch.tensor(state, dtype=torch.float)
        out = policy_net.forward(state)
        out = out.detach().numpy()
        coin = random.random()
        if coin < epsilon:
            return actions[np.random.choice(range(len(actions)))]
        else:
            return actions[np.argmax(out)]
    
    # memory_buffer sampling
    def memory_sampling(memory):

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def optimize_model():
        state_action_values = policy_net(state_batch)    
        next_state_values = target_net(next_state_batch)
        for next_state_value in next_state_values:
            max_val = max(next_state_value).tolist()
            max_val_list.append(max_val)
        expected_state_action_values = state_action_values.tolist()
        for i in range(len(state_action_values)):
            action = action_batch[i]
            expected_state_action_values[i][action] = (max_val_list[i] * GAMMA) + reward_batch[i]
        expected_state_action_values = torch.tensor(expected_state_action_values)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        #print('loss', loss)
        optimizer = optim.RMSprop(self.policy_net.parameters())
        optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    simulator = Simulator(config=config, mode='train')
    graph = Graph(config=config)
    num_users = len(simulator)
    for e in range(config.epochs):
        for u in range(num_users):
            user_id, item_ids, rates = simulator.get_data(u)
            x = []
            candidates = []
            for t, (item_id, rate) in enumerate(zip(item_ids, rates)):
                # TODO
                # Embed item using GCN Algorithm1 line 6 ~ 7
                n_hop_dict = graph.get_n_hop(item_id)
                embedded_item = tmp_item_embed(n_hop_dict)
                x.append(embedded_item)
                embedded_state = tmp_state_embed(x)

                # TODO
                # Candidate selection and embedding
                if rate > config.threshold:
                    n_hop_dict = graph.get_n_hop(item_id)
                    candidates.extend(n_hop_dict[1])
                    candidates = list(set(candidates))      # Need to get rid of recommended items

                candidates_embeddings = item_ids  # Embed each item in n_hop_dict using each item's n_hop_dict
                # candidates_embeddings' shape = (# of candidates, config.item_embed_dim)

                # Recommendation using epsilon greedy policy
                recommend_item_id = tmp_Q_eps_greedy(state=embedded_state, actions=candidates_embeddings)
                reward = simulator.step(user_id, recommend_item_id)

                # TODO
                # Q learning
                # Store transition to buffer
                state, action, reward, next_state, done = embedded_state, recommend_item_id, 
                    reward, tmp_state_embed(x.append(recommend_item_id)), done? # done을 어떻게 하지?
                Tuple = (state, action, reward, next_state, done)        
                memory.append(Tuple)
                # target update
                if total_step_count % TARGET_UPDATE ==0:
                    target_net.load_state_dict(policy_net.state_dict())
                optimize_model()
                
if __name__ == '__main__':
    from env import Config
    train(Config())
