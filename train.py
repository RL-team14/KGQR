import numpy as np

import torch

from env import Simulator, Graph


def train(config):
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
        # return recommend item id
        return actions[np.random.choice(range(len(actions)))]
    # Delete above lines

    simulator = Simulator(config=config, mode='train')
    graph = Graph(config=config)
    num_users = len(simulator)
    for e in range(config.epochs):
        for u in range(num_users):
            user_id, item_ids, rates = simulator.get_data(u)
            x = []
            positive_ids = []
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
                    positive_ids.append(item_id)

                candidates_embeddings = item_ids  # Embed each item in n_hop_dict using each item's n_hop_dict
                # candidates_embeddings' shape = (# of candidates, config.item_embed_dim)

                # Recommendation using epsilon greedy policy
                recommend_item_id = tmp_Q_eps_greedy(state=embedded_state, actions=candidates_embeddings)
                reward = simulator.step(user_id, recommend_item_id)

                # TODO
                # Q learning
                # Store transition to buffer


if __name__ == '__main__':
    from env import Config
    train(Config())
