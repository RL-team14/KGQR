import numpy as np
import pickle
import torch
import torch.optim as optim
from model import GCN_GRU
from env import Simulator, Config
from dataloader import *

def pretrain_embedding(config, entity_vocab, relation_vocab, model, optimizer):
	model.train()

	dataloader = get_TransE_dataloader(config, entity_vocab, relation_vocab)
	for epoch in range(200):
		total_loss = 0
		for positive_triples, negative_triples in dataloader:	
			optimizer.zero_grad()
			loss = model.TransE_forward(positive_triples, negative_triples)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
		print('TransE epoch', epoch, 'loss', total_loss)


def train(config, item_vocab, model, optimizer):

	def tmp_Q_eps_greedy(state, actions):
		# return recommend item id
		return actions[np.random.choice(range(len(actions)))]
	# Delete above lines

	simulator = Simulator(config=config, mode='train')
	num_users = len(simulator)
	for e in range(config.epochs):
		for u in range(num_users):
			user_id, item_ids, rates = simulator.get_data(u)
			candidates = []
			print('user_id:', user_id)
			for t, (item_id, rate) in enumerate(zip(item_ids, rates)):
				print('t',t,'item_id',item_id,'rate',rate)
				# TODO
				# Embed item using GCN Algorithm1 line 6 ~ 7
				item_idx = item_id
				embedded_item_state = model.forward_GCN(item_idx)	# (50)
				embedded_user_state = model(item_idx)				# (20)

				# TODO
				# Candidate selection and embedding
				if rate > config.threshold:
					n_hop_dict = model.get_n_hop(item_id)
					candidates.extend(n_hop_dict[1])
					candidates = list(set(candidates))      # Need to get rid of recommended items

				candidates_embeddings = model.forward_GCN(torch.tensor(candidates))
				print('candidate shape:',candidates_embeddings.shape)
				# candidates_embeddings = item_ids  # Embed each item in n_hop_dict using each item's n_hop_dict
				# candidates_embeddings' shape = (# of candidates, config.item_embed_dim)

				# Recommendation using epsilon greedy policy
				recommend_item_id = tmp_Q_eps_greedy(state=embedded_user_state, actions=candidates_embeddings)
				reward = simulator.step(user_id, recommend_item_id)

				# TODO
				# Q learning
				# Store transition to buffer


if __name__ == '__main__':

	with open('./data/movie/entity_vocab.pkl','rb') as f:
		entity_vocab = pickle.load(f)
	with open('./data/movie/item_vocab.pkl','rb') as f:
		item_vocab = pickle.load(f)
	with open('./data/movie/relation_vocab.pkl','rb') as f:
		relation_vocab = pickle.load(f)

	print('| Building Net')
	model = GCN_GRU(Config(), 50, entity_vocab, relation_vocab)
	optimizer = optim.SGD(model.parameters(), lr=0.01)
	
	print('Embedding pretrain by TransE...')
	pretrain_embedding(Config(), entity_vocab, relation_vocab, model, optimizer)

	print('Save embedding_pretrained model...')
	path = './embedding_pretrained.pth'
	torch.save(model.state_dict(),path)
	
	print('Load embedding_pretrained model...')
	path = './embedding_pretrained.pth'
	model.load_state_dict(torch.load(path))

	print('Train...')
	train(Config(), item_vocab, model, optimizer)
