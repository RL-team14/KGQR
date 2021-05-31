# KGQR

Implementation of ["Interactive Recommender System via Knowledge
Graph-enhanced Reinforcement Learning"](https://arxiv.org/pdf/2006.10389.pdf)


### File Tree 
- `raw_data`
  - `movie`
    - `ratrings.csv`: raw rating file of Movielens-20M dataset;
    - `sorted.csv`: sorted (by user id and timestamp) rating file of Movielens-20M dataset;
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
- `data`: preprocessed files
  - `movie`
    - `n_hop_kg.pkl`: save entity's n-hop relations in (Head, Relation, Tail) format;
        ```python
          n_hop_kg.pkl[entity_id][1] = list of 1-hop data (H=entity_id, R, T)
          n_hop_kg.pkl[entity_id][2] = list of 2-hop data (H=entity_ids in 1-hop data, R, T)
        ```  
    - `*_data_dict.pkl`: can refer user's rating history sorted in chronological order by user_id;
    - `*_vocab.pkl`: Change indicators in raw_data files to index used in this project