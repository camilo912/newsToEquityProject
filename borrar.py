import pandas as pd
import numpy as np

with open('glove.6B.50d.selected.txt', 'r') as f:
	words = set()
	word_to_vec_map = {}
	words.add('_unk')
	word_to_vec_map['_unk'] = np.zeros((50))
	cont = 0
	for line in f:
		line = line.strip().split()
		curr_word = line[0]
		words.add(curr_word)
		word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
	f.close()