import pandas as pd
import numpy as np

def read_embedd_vectors():
	with open('glove.6B.50d.txt', 'r', encoding = "utf8") as f:
		words = set()
		word_to_vec_map = {}
		words.add('_unk')
		word_to_vec_map['_unk'] = np.zeros((50))
		for line in f:
			line = line.strip().split()
			curr_word = line[0]
			words.add(curr_word)
			word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
		f.close()
	return word_to_vec_map

df = pd.read_csv('data14Glove.csv')
words = []
for c in df['content']:
	words.extend(c.split(' '))
	words = list(set(words))

f2 = open('glove.6B.50d.selected.txt', 'w', encoding='utf8')
with open('glove.6B.50d.txt', 'r', encoding = "utf8") as f:
	for line in f:
		curr_word = line.strip().split()[0]
		if curr_word in words:
			f2.write(line)
	f.close()
f2.close()





# word_to_vec_map = read_embedd_vectors()
# lines = []
# for w in words:
# 	if(w in word_to_vec_map.keys()):
# 		line = list(word_to_vec_map[w])
# 		line = [w] + line
# 		lines.append(line)

# f = open('glove.6B.50d.selected.txt', 'w')
# for l in lines:
# 	f.write((' '.join([str(x) for x in l]) + '\n').encode('utf8').decode('utf8'))
# f.close()
