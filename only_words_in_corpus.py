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

def main():
	"""
		Main del archivo, este se encarga de leer las palabras en nuestra data, luego lee todos los embedings que se tienen,
		y por cada palabra en la lista de las que se posee el embedding se mira s esa palabra aparece en nuestra data, y si aparece
		escribe ste embeding en otro archivo. Esto se hace para hacer ela rhcivo con los embedding más liviano. En terminos generales
		lo que hace es separar solo los embedding que se usan en un arhcivo aparte para leer de este y no del total de embeddings.
	"""
	# df = pd.read_csv('data14Glove.csv')
	# df = pd.read_csv('data14Glove_noStem.csv')
	df = pd.read_csv('data14Glove_noStem_train.csv')
	dfte = pd.read_csv('data14Glove_noStem_test.csv')
	words = []
	for c in df['content']:
		words.extend(c.lower().split(' '))
		words = list(set(words))
	for c in dfte['content']:
		words.extend(c.lower().split(' '))
		words = list(set(words))
	f2 = open('glove.6B.50d.selected.txt', 'w', encoding='utf8')
	with open('glove.6B.50d.txt', 'r', encoding = "utf8") as f:
		for line in f:
			curr_word = line.strip().split()[0]
			if curr_word in words:
				f2.write(line)
		f.close()
	f2.close()

if __name__ == '__main__':
	main()



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
