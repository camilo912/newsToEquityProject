import pandas as pd
import numpy as np

def main():
	df = pd.read_csv('newsDatabaseComplete14_filtered.csv', header=0, index_col=0)
	import nltk.data
	import itertools
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	permutations = list(itertools.permutations([0, 1, 2, 3])) # [[0,2,1,3], [3,2,1,0], [0,1,3,2], [0,3,1,2], [1,0,2,3]]
	df.dropna(subset=['classes'], inplace=True)
	df.index = np.arange(df.shape[0])
	for i in range(len(df)):
		sentences = np.array(tokenizer.tokenize(df.loc[i, 'content']))[:len(permutations[0])]
		if(len(sentences) == 4):
			for j in range(len(permutations)):
				df = df.append(pd.DataFrame([[df.loc[i, 'classes'], sentences[permutations[j]], df.loc[i, 'date'], df.loc[i, 'related to'], df.loc[i, 'source'], df.loc[i, 'title']]], columns=df.columns), ignore_index=True)


	df.to_csv('newsDatabaseComplete14_filtered_augmented.csv')

def augment_data(df):
	"""
		Función para realizar daat augmentación de noticias, lo que hace es cambiar el orden de las oraciones

		Parámetros:
		- df -- DataFrame de pandas, dataframe que contiene las noticias

		Retorna:
		- df -- DataFrame de pandas, dataframe con las noticias originales y las "aumentadas"
		- [valor] -- Entero, número de "aumentaciones" que se hicieron a cada noticia

	"""
	import nltk.data
	import itertools
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	permutations = [[0,2,1,3], [3,2,1,0], [0,1,3,2], [0,3,1,2], [1,0,2,3]] # list(itertools.permutations([0, 1, 2, 3])) # [[0,2,1,3], [3,2,1,0], [0,1,3,2], [0,3,1,2], [1,0,2,3]]
	df.dropna(subset=['classes'], inplace=True)
	df.index = np.arange(df.shape[0])
	for i in range(len(df)):
		sentences = np.array(tokenizer.tokenize(df.loc[i, 'content']))[:len(permutations[0])]
		if(len(sentences) == 4):
			for j in range(len(permutations)):
				df = df.append(pd.DataFrame([[df.loc[i, 'classes'], ''.join(sentences[list(permutations[j])]), df.loc[i, 'date'], df.loc[i, 'related to'], df.loc[i, 'source'], df.loc[i, 'title']]], columns=df.columns), ignore_index=True)
	return df, len(permutations)


if __name__ == '__main__':
	main()