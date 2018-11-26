import pandas as pd
import numpy as np
import datetime
import seaborn as sns
from matplotlib import pyplot as plt
from langdetect import detect

def delete_weard_values(data):
	"""
		Función que elimina los valores atipicos de un conjunto de datos, lo hace mediante el rango intercuartilico
		Todo dato que este 1.5 rangos intercuartilicos por debajo del cuartil 1 o por encima del cuartil 3 es un valor atipico y por l otanto se elimina

		Parámetros:
		- data -- Arrego de numpy | lista, conjunto de datos a los cauels se les va a eliminar los datos atipicos

		Retorna:
		- data -- Arreglo de numpy, conjunto de datos sin valores atipicos
		
	"""
	q1 = np.percentile(data, 25)
	q3 = np.percentile(data, 75)
	iqr = q3 - q1
	deletes = []
	for i in range(len(data)):
		if(data[i] < q1 - 1.5*iqr or data[i] > q3 + 1.5*iqr):
			deletes.append(i)
	data = np.delete(data, deletes, 0)
	return data


def replace_bad_characters(str):
	"""
		Función que elimina los caracteres que no se vana tener en cuenta como caracteres extraños pro ejemplo

		Parámetros:
		- str -- String, string de la noticia 

		Retorna:
		- str -- String, string de la noticia sin los caracteres indeseados
	"""
	import re
	str = re.sub(r'([\t?¿\n*.;,’\r:/&”“"()$#!°\'><_—\[\]+©=‘…\v\b\f€£•´^])', r' ', str)
	return str

def delete_stop_words(str, lang):
	"""
		Función que elimina las palabras de parada de la noticia

		Parámetros:
		- str -- String, string de la noticia 
		- lang -- String, lenguaje en el que está escrito la noticia

		Retorna:
		- [valor] -- String, string de la noticia sin las palabras de parada

	"""
	from nltk.corpus import stopwords
	sw = stopwords.words(lang)
	# print(len(sw))
	std = ''
	for w in str.split(' '):
		if not w in sw and not w == '':
			std += w + ' '
	return std[:-1]

def stem_words(str):
	"""
		Función que convierte las palabras en sus "stems" o raíces

		Parámetros:
		- str -- String, string de la noticia

		Retorna:
		- [valor] -- String, string de la noticia con sus palabras "stemizadas"

	"""

	from nltk.stem.porter import PorterStemmer
	stemmer = PorterStemmer()
	#from nltk.stem import SnowballStemmer
	#stemmer = SnowballStemmer('english')
	std = ''
	for w in str.split(' '):
		std += stemmer.stem(w) + ' '
	return std[:-1]

def lemmatize_words(str):
	"""
		Función que convierte las palabras en su "lemas" o raíces 

		Parámetros:
		- str -- String, string de la noticia

		Retorna:
		- [valor] -- String, string de la noticia con sus palabras "lematizadas"
	"""
	from nltk.stem.wordnet import WordNetLemmatizer
	lemmatizer = WordNetLemmatizer()
	std = ''
	for w in str.split(' '):
		std += lemmatizer.lemmatize(w) + ' '
	return std[:-1]

def read_embedd_vectors(embedd):
	"""
		Función que lee el dataset de embeddings de glove y retorna uan lista con las palabras que hay allí

		Parámetros:
		- embedd -- Entero, indica que ebedding se va a usar 0 si glove, 1 si deps

		Retorna:
		- [valor] Lista, lista con las palabras que hay en el dataset de words embeddings
	"""
	if(embedd == 0):
		with open('glove.6B.50d.txt', 'r', encoding = "utf8") as f:
			words = set()
			word_to_vec_map = {}
			for line in f:
				line = line.strip().split()
				curr_word = line[0]
				words.add(curr_word)
				word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
	elif(embedd == 1):
		with open('deps.txt', 'r', encoding = "utf8") as f:
			words = set()
			word_to_vec_map = {}
			for line in f:
				line = line.strip().split()
				curr_word = line[0]
				words.add(curr_word)
				word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

	return word_to_vec_map.keys()

def only_glove_words(str, words_in_glove, source):
	"""
		Función que elimina las palabras que no estén en el dataset de vectores de embedding de glove

		Parámetros:
		- str -- String, string de la noticia
		- words_in_glove -- Lista, listado de palabras que hay en el dataset glove

		Retorna:
		- [valor] --String, string d ela noticia solo con palabras que estén en el dataset de glove
	"""
	std = ''
	cont=0
	for w in str.split(' '):
		if(w in words_in_glove and w != '-' and w != source[:-1] and w != source and w != "–"):
			std += w + ' '
		# else:
		#	std += '_UNK '
		cont+=1
	return std[:-1]

def get_word_to_frecuency(data):
	"""
		función que mapea cada palabra a la cantidad de veces que aparece en el corpus

		Parámetros:
		- data -- Arreglo de numpy, arreglo con todas las noticias

		Retorna:
		- word_to_frecuency -- Diccionario, diccionario que mapea cada palabra a la cantidad de veces que aparece en el corpus

	"""

	word_to_frecuency = {}
	cont=0
	for l in data:
		for w in l.split(' '):
			if w in list(word_to_frecuency.keys()):
				word_to_frecuency[w] += 1
			else:
				word_to_frecuency[w] = 1
		cont += 1
	return word_to_frecuency

def eliminate_less_frequent_words(df, limit, word_to_frecuency):
	"""
		Función que elimina las palabras menos frecuentes en todo el corpus. Tiene una mejora adicional que elimina las noticias que tengan pocas palabras luego de haber
		eliminado las palabras menos frecuentes.

		Parámetros:
		- df -- DataFrame de pandas, dataframe que contiene los datos a procesar (las noticias)
		- limit -- Entero, número minimo de apariciones en el corpus para que una palabra sea tenida en cuenta
		- word_to_frecuency -- Diccionario, diccionario que mapea cada palabra a la cantidad de veces que aparece en el corpus

		Retorna:
		- df -- DataFrame de pandas, el mismo dataframe de entrada pero sin las palabras menos repetidas y en el caso de que la noticia sea muy corta la elimina
	"""
	for i in range(df.shape[0]):
		content = df.loc[i, 'content']
		new = ''
		for w in content.split(' '):
			if(word_to_frecuency[w] >= limit):
				new += w + ' '
		new = new[:-1]
		if(len(new.split(' ')) < 8): new = ''
		df.loc[i, 'content'] = new
	return df

def get_raw_data(title, content):
	"""
		Función que retorna el titulo y las tres primeras oraciones de una noticia, hace la eparación en oraciones mediante un tokenizer de la ibreria nltk

		Parámetros:
		- title -- String, título de la noticia
		- content -- String, contenido de la noticia

		Retorna:
		- [valor] -- String, título de la noticia concatenado con las tres primeras oraciones.
	"""
	import nltk.data
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(content)
	return title + ''.join(sentences[:5])

def transform_string(cad, words_in_glove, lang, source):
	"""
		Función que preprocesa las noticias, recibe el string de la noticia completa y retorna el string preprocesado

		Parámetros:
		- str -- String, string de la noticia que se va a preprocesar
		- words_in_glove -- Lista, lista de palabras en el dataset de vectores de word embeddings de glove
		- lang -- String, lenguaje en el que está escrita la noticia

		Retorna:
		- str -- String, strng de la noticia preprocesada
	"""
	assert type(source) == str
	languages = {'en':'english', 'es':'spanish'}
	lang =languages[lang]
	# Quitar signos y caracteres que no se van a tener en cuenta
	cad = replace_bad_characters(cad)
	# Minuscula
	cad = cad.lower()
	# Stop words y varios espacios
	cad = delete_stop_words(cad, lang)
	# Stemming
	cad = stem_words(cad)
	# lemmatization
	# cad = lemmatize_words(cad)
	# Delete words that aren't in embedding dictionary
	cad = only_glove_words(cad, words_in_glove, source)
	return cad


def main():
	"""
		Main del archivo, este archivo se encarga de preprocesar las noticias, "limpiar la data" en terminos geenrales. Hace las operaciones que estén definidas en el metodo
		transform_tring.

		Primero ejecuta al funcion transform string, luego calcula las salidas de los ejemplos, luego elimina las palabras menos frecuentes y por ultimo guarda estos ultimos datos.

	"""
	# dfr = pd.read_csv("newsDatabaseComplete14.csv", header=0, index_col=0)
	dfr = pd.read_csv("newsDatabaseComplete14_filtered.csv", header=0, index_col=0)
	# dfr = pd.read_csv("newsDatabaseComplete14_filtered_augmented.csv", header=0, index_col=0)
	#vdf = pd.read_csv('variationsImmediately.csv', header=0, index_col=0)
	#variations = {}
	#walls = {}
	#equitys = []
	#total = []
	words_in_glove = read_embedd_vectors(0) ############# change for different embedding
	#idxs = []

	supported_langs=['en']

	# eliminate non-classes examples
	dfr.dropna(subset=['classes'], inplace=True)
	dfr.index = np.arange(dfr.shape[0])

	# implementation before asking people for classes

	#for eq in list(set(dfr['related to'].values)):
	#	equitys.append(eq)
	#	variations[eq] = []

	# for i in range(dfr.shape[0]):
	# 	print(i)
	# 	step = vdf[vdf['date'] == dfr.loc[i, 'date'].split(' ')[0]]
	# 	lang = detect(dfr['content'][i])
	# 	if(len(step[step['related to'] == dfr.loc[i, 'related to']]) > 0 and lang in supported_langs):
	# 		tmp = get_raw_data(dfr['title'][i], dfr['content'][i])
	# 		dfr.loc[i, 'content'] = transform_string(tmp, words_in_glove, lang)
	# 		if('Standards:' in dfr.loc[i, 'content']): 
	# 			print(dfr.loc[i, 'content'])
	# 			raise Exception('Debug')
	# 		variation = step[step['related to'] == dfr.loc[i, 'related to']]['variation'].values[0]
	# 		variations[dfr.loc[i, 'related to']].append(variation)
	# 		total.append(variation)
	# 		idxs.append(i)

	# implmentation asking people for classes
	for i in range(dfr.shape[0]):
		lang = detect(dfr['content'][i])
		if(lang in supported_langs):
			tmp = get_raw_data(dfr['title'][i], dfr['content'][i])
			dfr.loc[i, 'content'] = transform_string(tmp, words_in_glove, lang, dfr['source'][i])
			#variation = step[step['related to'] == dfr.loc[i, 'related to']]['variation'].values[0]
			#variations[dfr.loc[i, 'related to']].append(variation)
			#total.append(variation)
			#idxs.append(i)
		else:
			print('language: %s not supported. Notice id: %d' % (lang, i))
			dfr.loc[i, 'content'] = ''

	# implementation before asking people for classes
	# classes_arr = []
	# means_and_desvs = {}
	# dfr = dfr.loc[idxs, :]
	# dfr.index=np.arange(dfr.shape[0])
	# for eq in equitys:
	# 	#### Implementacion de rango absoluto sin valores atipicos
	# 	variations[eq] = delete_weard_values(variations[eq])
	# 	walls[eq] = [min(variations[eq]), max(variations[eq])]

	# import calcular_salidas
	# classes_all = calcular_salidas.mani(0.6893894448599668)

	# for i in range(dfr.shape[0]):
	# 	eq = dfr.loc[i, 'related to']
	# 	day = datetime.datetime.strptime(dfr.loc[i, 'date'].split(' ')[0], '%Y-%m-%d')
	# 	tmp = classes_all.loc[classes_all['date'] == day, ['classe', 'related to']]
	# 	step = tmp.loc[tmp['related to'] == eq, ['classe']].values
	# 	if(len(step)>0):
	# 		classes_arr.append(int(step[0]))
	# 	else:
	# 		classes_arr.append(-1)
	# 	#print(tmp)

	# 	#print(eq)
	# 	# classes_arr.append(int(tmp.loc[tmp['related to'] == eq, ['classe']].values[0]))

	# dfr['classes'] = pd.Series(classes_arr, index=dfr.index, dtype=np.int64)

	# # eliminate non-classes examples
	# dfr['classes'].replace(-1, np.nan, inplace=True)

	word_to_frecuency = get_word_to_frecuency(dfr['content'])

	# dfr = eliminate_less_frequent_words(dfr, 5, word_to_frecuency)
	dfr = eliminate_less_frequent_words(dfr, 5*6, word_to_frecuency)

	# eliminate empty strings from dataframe
	dfr['content'].replace('', np.nan, inplace=True)
	dfr.dropna(subset=['content'], inplace=True)
	dfr.index = np.arange(dfr.shape[0])

	# formating problem with pytorch
	dfr['classes'].replace(1, 2, inplace=True)
	dfr['classes'].replace(0, 1, inplace=True)
	dfr['classes'].replace(-1, 0, inplace=True)

	# dfr.to_csv('data14Deps.csv')
	dfr.to_csv('data14Glove_noStem.csv') ########### change for different embedding

	sns.barplot(x=dfr.classes.unique(),y=dfr.classes.value_counts())
	plt.show()

	# d = ['.', ';', ',', '’', ':', ' ', '/', '&', '”', '“', '"', "(", ")", "%", "@", "–", "-"]
	# for e in dfr['content']:
	# 	for c in e:
	# 		if(not c.isalpha() and not c.isdigit() and not c in d):
	# 			print(c, end='')

if __name__ == '__main__':
	main()