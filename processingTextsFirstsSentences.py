import pandas as pd
import numpy as np
import datetime
import seaborn as sns
from matplotlib import pyplot as plt
from langdetect import detect

def delete_weard_values(data):
	q1 = np.percentile(data, 25)
	q3 = np.percentile(data, 75)
	iqr = q3 - q1
	deletes = []
	for i in range(len(data)):
		if(data[i] < q1 - 1.5*iqr or data[i] > q3 + 1.5*iqr):
			deletes.append(i)
	data = np.delete(data, deletes, 0)
	return data

def classes(variation, qs, mean, desv):
	transformed = (variation - mean) / desv
	if(transformed > qs[-1]):
		return 0 # Very good
	elif(transformed > qs[-2]):
		return 1 # Good
	elif(transformed > qs[-3]):
		return 2 # Normal
	elif(transformed > qs[-4]):
		return 3 # Bad
	else:
		return 4 # So Bad
	

def classes_absolute_variation(variation, mini, maxi, day, eq):
	df = pd.read_csv('clases.csv')
	day = datetime.datetime.strptime(str(day).split(' ')[0], '%Y-%m-%d')
	df = df.loc[df['related to']==eq, ['date', 'classe']]
	df.date = df.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
	cal = df.loc[df.loc[:,'date']==day, 'classe'].values
	if len(cal) < 1: cal = [-1]
	return int(cal[0])


def replace_bad_characters(str):
	import re
	str = re.sub(r'([\t?¿\n*.;,’\r:/&”“"()$#!°\'><_—\[\]+©=‘…\v\b\f€£•´^])', r' ', str)
	return str

def delete_stop_words(str, lang):
	from nltk.corpus import stopwords
	sw = stopwords.words(lang)
	# print(len(sw))
	std = ''
	for w in str.split(' '):
		if not w in sw and not w == '':
			std += w + ' '
	return std[:-1]

def stem_words(str):
	from nltk.stem.porter import PorterStemmer
	stemmer = PorterStemmer()
	#from nltk.stem import SnowballStemmer
	#stemmer = SnowballStemmer('english')
	std = ''
	for w in str.split(' '):
		std += stemmer.stem(w) + ' '
	return std[:-1]

def lemmatize_words(str):
	from nltk.stem.wordnet import WordNetLemmatizer
	lemmatizer = WordNetLemmatizer()
	std = ''
	for w in str.split(' '):
		std += lemmatizer.lemmatize(w) + ' '
	return std[:-1]

def read_embedd_vectors(embedd):
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

def only_glove_words(str, words_in_glove):
	std = ''
	for w in str.split(' '):
		if(w in words_in_glove and w != '-' and w != 'reuter' and w != 'reuters' and w != "–"):
			std += w + ' '
		# else:
		#	std += '_UNK '
	return std[:-1]

def transform_string(str, words_in_glove, lang):
	languages = {'en':'english', 'es':'spanish'}
	lang =languages[lang]
	# Quitar signos
	str = replace_bad_characters(str)
	# Minuscula
	str = str.lower()
	# Stop words y varios espacios
	str = delete_stop_words(str, lang)
	# Stemming
	# str = stem_words(str)
	# lemmatization
	# str = lemmatize_words(str)
	# Delete words who aren't in glove dictioanry
	str = only_glove_words(str, words_in_glove)
	return str

def get_word_to_frecuency(data):
	word_to_frecuency = {}
	for l in data:
		for w in l.split(' '):
			if w in list(word_to_frecuency.keys()):
				word_to_frecuency[w] += 1
			else:
				word_to_frecuency[w] = 0
	return word_to_frecuency

def eliminate_less_frequent_words(df, limit, word_to_frecuency):
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
	import nltk.data
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(content)
	return title + ''.join(sentences[:3])


dfr = pd.read_csv("newsDatabaseComplete14.csv", header=0, index_col=0)
vdf = pd.read_csv('variationsImmediately.csv', header=0, index_col=0)
variations = {}
walls = {}
equitys = []
total = []
words_in_glove = read_embedd_vectors(0) ############# change for different embedding
idxs = []

supported_langs=['en']

for eq in list(set(dfr['related to'].values)):
	equitys.append(eq)
	variations[eq] = []

for i in range(dfr.shape[0]):
	step = vdf[vdf['date'] == dfr.loc[i, 'date'].split(' ')[0]]
	lang = detect(dfr['content'][i])
	if(len(step[step['related to'] == dfr.loc[i, 'related to']]) > 0 and lang in supported_langs):
		tmp = get_raw_data(dfr['title'][i], dfr['content'][i])
		dfr.loc[i, 'content'] = transform_string(tmp, words_in_glove, lang)
		variation = step[step['related to'] == dfr.loc[i, 'related to']]['variation'].values[0]
		variations[dfr.loc[i, 'related to']].append(variation)
		total.append(variation)
		idxs.append(i)

classes_arr = []
means_and_desvs = {}
dfr = dfr.loc[idxs, :]
dfr.index=np.arange(dfr.shape[0])
for eq in equitys:
	#### Implementacion de rango absoluto sin valores atipicos
	variations[eq] = delete_weard_values(variations[eq])
	walls[eq] = [min(variations[eq]), max(variations[eq])]

import calcular_salidas
classes_all = calcular_salidas.mani(0.6893894448599668)

for i in range(dfr.shape[0]):
	eq = dfr.loc[i, 'related to']
	day = datetime.datetime.strptime(dfr.loc[i, 'date'].split(' ')[0], '%Y-%m-%d')
	tmp = classes_all.loc[classes_all['date'] == day, ['classe', 'related to']]
	step = tmp.loc[tmp['related to'] == eq, ['classe']].values
	if(len(step)>0):
		classes_arr.append(int(step[0]))
	else:
		classes_arr.append(-1)
	#print(tmp)

	#print(eq)
	# classes_arr.append(int(tmp.loc[tmp['related to'] == eq, ['classe']].values[0]))

dfr['classes'] = pd.Series(classes_arr, index=dfr.index, dtype=np.int64)


word_to_frecuency = get_word_to_frecuency(dfr['content'])

dfr = eliminate_less_frequent_words(dfr, 5, word_to_frecuency)

# eliminate non-classes examples
dfr['classes'].replace(-1, np.nan, inplace=True)
dfr.dropna(subset=['classes'], inplace=True)

# eliminate empty strings from dataframe
dfr['content'].replace('', np.nan, inplace=True)
dfr.dropna(subset=['content'], inplace=True)
dfr.index = np.arange(dfr.shape[0])


# dfr.to_csv('data14Deps.csv')
dfr.to_csv('data14Glove_noStem.csv') ########### change for different embedding

sns.barplot(x=dfr.classes.unique(),y=dfr.classes.value_counts())
plt.show()

# d = ['.', ';', ',', '’', ':', ' ', '/', '&', '”', '“', '"', "(", ")", "%", "@", "–", "-"]
# for e in dfr['content']:
# 	for c in e:
# 		if(not c.isalpha() and not c.isdigit() and not c in d):
# 			print(c, end='')