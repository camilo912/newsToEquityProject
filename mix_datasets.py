import pandas as pd
import numpy as np

def join_classes(classe):
	classe = int(classe)
	if(classe <= 3):
		return -1
	elif(classe <= 6):
		return 0
	elif(classe <= 9):
		return 1
	else:
		raise Exception('clase desconocida: ', classe)

def change_date_format(date):
	############# falta ponerle 0 a los días y meses menores de 10
	parts = date.split('/')
	if(int(parts[2]) < 25):
		parts[2] = '20' + parts[2]
	else:
		parts[2] = '19' + parts[2]

	if(int(parts[0]) < 10):
		parts[0] = '0' + parts[0]

	if(int(parts[1]) < 10):
		parts[1] = '0' + parts[1]

	return parts[2] + '-' + parts[0] + '-' + parts[1] + ' 12:00:00'

def eliminate_non_utf8_characters(cadena):
	# return string.encode("utf-8", 'ignore').decode("utf-8", 'ignore').encode('utf-8').decode('utf-8')
	import string

	return ''.join(x for x in cadena if x in string.printable)


df = pd.read_csv('Full-Economic-News-DFE-839861.csv', encoding='latin-1')
df = df[['positivity', 'date', 'headline', 'text']]
df.rename(columns={'positivity':'classes', 'headline':'title', 'text':'content'}, inplace=True)
df_original = pd.read_csv("newsDatabaseComplete14_filtered.csv", header=0, index_col=0)

# eliminate nan values
df.dropna(subset=['classes'], inplace=True)
df.index = np.arange(df.shape[0])

# preprocess
df['content'] = df.content.apply(eliminate_non_utf8_characters)
df['title'] = df.title.apply(eliminate_non_utf8_characters)
df['date'] = df.date.apply(change_date_format)
df['classes'] = df.classes.apply(join_classes)
df.insert(1, 'related to', 'npi', True)
df.insert(1, 'source', 'npi', True)


df = df[df_original.columns]
#print(df.columns)
#print(df_original.columns)

df_total = df_original.append(df, ignore_index=True)

df_total.dropna(subset=['classes'], inplace=True)
df_total.index = np.arange(df_total.shape[0])

# print(df_total['date'])
df.to_csv('newsDatabaseComplete14_filtered_mixed.csv')