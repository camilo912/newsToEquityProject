import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import math

#df = pd.read_csv('data9Deps.csv')
df = pd.read_csv('data14Glove.csv')

###################### implementacion de clases con normalizacion y eso srgun Daniel #############

fig = plt.figure(figsize=(8,5))
ax = sns.barplot(x=df.classes.unique(),y=df.classes.value_counts())
ax.set(xlabel='Labels')
plt.show()


##################### implementacion repartir diferencias #####################

# def get_limits(data):
# 	q1 = np.percentile(data, 25)
# 	q3 = np.percentile(data, 75)
# 	iqr = q3 - q1
# 	deletes = []
# 	for i in range(len(data)):
# 		if(data[i] < q1 - 1.5*iqr or data[i] > q3 + 1.5*iqr):
# 			deletes.append(i)
# 	data = np.delete(data, deletes, 0)
# 	return [min(data), max(data)]

# def get_class(variation, mini, maxi):
# 	inter = (maxi - mini) / 5
# 	if(variation > maxi - inter):
# 		return 4 # muy bueno
# 	elif(variation > maxi - inter*2):
# 		return 3 # bueno
# 	elif(variation > maxi - inter*3):
# 		return 2 # regular
# 	elif(variation > maxi - inter*4):
# 		return 1 # malo
# 	else:
# 		return 0 # muy malo

# dfv = pd.read_csv('variations.csv')
# cocavars = dfv[dfv['related to'] == 'coca cola']
# bitvars = dfv[dfv['related to'] == 'bitcoin']
# # cocalims = [min(cocavars['variation']), max(cocavars['variation'])]
# # bitlims = [min(bitvars['variation']), max(bitvars['variation'])]
# cocalims = get_limits(cocavars.variation.values)
# bitlims = get_limits(bitvars.variation.values)
# dfn = df[['date', 'related to']]

# classes = []

# for i in range(dfn.shape[0]):
# 	sec = dfn.loc[i, 'related to']
# 	date = dfn.loc[i, 'date'].split(' ')[0]
# 	if(sec == 'coca cola'):
# 		row = cocavars[cocavars['date'] == date]
# 		classes.append(get_class(row.variation.values, cocalims[0], cocalims[1]))
# 	elif(sec == 'bitcoin'):
# 		row = bitvars[bitvars['date'] == date]
# 		classes.append(get_class(row.variation.values, bitlims[0], bitlims[1]))
# 	else:
# 		print('idx: %d not recognized: %s' % (i, dfn.loc[i, 'related to']))

# dfn['classe'] = pd.Series(classes, index=dfn.index)
# fig = plt.figure(figsize=(8,5))
# ax = sns.barplot(x=dfn.classe.unique(),y=dfn.classe.value_counts())
# ax.set(xlabel='Labels')
# plt.show()


#################### implementacion vieja ##############################

# max_variation = max(df.variation.values)
# min_variation = min(df.variation.values)
# print(max_variation)
# print(min_variation)

# def Nclasses(variation):
# 	inter = (max_variation - min_variation) / 5
# 	if(variation > max_variation - inter):
# 		return 4 # muy bueno
# 	elif(variation > max_variation - inter*2):
# 		return 3 # bueno
# 	elif(variation > max_variation - inter*3):
# 		return 2 # regular
# 	elif(variation > max_variation - inter*4):
# 		return 1 # malo
# 	else:
# 		return 0 # muy malo

# def classes(variation):
# 		if(math.isnan(variation)):
# 			raise ValueError('Variation is not a number')
# 		if(variation > 0.05):
# 			return 4 # So good
# 		elif(variation > 0.025):
# 			return 3 # Good
# 		elif(variation > 0):
# 			return 2 # Normal
# 		elif(variation > -0.025):
# 			return 1 # Bad
# 		else:
# 			return 0 # So bad

# df['classes'] = df.variation.apply(classes)
