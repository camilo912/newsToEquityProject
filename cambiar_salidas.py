import pandas as pd
import numpy as np
import calcular_salidas
import datetime
from matplotlib import pyplot as plt

def mani(confidence):
	df = pd.read_csv('data14Glove_noStem_classes.csv', index_col=0)
	classes = calcular_salidas.mani(confidence)

	for i in range(len(df)):
		day, eq = df.loc[i, 'date'], df.loc[i, 'related to']
		day = datetime.datetime.strptime(str(day).split(' ')[0], '%Y-%m-%d')
		step = classes.loc[classes['date']==day, ['related to', 'classe']]
		classe = step.loc[step['related to']==eq, 'classe'].values
		if len(classe) < 1: classe = [-1]
		df.loc[i, 'classes'] = classe[0]

	# eliminate non-classes examples
	df['classes'].replace(-1, np.nan, inplace=True)
	df.dropna(subset=['classes'], inplace=True)
	df.to_csv('data14Glove_noStem_classes.csv')

	# df1 = pd.read_csv('data14Glove_noStem_classes.csv')
	# data1 = df1['classes'].values

	# df2 = pd.read_csv('data14Glove_noStem.csv')
	# data2 = df2['classes'].values

	# plt.plot(data1, color='r')
	# plt.plot(data2, color='b')
	# plt.show()

if __name__ == '__main__':
	mani(0.6893894448599668)