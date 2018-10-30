import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import scipy.stats

# get dates in common
def get_dates(days, dates):
	fechas = [x for x in days if x in dates]
	for day in days:
		if day in dates: fechas.append(day)
	return fechas

def mani(confidence):
	df = pd.read_csv('S&P500.csv')
	df2 = pd.read_csv('variationsImmediately.csv')
	df.index = df.Date.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))

	model = LinearRegression()

	eqs = set(list(df2['related to']))
	output = []
	for eq in eqs:
		dates = df2.loc[df2['related to'] == eq, ['date', 'variation']]
		dates.index = dates.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
		fechas = get_dates(dates.index, df.index)
		X = df.loc[fechas, 'variation'].values
		y = dates.loc[fechas, 'variation'].values
		X = np.array(X, dtype=np.float64).reshape(-1, 1)
		y = np.array(y, dtype=np.float64).reshape(-1, 1)
		model.fit(X, y)
		# confidence = 0.6
		std = np.std(y)
		for day in fechas:
			x = df.loc[day, 'variation'].reshape(-1, 1)
			y = dates.loc[day, 'variation']
			pred = model.predict(x)
			ci = scipy.stats.norm.interval(confidence, loc=pred, scale=std)
			ci2 = scipy.stats.norm.interval(0.7, loc=pred, scale=std)
			if(y > ci[1]): 
				out = 0
			elif(y < ci[0]): 
				out = 2
			else: 
				out = 1
			output.append([day, out, eq])

	df3 = pd.DataFrame(output)
	df3.columns = ['date', 'classe', 'related to']
	df3.index = np.arange(len(output))
	return df3
	

if __name__ == '__main__':
	df = mani(0.6)
	df.to_csv('clases.csv')