import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import scipy.stats

# get dates in common
def get_common(days, dates):
	"""
		Función que obtiene cosas en común entre dos listados, en este caso, fechas en común

		Parámetros:
		- days -- Arreglo de numpy, primer arreglo con fechas
		- dates -- Arreglo de numpy, segundo arreglo con fechas

		Retorna
		- fechas -- Lista, lista con todas las fechas en común entre los dos arreglos de fechas.
	"""
	fechas = [x for x in days if x in dates]
	return fechas

def main(confidence):
	"""
		Main del script, función que según una confianza cálcula las clases para cada ejemplo, teniendo en cuenta un intervalo de confianza.
		Para calcular la clase lo que hace es que entrena un modelo de regresion lineal (CAPM), con la prediccion de este modelo se saca un intervalo de confianza con
		media la´predicción y varianza la varianza de las observaciones. Luego se toma la observación especifica y si la observación está en este intervalo se considera de la clase
		normal, si está por encima se considera de la clase positiva y si está por debajo se considera de la clase negativa.

		Parámetros:
		- confidence -- Flotante, nivel de confinza para calcular el intervalo

		Retorna:
		- df3 -- DataFrame de pandas, dataframe que tiene los datos de fecha, clase a la que pertenece el ejemplo y a que equity está relacionado el ejemplo
	"""
	df = pd.read_csv('S&P500.csv')
	df2 = pd.read_csv('variationsImmediately.csv')
	df.index = df.Date.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))

	model = LinearRegression()

	eqs = set(list(df2['related to']))
	output = []
	for eq in eqs:
		dates = df2.loc[df2['related to'] == eq, ['date', 'variation']]
		dates.index = dates.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
		fechas = get_common(dates.index, df.index)
		X = df.loc[fechas, 'variation'].values
		y = dates.loc[fechas, 'variation'].values
		X = np.array(X, dtype=np.float64).reshape(-1, 1)
		y = np.array(y, dtype=np.float64).reshape(-1, 1)

		model.fit(X, y)
		std = np.std(y)
		for day in fechas:
			x = df.loc[day, 'variation'].reshape(-1, 1)
			y = dates.loc[day, 'variation']
			pred = model.predict(x)
			ci = scipy.stats.norm.interval(confidence, loc=pred, scale=std)
			# ci2 = scipy.stats.norm.interval(0.7, loc=pred, scale=std)
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
	df = main(0.6)
	df.to_csv('clases.csv')