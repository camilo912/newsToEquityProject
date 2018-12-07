import numpy as np
import pandas as pd

def main():
	import trainFirstsSentences
	import models
	import torch.optim as optim
	import torch.nn.functional as F

	dm = trainFirstsSentences.DataManager() # se crea un gestionador de los datos. Esta clase se encarga de leer, normalizar para que cada clase tenga el mismo número de ejemplos, 
					   # sacar el vocabulario y los word embedding de estas palabras
	df, dfte, words, max_len, idx2word, word2embedd = dm.get_data() # se obtienen los datos que procesa el gestonador de datos. Los cuales son:
																	# - df 			---> 	dataframe de entrenamiento
																	# - dfte 		---> 	dataframe de prueba
																	# - words		--->	arreglo con las palabras del vocabulario
																	# - max_len		--->	maxima longitud de un ejemplo (cantidad de palabras)
																	# - idx2word 	--->	diccionario que mapea de indices a paabras (otorga un indice a cada palabra, ordenado alfabeticamente)
																	# - word2embedd --->	diccionario que mapea de una palabra a su respectivo word embedding

	embedd_dic = trainFirstsSentences.get_embedd_dic(idx2word, word2embedd) # se obtiene un vector de embedding, es decir se pasa de tener dos diccioanrios uno de indices a palabras y otro de palabras 
																			# a embeddings, a tener solamente un arreglo que mapea de indices a embeddings.

	n_out = 3 # número de clases en las cuales se puede clasificar una noticia
	train_size = 0.8 # tamaño del set de entrenamiento con respecto al total de ejemplo, en este caso 80%
	embedding_dim = len(word2embedd[next(iter(word2embedd))]) # obtener la dimensión del embedding

	# Se hace la lista de modelos que podrían ser usados para luego seleccionar uno
	modelos=[models.Model0, models.Model1, models.Model2, models.Model3, models.Model4, models.Model5, models.Model6, models.Model7, models.Model8, models.Model9, models.Model10, models.Model11, models.Model12, models.Model13, models.Model14, models.Model15, models.Model16]

	id_model = 13 # id del modelo a utilizar
	verbose = 1 # nivel de verbosidad de la ejecución, en este caso 1 para que muestre los resultados de cada *epoch* de entrenamiento
	bads = False  # indica si se quieren sacar en una rchivo aparte los ejemplos que no está clasificacndo bien para analizarlos manualmente, en este caso no.
	parameters = None # variable para hacer optimización de parámetros, si se coloca algún entero se hará ese número de iteraciones de buscar mejores parámetros.

	if(type(parameters) == int):
		if(parameters <= 0):
			MAX_EVALS = 1
		else:
			MAX_EVALS = parameters
		best = trainFirstsSentences.bayes_optimization(MAX_EVALS, n_out, max_len, df, dfte, embedding_dim, train_size, id_model, embedd_dic, verbose, False) # este último False es el parámetro bads
																																							 # por default es False para la optimización
		print('best is: ', best)

		batch_size = int(best['batch_size'])
		drop_p = best['drop_p']
		lr = best['lr']
		n_epochs = int(best['n_epochs'])
		n_hidden = int(best['n_hidden'])
		weight_decay = best['weight_decay']

	else:
		# en este caso utilizamos unos parámetros por default
		batch_size = 1000
		drop_p = 0.5 
		lr = 0.001 
		n_epochs = 40 
		n_hidden = 20 
		weight_decay = 0.0005

	m = modelos[id_model](embedding_dim, n_hidden, n_out, drop_p) # se crea el modelo con el que se va a entrenar
	opt = optim.Adam(m.parameters(), lr, weight_decay=weight_decay) # se crea el optimizador que va a optimizar el modelo
	loss_fn = F.nll_loss # se crea la función de perdida del modelo
	trainFirstsSentences.fit(m, df, dfte, loss_fn, opt, n_epochs, max_len, batch_size, train_size, embedding_dim, embedd_dic, verbose, bads) # se invoca la función fit que es la encargada de entrenar el modelo

	# se muestran 4 lineas por cada iteración:
	# 1: muestra el número de epoch, la pérdida de entrenamiento y la exactitud del entrenamiento en % de aciertos
	# 2: muesta la pérdida de prueba y la exactitud de prueba en % de aciertos
	# 3: muestra una exactitud de prueba modificada, esto solo sirve para la optimización. Lo que hace es juntar varias metricas para tener una sola conjunta. Junta el error de entrenamiento y prueba
	# 4: muestra el tiempo que transcurrió para realizar este *epoch*

if __name__ == '__main__':
	main()