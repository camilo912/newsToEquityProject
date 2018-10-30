import numpy as np
import pandas as pd
from subprocess import call
import csv
import cambiar_salidas

from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import hp
from hyperopt import STATUS_OK

from timeit import default_timer as timer


def objective(params):
	
	# Keep track of evals
	global ITERATION
	
	ITERATION += 1
	print(ITERATION, params)

	# Make sure parameters that need to be float are float
	for parameter_name in ['confidence']:
		params[parameter_name] = float(params[parameter_name])
	
	out_file = 'gbm_trials_confidence.csv'

	start = timer()

	cambiar_salidas.mani(params['confidence'])
	call(['python', 'trainHeadlines_parameters_gpu.py', '-m', '13'])
	df = pd.read_csv('bests.txt', index_col=-1, header=-1)
	loss = list(df[0])[-1]
	# print(df)
	# print(df.columns)
	# print(loss)
	
	run_time = timer() - start

	# Write to the csv file ('a' means append)
	of_connection = open(out_file, 'a')
	writer = csv.writer(of_connection)
	writer.writerow([loss, params, ITERATION, run_time])
	of_connection.close()

	# Dictionary with information for evaluation
	return {'loss': loss, 'params': params, 'iteration': ITERATION,
			'train_time': run_time, 'status': STATUS_OK}

def bayes_optimization():
	global ITERATION, MAX_EVALS
	ITERATION = 0

	space = {'confidence': hp.uniform('confidence', 0.3, 0.7)}

	# Keep track of results
	bayes_trials = Trials()

	# File to save first results
	out_file = 'gbm_trials_confidence.csv'
	of_connection = open(out_file, 'w')
	writer = csv.writer(of_connection)

	# Write the headers to the file
	writer.writerow(['no-score', 'params', 'iteration', 'train_time'])
	of_connection.close()

	# Run optimization
	best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))

	# store best results
	of_connection = open('bests_confidence.txt', 'a')
	writer = csv.writer(of_connection)
	bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
	#print(bayes_trials_results[:2])
	writer.writerow([bayes_trials_results[0]['loss'], bayes_trials_results[0]['params']['confidence'], MAX_EVALS])
	of_connection.close()

	return best


MAX_EVALS = 1000
best = bayes_optimization()
print(best)
