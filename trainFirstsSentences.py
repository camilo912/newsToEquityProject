import pandas as pd
import numpy as np
import time
import models
import argparse
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import hp
from hyperopt import STATUS_OK

from timeit import default_timer as timer

import numba

# eliminate warning from pandas
# pd.options.mode.chained_assignment = None  # default='warn'

class DataManager():
	def __init__(self):
		# Load data
		self.df = pd.read_csv('data14Glove.csv', error_bad_lines=False)
		# self.df = pd.read_csv('data14Glove.csv', error_bad_lines=False)
		#self.df = pd.read_csv('data14Deps.csv', error_bad_lines=False)
		self.df = self.df.sample(frac=1.0).reset_index(drop=True)
		mini = min(self.df.groupby('classes').count()['date'].values)
		self.df = self.df.groupby('classes').head(mini)[['date', 'content', 'classes', 'related to']]
		self.df.index = np.arange(self.df.shape[0])
		self.df = self.df.sample(frac=1.0).reset_index(drop=True)
		for i in range(self.df.shape[0]):
			if(type(self.df.loc[i, 'content']) == float):
				print(i, self.df.loc[i, 'content'])
		self.df['content'] = self.df.content.apply(lambda x: x.strip())

		# Construct vocabulary
		self.words = Counter()
		self.max_len = 0
		for sent in self.df.content.values:
			self.words.update(w.lower() for w in sent.split())
			self.max_len = max(len(sent.split()), self.max_len)
		# Sort vocabulary by frecuency
		self.words = sorted(self.words, key=self.words.get, reverse=True)
		# Add special tokens to vocabulary
		# self.words = ['_PAD','_UNK'] + self.words
		
		# Construct dicts
		self.word2idx = {o:i for i,o in enumerate(self.words)}
		self.idx2word = {i:o for i,o in enumerate(self.words)}

		# Index content
		self.df['contentidx'] = self.df.content.apply(self.indexer)

		# embedd vector map
		self.word2embedd = self.read_embedd_vectors(0)

	def get_data(self):
		return self.df, self.words, self.max_len, self.idx2word, self.word2embedd

	def indexer(self, s): 
		return [self.word2idx[w.lower()] for w in s.split()]

	def read_embedd_vectors(self, embedd):
		if(embedd == 0):
			# with open('glove.6B.50d.txt', 'r', encoding = "utf8") as f:
			with open('glove.6B.50d.selected.txt', 'r', encoding='utf8') as f:
				words = set()
				word_to_vec_map = {}
				words.add('_unk')
				word_to_vec_map['_unk'] = np.zeros((50))
				for line in f:
					line = line.strip().split()
					curr_word = line[0]
					words.add(curr_word)
					word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
				f.close()

		elif(embedd == 1):
			with open('deps.txt', 'r', encoding = "utf8") as f:
				words = set()
				word_to_vec_map = {}
				words.add('_unk')
				word_to_vec_map['_unk'] = np.zeros((300))
				for line in f:
					line = line.strip().split()
					curr_word = line[0]
					words.add(curr_word)
					word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
				f.close()

		return word_to_vec_map

def get_embedd_dic(idx2word, word2embedd):
	dic = []
	for i in idx2word.keys():
		#if(i > 1):
		dic.append(word2embedd[idx2word[i]])
	dic =  np.array(dic, dtype=np.float32)
	return dic

def embedd(idxs, default, new2, embedd_dic):
	embedd_dic2 = embedd_dic.copy()
	new = new2.copy()
	cont = 0
	for i in idxs:
		if(i > 1):
			new[cont] = embedd_dic2[i-2]
		else:
			new[cont] = default
		cont += 1
	return new

@numba.jit(nopython=True)
def embedd_gpu(idxs, default, new2, embedd_dic):
	embedd_dic2 = embedd_dic.copy()
	new = new2.copy()
	cont = 0
	for i in idxs:
		if(i > 1):
			new[cont] = embedd_dic2[i-2]
		else:
			new[cont] = default
		cont += 1
	return new

def media_movil(data, n):
	return np.mean(data[-n:])

class VectorizeData(Dataset):
	def __init__(self, df, max_len, embedding_dim, embedd_dic):
		# global embedding_dim, embedd_dic
		self.df, self.maxlen = df, max_len
		self.df.loc[:, 'lengths'] = self.df.contentidx.apply(lambda x: self.maxlen if len(x) > self.maxlen else len(x))
		self.df.loc[:, 'contentpadded'] = self.df.contentidx.apply(self.pad_data)
		if(torch.cuda.is_available()):
			self.df.loc[:, 'emb'] = self.df.contentpadded.apply((lambda x: embedd_gpu(x, np.zeros([embedding_dim], dtype=np.float32), np.zeros([len(x), embedding_dim], dtype=np.float32), embedd_dic)))
		else:
			self.df.loc[:, 'emb'] = self.df.contentpadded.apply((lambda x: embedd(x, np.zeros([embedding_dim], dtype=np.float32), np.zeros([len(x), embedding_dim], dtype=np.float32), embedd_dic)))
		self.df.index = np.arange(self.df.shape[0])
		
	def __len__(self):
		return self.df.shape[0]
	
	def __getitem__(self, idx):
		related = self.df.loc[idx, 'related to']
		content = self.df.content[idx]
		X = self.df.emb[idx]
		lens = self.df.lengths[idx]
		y = self.df.classes[idx]
		return X, y, lens, content, related
	
	def pad_data(self, s):
		padded = np.zeros((self.maxlen,), dtype=np.int64)
		if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
		else: padded[:len(s)] = s
		return padded

def sort_batch(X, y, lengths):
	lengths, indx = lengths.sort(dim=0, descending=True)
	X = X[indx]
	y = y[indx]
	return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)

def split_uniformly(df, train_size):
	trdfc = []
	tedfc = []
	for c in np.unique(df.classes):
		dfc = df[df['classes'] == c]
		wall = int(dfc.shape[0]*train_size)
		trdfc.append(dfc[:wall])
		tedfc.append(dfc[wall:])

	trdf = pd.DataFrame()
	tedf = pd.DataFrame()

	for i in range(len(trdfc)):
		trdf = trdf.append(trdfc[i], ignore_index=True)
		tedf = tedf.append(tedfc[i], ignore_index=True)

	# shuffle datasets
	trdf = trdf.sample(frac=1.0).reset_index(drop=True)
	tedf = tedf.sample(frac=1.0).reset_index(drop=True)

	return trdf, tedf


def fit(model, df, loss_fn, opt, n_epochs, max_len, batch_size, train_size, embedding_dim, embedd_dic, verbose, bads):
	df_train, df_test = split_uniformly(df, train_size)

	# prevent size 1 batches in training
	if(df_train.shape[0] % batch_size == 1 or df_test.shape[0] % batch_size == 1):
		batch_size += 1

	trdv = VectorizeData(df_train, max_len, embedding_dim, embedd_dic)
	train_dl = DataLoader(trdv, batch_size=batch_size)
	tevd = VectorizeData(df_test, max_len, embedding_dim, embedd_dic)
	test_dl = DataLoader(tevd, batch_size=batch_size)

	historic_acc = []
	historic_train_acc = []
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# for regularization
	lambda_term = 0.05

	for epoch in range(n_epochs):
		start = time.time()
		start1 = start
		train_loss, train_acc = 0, 0

		y_true_train = list()
		y_pred_train = list()
		total_loss_train = 0
		
		# train
		for X, y, lengths, _, _ in train_dl:
			X, y, lengths = sort_batch(X,y,lengths)
			X = Variable(X).to(device)
			y = Variable(y).long().to(device)
			lengths = lengths.numpy()
			
			opt.zero_grad()
			pred = model(X, lengths, train=True)
			loss = loss_fn(pred, y).to(device)

			# # l1 regularization
			# params = model.parameters()
			# l1_regularization = 0
			# for p in params:
			# 	l1_regularization += torch.norm(p, 1)
			# l1_regularization *= lambda_term
			# loss = loss_fn(pred, y).to(device) + l1_regularization

			loss.backward()

			# # gradient clipping for gradient exploding
			# torch.nn.utils.clip_grad_norm(model.parameters(), 1)

			opt.step()

			pred_idx = torch.max(pred, dim=1)[1]
			
			y_true_train += list(y.cpu().data.numpy())
			y_pred_train += list(pred_idx.cpu().data.numpy())
			total_loss_train += loss.data.item()

		if(verbose > 1): print('time elapsed train: %f' % (time.time() - start1))
		start1 = time.time()

		train_acc = accuracy_score(y_true_train, y_pred_train)
		train_loss = total_loss_train/len(train_dl)

		y_true_test = list()
		y_pred_test = list()
		total_loss_test = 0

		for X,y,lengths, _, _ in test_dl:
			X, y,lengths = sort_batch(X,y,lengths)
			X = Variable(X).to(device)
			y = Variable(y).long().to(device)
			pred = model(X, lengths.numpy(), train=False)
			loss = loss_fn(pred, y)
			pred_idx = torch.max(pred, 1)[1]
			y_true_test += list(y.cpu().data.numpy())
			y_pred_test += list(pred_idx.cpu().data.numpy())
			total_loss_test += loss.data.item()#[0]
		
		test_acc = accuracy_score(y_true_test, y_pred_test)
		test_loss = total_loss_test/len(test_dl)
		if(verbose > 1): print('time elapsed test: %f' % (time.time() - start1))
		start1 = time.time()
		

		# media movil
		historic_acc.append(test_acc)
		historic_train_acc.append(train_acc)
		modified_test_acc = media_movil(historic_acc, min(epoch+1, 10))
		modified_train_acc = media_movil(historic_train_acc, min(epoch+1, 10))
		if(verbose > 1): print('time elapsed M.A.: %f' % (time.time() - start1))
		
		# estabilidad + buen train_acc
		modified_test_acc = (modified_test_acc * 0.5 + (modified_train_acc >= 0.45) * 0.15 + (modified_test_acc > 0.34) * 0.15 + (modified_train_acc >= 0.45 and modified_test_acc > 0.34) * 0.2)

		if(verbose > 0):
			print(' Epoch {}: Train loss: {} acc: {}'.format(epoch + 1, train_loss, train_acc))
			print('test_loss: {} acc: {}'.format(test_loss, test_acc))
			print('modified test acc: {}'.format(modified_test_acc))
			print('time elapsed: %f' % (time.time() - start), end='\n\n')

	
	if(bads):
		onedv = VectorizeData(df_test, max_len, embedding_dim, embedd_dic)
		onedl = DataLoader(onedv, batch_size=1)

		salidas = []
		for X, y, lengths, content, related in onedl:
			X, y, lengths = sort_batch(X,y,lengths)
			X = Variable(X)
			y = Variable(y).long()
			pred = model(X, lengths.numpy(), train=False)
			pred_idx = torch.max(pred, dim=1)[1]
			if(int(pred_idx) != int(y)):
				salidas.append([int(pred_idx), int(y), related[0], content, pred.detach().numpy()])

		dfu = pd.DataFrame(salidas)
		dfu.index = np.arange(len(dfu))
		dfu.columns=['pred', 'true', 'related', 'content', 'probs']
		dfu.to_csv('debug_bad_classifications.csv')


	return modified_test_acc

def objective(params, df, max_len, n_out, embedding_dim, train_size, id_model, embedd_dic, verbose, bads):
	
	# Keep track of evals
	global ITERATION
	
	ITERATION += 1
	if(verbose > 0): print(ITERATION, params)

	# Make sure parameters that need to be integers are integers
	for parameter_name in ['n_hidden', 'batch_size', 'n_epochs']:
		params[parameter_name] = int(params[parameter_name])

	# Make sure parameters that need to be float are float
	for parameter_name in ['lr', 'drop_p']:
		params[parameter_name] = float(params[parameter_name])
	
	out_file = 'gbm_trials.csv'

	start = timer()

	modelos=[models.Model0, models.Model1, models.Model2, models.Model3, models.Model4, models.Model5, models.Model6, models.Model7, models.Model8, models.Model9, models.Model10, models.Model11, models.Model12, models.Model13, models.Model14]
	m = modelos[id_model](embedding_dim, params['n_hidden'], n_out, params['drop_p'])
	opt = optim.Adam(m.parameters(), params['lr'])#, weight_decay=params['weight_decay'])
	acc = fit(m, df, F.nll_loss, opt, params['n_epochs'], max_len, params['batch_size'], train_size, embedding_dim, embedd_dic, verbose, bads)

	# calculate no-score
	no_score = 1 - acc # change from acc to loss. If first raises, second downs.
	run_time = timer() - start

	# Write to the csv file ('a' means append)
	of_connection = open(out_file, 'a')
	writer = csv.writer(of_connection)
	writer.writerow([no_score, params, ITERATION, run_time])
	of_connection.close()

	# Dictionary with information for evaluation
	return {'loss': no_score, 'params': params, 'iteration': ITERATION,
			'train_time': run_time, 'status': STATUS_OK}

def bayes_optimization(MAX_EVALS, n_out, max_len, df, embedding_dim, train_size, id_model, embedd_dic, verbose ,bads):
	global ITERATION
	ITERATION = 0

	# space
	# big
	# space = {'batch_size': hp.quniform('batch_size', 5, 120, 1),
	# 		'drop_p': hp.uniform('drop_p', 0.0, 1.0),
	# 		'lr': hp.uniform('lr', 0.00001, 0.8),
	# 		'n_epochs': hp.quniform('n_epochs', 5, 150, 1),
	# 		'n_hidden': hp.quniform('n_hidden', 5, 100, 1)}
	space = {'batch_size': hp.quniform('batch_size', 68, 100, 1),
			'drop_p': hp.uniform('drop_p', 0.01, 0.4),
			'lr': hp.uniform('lr', 0.001, 0.1),
			'n_epochs': hp.quniform('n_epochs', 95, 130, 1),
			'n_hidden': hp.quniform('n_hidden', 15, 70, 1)}

	# Keep track of results
	bayes_trials = Trials()

	# File to save first results
	out_file = 'gbm_trials.csv'
	of_connection = open(out_file, 'w')
	writer = csv.writer(of_connection)

	# Write the headers to the file
	writer.writerow(['no-score', 'params', 'iteration', 'train_time'])
	of_connection.close()

	# Run optimization
	best = fmin(fn = lambda x: objective(x, df, max_len, n_out, embedding_dim, train_size, id_model, embedd_dic, verbose, bads), space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(np.random.randint(100)))

	# store best results
	of_connection = open('bests.txt', 'a')
	writer = csv.writer(of_connection)
	bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
	writer.writerow([bayes_trials_results[0]['loss'], bayes_trials_results[0]['params']['n_hidden'], bayes_trials_results[0]['params']['batch_size'], bayes_trials_results[0]['params']['n_epochs'], bayes_trials_results[0]['params']['lr'], bayes_trials_results[0]['params']['drop_p'], MAX_EVALS])
	of_connection.close()

	return best

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--model', help='number of model to train with', type=int, required=True)
	parser.add_argument('-p', '--parameters', nargs='?', help='number of iterations for performing parameters optimization with bayes optimization if none no optimization is done', type=int)
	parser.add_argument('-v', '--verbose', help='verbosity level', type=int, default=1)
	# parser.add_argument('-p', '--parameters', action='store_true', help='flag for performing parameters optimization with bayes optimization')
	args = parser.parse_args()
	if(args.model == None):
		raise Exception('Debe ingresar un modelo')

	dm = DataManager()
	df, words, max_len, idx2word, word2embedd = dm.get_data()
	embedd_dic = get_embedd_dic(idx2word, word2embedd)
	n_out = 3
	train_size = 0.8
	embedding_dim = len(word2embedd[list(word2embedd.keys())[0]])
	modelos=[models.Model0, models.Model1, models.Model2, models.Model3, models.Model4, models.Model5, models.Model6, models.Model7, models.Model8, models.Model9, models.Model10, models.Model11, models.Model12, models.Model13, models.Model14]
	id_model = int(args.model)
	verbose = args.verbose
	bads = True

	# # Parameters
	if(type(args.parameters) == int):
		if(args.parameters == 0):
			MAX_EVALS = 1
		else:
			MAX_EVALS = args.parameters
		best = bayes_optimization(MAX_EVALS, n_out, max_len, df, embedding_dim, train_size, id_model, embedd_dic, verbose, False)
		print('best is: ', best)

		batch_size = int(best['batch_size'])
		drop_p = best['drop_p']
		lr = best['lr']
		n_epochs = int(best['n_epochs'])
		n_hidden = int(best['n_hidden'])
		# weight_decay = best['weight_decay']

	else:
		# for old approach
		# batch_size = 106 # 88 # 46 # 85
		# drop_p = 0.5 # 0.8227912969918867 # 0.35966039486334267
		# lr =  0.058859628108637673 # 0.009535409031006019 # 0.00017578637883698605
		# n_epochs = 500
		# n_hidden =  13 # 84 # 77
		# # weight_decay = 0.00005

		# for new approach
		batch_size = 116 # 67
		drop_p = 0.12180530013355763 # 0.5
		lr = 0.015143175534512585 # 0.008102095403861038
		n_epochs = 1#117 # 126
		n_hidden = 46 # 84
		# weight_decay = 0.0005

	m = modelos[args.model](embedding_dim, n_hidden, n_out, drop_p)
	opt = optim.Adam(m.parameters(), lr)#, weight_decay=weight_decay)
	#loss_fn = nn.CrossEntropyLoss()
	loss_fn = F.nll_loss

	fit(m, df, loss_fn, opt, n_epochs, max_len, batch_size, train_size, embedding_dim, embedd_dic, verbose, bads)



