import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

"""
	Clase que contiene todos los mmodelos qu se puden utilizar par entrenar, se pueden agregar y quitar modelos, teniendo en cuenta qu se debe actualizar también la lista de modelos
	en el main que está en el archivo trainFirstsSentences.py

	No se va a documentar cada modelo por que sería muy repetitivo y todos comparten estructuras en común además de que se pueden modificar mucho.

	Todos estos modelos heredan de la clase torch.nn.Module de pytorch 
"""


# Model with: dropout, gru, concat-pooling, linear, tanh, batchnorm, linear(out=h)
class Model0(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.rnn = nn.GRU(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.lin = nn.Linear(self.n_hidden*3, self.n_hidden*2).to(self.device)
		self.bn = nn.BatchNorm1d(self.n_hidden*2).to(self.device)
		self.out = nn.Linear(self.n_hidden*2, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		self.h = self.init_hidden(seq.size(1), gpu)
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs) # Dropout
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs, self.h)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		avg_pool = F.adaptive_avg_pool1d(rnn_out.permute(1,2,0),1).view(seq.size(1),-1)
		max_pool = F.adaptive_max_pool1d(rnn_out.permute(1,2,0),1).view(seq.size(1),-1)
		ln = self.lin(torch.cat([self.h[-1],avg_pool,max_pool],dim=1))
		ln = torch.tanh(ln).to(self.device)
		bn = self.bn(ln)
		outp = self.out(bn)
		return F.log_softmax(outp, dim=-1)

	def init_hidden(self, batch_size):
		return torch.Tensor(torch.zeros((self.n_layers, batch_size, self.n_hidden))).to(self.device)

# Model with: dropout, gru, (linear only for reduction), tanh, linear(out=out)
class Model1(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.rnn = nn.GRU(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		self.h = self.init_hidden(seq.size(1), gpu)
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs, self.h)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		inter_layer = nn.Linear(max(lengths), 1).to(self.device)
		inter = inter_layer(rnn_out.transpose(0,2)).view(self.h[-1].shape)
		inter = torch.tanh(inter).to(self.device)
		outp = self.out(inter)
		return F.log_softmax(outp, dim=-1)

	def init_hidden(self, batch_size):
		return torch.Tensor(torch.zeros((self.n_layers, batch_size, self.n_hidden))).to(self.device)

# Model with: dropout, gru*2(dropout), (linear only for reduction), tanh, linear(out=out)
class Model2(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 2
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.rnn = nn.GRU(self.embedding_dim, self.n_hidden, num_layers=self.n_layers, dropout=drop_p).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		self.h = self.init_hidden(seq.size(1))
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs, self.h)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		inter_layer = nn.Linear(max(lengths), 1).to(self.device)
		inter = inter_layer(rnn_out.transpose(0,2)).view(self.h[-1].shape)
		inter = torch.tanh(inter).to(self.device)
		outp = self.out(inter)
		return F.log_softmax(outp, dim=-1)

	def init_hidden(self, batch_size):
		return torch.Tensor(torch.zeros((self.n_layers, batch_size, self.n_hidden))).to(self.device)

# Model with: gru, concat-pooling, linear, tanh, batchnorm, linear(out=h)
class Model3(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.rnn = nn.GRU(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.lin = nn.Linear(self.n_hidden*3, self.n_hidden*2).to(self.device)
		self.bn = nn.BatchNorm1d(self.n_hidden*2).to(self.device)
		self.out = nn.Linear(self.n_hidden*2, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		self.h = self.init_hidden(seq.size(1), gpu)
		embs = torch.tensor(seq).float().to(self.device)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs, self.h)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		avg_pool = F.adaptive_avg_pool1d(rnn_out.permute(1,2,0),1).view(seq.size(1),-1)
		max_pool = F.adaptive_max_pool1d(rnn_out.permute(1,2,0),1).view(seq.size(1),-1)
		ln = self.lin(torch.cat([self.h[-1],avg_pool,max_pool],dim=1))
		ln = torch.tanh(ln)
		bn = self.bn(ln)
		outp = self.out(bn)
		return F.log_softmax(outp, dim=-1)

	def init_hidden(self, batch_size):
		return torch.Tensor(torch.zeros((self.n_layers, batch_size, self.n_hidden))).to(self.device)

# Model with: dropout, gru, (linear only for reduction), batchnorm, linear(out=out)
class Model4(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.bn = nn.BatchNorm1d(self.n_hidden).to(self.device)
		self.rnn = nn.GRU(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		self.h = self.init_hidden(seq.size(1), gpu)
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs, self.h)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		inter_layer = nn.Linear(max(lengths), 1).to(self.device)
		inter = inter_layer(rnn_out.transpose(0,2)).view(self.h[-1].shape)
		inter = self.bn(inter)
		outp = self.out(inter)
		return F.log_softmax(outp, dim=-1)

	def init_hidden(self, batch_size):
		return torch.Tensor(torch.zeros((self.n_layers, batch_size, self.n_hidden))).to(self.device)

# Model with: dropout, lstm, (linear only for reduction), batchnorm, linear(out=out)
class Model5(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.bn = nn.BatchNorm1d(self.n_hidden).to(self.device)
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		# self.h = self.init_hidden(seq.size(1), gpu)
		#print(self.h.shape)
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)#self.rnn(embs, self.h)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		inter_layer = nn.Linear(max(lengths), 1).to(self.device)
		inter = inter_layer(rnn_out.transpose(0,2)).view(self.h[-1].shape[1:3])
		inter = self.bn(inter)
		outp = self.out(inter)
		return F.log_softmax(outp, dim=-1)

# Model with: lstm, cocat-pooling, (linear only for reduction), linear, tanh, batchnorm, linear(out=out)
class Model6(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.lin = nn.Linear(self.n_hidden*3, self.n_hidden*2).to(self.device)
		self.bn = nn.BatchNorm1d(self.n_hidden*2).to(self.device)
		self.out = nn.Linear(self.n_hidden*2, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		#self.h = self.init_hidden(seq.size(1), gpu)
		embs = torch.tensor(seq).float().to(self.device)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		avg_pool = F.adaptive_avg_pool1d(rnn_out.permute(1,2,0),1).view(seq.size(1),-1)
		max_pool = F.adaptive_max_pool1d(rnn_out.permute(1,2,0),1).view(seq.size(1),-1)
		inter_layer = nn.Linear(max(lengths), 1).to(self.device)
		inter = inter_layer(rnn_out.transpose(0,2)).view(self.h[-1].shape[1:3])
		ln = self.lin(torch.cat([inter,avg_pool,max_pool],dim=1))
		ln = torch.tanh(ln).to(self.device)
		bn = self.bn(ln)
		outp = self.out(bn)
		return F.log_softmax(outp, dim=-1)

# Model with: dropout, lstm, batchnorm, linear(out=h)
class Model7(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.bn = nn.BatchNorm1d(self.n_hidden).to(self.device)
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		# self.h = self.init_hidden(seq.size(1), gpu)
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)#self.rnn(embs, self.h)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		#inter_layer = nn.Linear(self.n_hidden, self.n_hidden)
		#inter = inter_layer(self.h[-1]).view(self.h[-1].shape[1:3])
		inter = self.bn(self.h[-1].view(self.h[-1].shape[1:3]))
		outp = self.out(inter)
		return F.log_softmax(outp, dim=-1)

# Model with: dropout, lstm*5(dropout), batchnorm, linear(out=h)
class Model8(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 5
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.bn = nn.BatchNorm1d(self.n_hidden).to(self.device)
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers, dropout=drop_p).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		inter = self.bn(self.h[-1][-1].view(self.h[-1].shape[1:3]))
		outp = self.out(inter)
		return F.log_softmax(outp, dim=-1)

# Model with: lstm, (linear only for reduction), tanh, linear(out=out)
class Model9(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		embs = torch.tensor(seq).float().to(self.device)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		inter_layer = nn.Linear(max(lengths), 1).to(self.device)
		inter = inter_layer(rnn_out.transpose(0,2)).view(self.h[-1].shape[1:3])
		inter = torch.tanh(inter).to(self.device)
		outp = self.out(inter)
		return F.log_softmax(outp, dim=-1)

# Model with: lstm, (linear only for reduction), linear(out=out)
class Model10(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		embs = torch.tensor(seq).float().to(self.device)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		inter_layer = nn.Linear(max(lengths), 1).to(self.device)
		inter = inter_layer(rnn_out.transpose(0,2)).view(self.h[-1].shape[1:3])
		outp = self.out(inter)
		return F.log_softmax(outp, dim=-1)

# Model with: lstm, linear(out=h)
class Model11(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		# self.h = self.init_hidden(seq.size(1), gpu)
		embs = torch.tensor(seq).float().to(self.device)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)#self.rnn(embs, self.h)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		outp = self.out(self.h[-1].view(self.h[-1].shape[1:3]))
		return F.log_softmax(outp, dim=-1)

# Model with: dropout, lstm*5(dropout), (linear only for reduction), batchnorm, linear(out=out)
class Model12(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 5
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		self.bn = nn.BatchNorm1d(self.n_hidden).to(self.device)
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers, dropout=drop_p).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		inter_layer = nn.Linear(max(lengths), 1).to(self.device)
		inter = inter_layer(rnn_out.transpose(0,2)).view(self.h[-1].shape[1:3])
		inter = self.bn(inter)
		outp = self.out(inter)
		return F.log_softmax(outp, dim=-1)

# Model with: dropout, lstm, linear(out=h)
class Model13(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		# self.bn = nn.BatchNorm1d(self.n_hidden).to(self.device)
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		#if train:
		#	outp = self.out(self.drop(self.h[-1]).view(self.h[-1].shape[1:3]))
		#else:
		#	outp = self.out(self.h[-1].view(self.h[-1].shape[1:3]))
		#print(len(self.h))
		#raise Exception('Debug')
		#outp = self.out(self.h[-1].view(self.h[-1].shape[1:3]))
		outp = self.out(self.h[-2].view(self.h[-2].shape[1:3]))
		return F.log_softmax(outp, dim=-1)

# Model with: lstm*2(dropout), linear(out=h)
class Model14(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 2
		# self.drop = nn.Dropout(p=drop_p).to(self.device)
		# self.bn = nn.BatchNorm1d(self.n_hidden).to(self.device)
		self.rnn = nn.LSTM(self.embedding_dim, self.n_hidden, num_layers=self.n_layers, dropout=drop_p).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		embs = torch.tensor(seq).float().to(self.device)
		#if train:
		#	embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		outp = self.out(self.h[-1][-1].view(self.h[-1].shape[1:3]))
		return F.log_softmax(outp, dim=-1)

# Model with: dropout, rnn, linear(out=h)
class Model15(nn.Module):
	def __init__(self, embedding_dim, n_hidden, n_out, drop_p):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.embedding_dim, self.n_hidden, self.n_out, self.n_layers = embedding_dim, n_hidden, n_out, 1
		self.drop = nn.Dropout(p=drop_p).to(self.device)
		# self.bn = nn.BatchNorm1d(self.n_hidden).to(self.device)
		self.rnn = nn.RNN(self.embedding_dim, self.n_hidden, num_layers=self.n_layers).to(self.device)
		self.out = nn.Linear(self.n_hidden, self.n_out).to(self.device)

	def forward(self, seq, lengths, train=False):
		embs = torch.tensor(seq).float().to(self.device)
		if train:
			embs = self.drop(embs)
		embs = pack_padded_sequence(embs, lengths)
		rnn_out, self.h = self.rnn(embs)
		rnn_out, lengths = pad_packed_sequence(rnn_out)
		#if train:
		#	outp = self.out(self.drop(self.h[-1]).view(self.h[-1].shape[1:3]))
		#else:
		#	outp = self.out(self.h[-1].view(self.h[-1].shape[1:3]))
		outp = self.out(self.h.view(self.h.shape[1:3]))
		return F.log_softmax(outp, dim=-1)