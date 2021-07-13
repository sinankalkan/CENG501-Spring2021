import csv
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm

def load_data(padded=True):
	"""
	with open("glove.840B.300d.pkl", "rb") as f:
		vectors = pkl.load(f)
	"""
	with open("used_vectors.pkl", "rb") as f:
		vectors=pkl.load(f)
	#avg_vec = torch.tensor(np.average(list(map(lambda x: x.numpy(), vectors.values())), axis=0))
	avg_vec = vectors["avg_vector"]
	with open("data/snips/train.csv", "r") as f:
		r = csv.reader(f)
		train_data, train_label = [], []
		for row in r:
			train_data.append(row[0])
			train_label.append(row[1])
	with open("data/snips/valid.csv", "r") as f:
		r = csv.reader(f)
		val_data, val_label = [], []
		for row in r:
			val_data.append(row[0])
			val_label.append(row[1])
	with open("data/snips/test.csv", "r") as f:
		r = csv.reader(f)
		test_data, test_label = [], []
		for row in r:
			test_data.append(row[0])
			test_label.append(row[1])
	if padded:
		max_len = max(list(map(lambda x: len(x.split()), train_data + val_data + test_data)))
		train_tensor = torch.zeros(max_len, len(train_data), 300)
		val_tensor = torch.zeros(max_len, len(val_data), 300)
		test_tensor = torch.zeros(max_len, len(test_data), 300)
		for i, d in tqdm(enumerate(train_data), total=len(train_data)):
			for j, w in enumerate(d.split()):
				if w in vectors:
					train_tensor[j,i,:] = vectors[w]
				else:
					train_tensor[j,i,:] = avg_vec
		for i, d in tqdm(enumerate(val_data), total=len(val_data)):
			for j, w in enumerate(d.split()):
				if w in vectors:
					val_tensor[j,i,:] = vectors[w]
				else:
					val_tensor[j,i,:] = avg_vec
		for i, d in tqdm(enumerate(test_data), total=len(test_data)):
			for j, w in enumerate(d.split()):
				if w in vectors:
					test_tensor[j,i,:] = vectors[w]
				else:
					test_tensor[j,i,:] = avg_vec
		return train_tensor, train_label, val_tensor, val_label, test_tensor, test_label
	else:
		train_tensor, val_tensor, test_tensor = [], [], []
		for i, d in tqdm(enumerate(train_data), total=len(train_data)):
			sentence = torch.zeros(len(d.split()), 300)
			for j, w in enumerate(d.split()):
				if w in vectors:
					sentence[j,:] = vectors[w]
				else:
					sentence[j,:] = avg_vec
			train_tensor.append(sentence)
		for i, d in tqdm(enumerate(val_data), total=len(val_data)):
			sentence = torch.zeros(len(d.split()), 300)
			for j, w in enumerate(d.split()):
				if w in vectors:
					sentence[j,:] = vectors[w]
				else:
					sentence[j,:] = avg_vec
			val_tensor.append(sentence)
		for i, d in tqdm(enumerate(test_data), total=len(test_data)):
			sentence = torch.zeros(len(d.split()), 300)
			for j, w in enumerate(d.split()):
				if w in vectors:
					sentence[j,:] = vectors[w]
				else:
					sentence[j,:] = avg_vec
			test_tensor.append(sentence)
		return train_tensor, train_label, val_tensor, val_label, test_tensor, test_label
