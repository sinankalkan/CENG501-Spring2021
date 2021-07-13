import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import load_data
from tqdm import tqdm, trange
from sklearn.metrics import classification_report, accuracy_score

padded = False

train_tensor, train_label, val_tensor, val_label, test_tensor, test_label = load_data(padded=padded)

unseen_intent = "GetWeather"

label_dict = {}
for i, l in enumerate(set(train_label)):
	if l != unseen_intent:
		label_dict[l] = len(label_dict)

label_dict[unseen_intent] = len(label_dict)

if padded:
	train_data_seen = torch.zeros(train_tensor.shape[0], train_tensor.shape[1] - train_label.count(unseen_intent), train_tensor.shape[2])
	train_label_seen = torch.zeros(train_tensor.shape[1] - train_label.count(unseen_intent), dtype=torch.long)
	val_data_seen = torch.zeros(val_tensor.shape[0], val_tensor.shape[1] - val_label.count(unseen_intent), val_tensor.shape[2])
	val_label_seen = torch.zeros(val_tensor.shape[1] - val_label.count(unseen_intent), dtype=torch.long)
	test_data_seen = torch.zeros(test_tensor.shape[0], test_tensor.shape[1] - test_label.count(unseen_intent), test_tensor.shape[2])
	test_label_seen = torch.zeros(test_tensor.shape[1] - test_label.count(unseen_intent), dtype=torch.long)
	train_data_unseen = torch.zeros(train_tensor.shape[0], train_label.count(unseen_intent), train_tensor.shape[2])
	train_label_unseen = torch.zeros(train_label.count(unseen_intent), dtype=torch.long)
	val_data_unseen = torch.zeros(val_tensor.shape[0], val_label.count(unseen_intent), val_tensor.shape[2])
	val_label_unseen = torch.zeros(val_label.count(unseen_intent), dtype=torch.long)
	test_data_unseen = torch.zeros(test_tensor.shape[0], test_label.count(unseen_intent), test_tensor.shape[2])
	test_label_unseen = torch.zeros(test_label.count(unseen_intent), dtype=torch.long)
	j,k = 0,0
	for i in range(train_tensor.shape[1]):
		if train_label[i] == unseen_intent:
			train_data_unseen[:,j,:] = train_tensor[:,i,:]
			train_label_unseen[j] = label_dict[train_label[i]]
			j += 1
		else:
			train_data_seen[:,k,:] = train_tensor[:,i,:]
			train_label_seen[k] = label_dict[train_label[i]]
			k += 1
	j,k = 0,0
	for i in range(val_tensor.shape[1]):
		if val_label[i] == unseen_intent:
			val_data_unseen[:,j,:] = val_tensor[:,i,:]
			val_label_unseen[j] = label_dict[val_label[i]]
			j += 1
		else:
			val_data_seen[:,k,:] = val_tensor[:,i,:]
			val_label_seen[k] = label_dict[val_label[i]]
			k += 1
	j,k = 0,0
	for i in range(test_tensor.shape[1]):
		if test_label[i] == unseen_intent:
			test_data_unseen[:,j,:] = test_tensor[:,i,:]
			test_label_unseen[j] = label_dict[test_label[i]]
			j += 1
		else:
			test_data_seen[:,k,:] = test_tensor[:,i,:]
			test_label_seen[k] = label_dict[test_label[i]]
			k += 1
else:
	train_data_seen, val_data_seen, test_data_seen, train_data_unseen, val_data_unseen, test_data_unseen = [], [], [], [], [], []
	train_label_seen = torch.zeros(len(train_tensor) - train_label.count(unseen_intent), dtype=torch.long)
	val_label_seen = torch.zeros(len(val_tensor) - val_label.count(unseen_intent), dtype=torch.long)
	test_label_seen = torch.zeros(len(test_tensor) - test_label.count(unseen_intent), dtype=torch.long)
	train_label_unseen = torch.zeros(train_label.count(unseen_intent), dtype=torch.long)
	val_label_unseen = torch.zeros(val_label.count(unseen_intent), dtype=torch.long)
	test_label_unseen = torch.zeros(test_label.count(unseen_intent), dtype=torch.long)
	j,k = 0,0
	for i in range(len(train_tensor)):
		if train_label[i] == unseen_intent:
			train_data_unseen.append(train_tensor[i])
			train_label_unseen[j] = label_dict[train_label[i]]
			j += 1
		else:
			train_data_seen.append(train_tensor[i])
			train_label_seen[k] = label_dict[train_label[i]]
			k += 1
	j,k = 0,0
	for i in range(len(val_tensor)):
		if val_label[i] == unseen_intent:
			val_data_unseen.append(val_tensor[i])
			val_label_unseen[j] = label_dict[val_label[i]]
			j += 1
		else:
			val_data_seen.append(val_tensor[i])
			val_label_seen[k] = label_dict[val_label[i]]
			k += 1
	j,k = 0,0
	for i in range(len(test_tensor)):
		if test_label[i] == unseen_intent:
			test_data_unseen.append(test_tensor[i])
			test_label_unseen[j] = label_dict[test_label[i]]
			j += 1
		else:
			test_data_seen.append(test_tensor[i])
			test_label_seen[k] = label_dict[test_label[i]]
			k += 1


class RNNClassifier(nn.Module):
	def __init__(self, input_size, hidden_size, hidden_vector, num_seen_class):
		super(RNNClassifier, self).__init__()
		self.hidden_size = hidden_size
		self.hidden_vector = nn.Parameter(hidden_vector)
		self.rnn = nn.RNN(input_size, hidden_size)
		self.linear = nn.Linear(hidden_size, num_seen_class)
	def forward(self, input):
		return self.linear(self.rnn(input, self.hidden_vector.repeat(1, input.shape[1], 1))[1]).view(-1, 1)
		#return self.linear(self.rnn(input, self.hidden_vector.repeat(1, input.shape[1], 1))[0])[-1,:,:].view(-1, 1)
	def to(self, device):
		super(RNNClassifier, self).to(device)
		return self


class IntentSpaceClassifier(nn.Module):
	def __init__(self, input_size, hidden_size, num_seen_class, euclidean=True):
		super(IntentSpaceClassifier, self).__init__()
		self.num_seen_class = num_seen_class
		self.hidden_size = hidden_size
		self.euclidean = euclidean
		self.device="cpu"
		self.coordinates = nn.Parameter(torch.eye(num_seen_class, requires_grad=True))
		self.softmax = nn.Softmax(dim=0)
		#self.bases = nn.Parameter(torch.randn(num_seen_class, hidden_size, hidden_size))
		self.rnns = [RNNClassifier(input_size, hidden_size, torch.randn(1, 1, hidden_size, requires_grad=True), num_seen_class) for _ in range(num_seen_class)]
		#self.rnn = nn.RNN(input_size, hidden_size)
		#self.rnn.weight_hh_l0 = nn.Parameter((self.coordinates.view(self.num_seen_class, self.num_seen_class, 1, 1) * \
		#self.bases.view(1, self.num_seen_class, self.hidden_size, self.hidden_size)).sum(axis=0).sum(axis=0))
		self.expansion = [nn.Parameter(torch.eye(num_seen_class, requires_grad=True)) for _ in range(num_seen_class)]
		self.fc = nn.Linear(num_seen_class, 1).requires_grad_(False)
	def forward(self, input):
		#torch.cat(list(map(lambda x: x(input), self.rnns)), axis=1).exp()
		#scores = torch.zeros(input.shape[1], self.num_seen_class).to(self.device)
		bases=[]
		for i in range(len(self.rnns)):
			#scores[:,i:i+1] = self.rnns[i](input)
			base = self.rnns[i](input)
			base = torch.matmul(self.expansion[i], base)
			bases.append(base)
		bases = torch.cat(bases, dim=1).to(self.device)
		if self.euclidean:
			scores = self.fc(torch.matmul(bases, self.coordinates)).view(1,-1)
		else:
			scores = self.fc(torch.matmul(bases, self.softmax(self.coordinates))).view(1,-1)
		return torch.softmax(scores, dim=1)
	"""
	def parameters(self):
		for iterable in list(map(lambda x:x.parameters(), self.rnns)) + list(super(IntentSpaceClassifier, self).parameters()):
			yield from iterable
	"""
	def to(self, device):
		super(IntentSpaceClassifier, self).to(device)
		self.device=device
		_=list(map(lambda x: x.to(device), self.rnns))
		self.expansion=list(map(lambda x: x.to(device), self.expansion))
		return self


device="cuda:0"

model = IntentSpaceClassifier(300, 128, 7, euclidean=False).train().to(device)

#optimizer = optim.Adam(model.parameters(), lr=0.1)

optimizer = optim.SGD([{"params":model.parameters(), "lr":0.1, "weight_decay":1e-5}] + [{"params":rnn.parameters(), "lr":0.1} for rnn in model.rnns])
model.coordinates.requires_grad = False
loss_fn = nn.CrossEntropyLoss()

if padded:
	batch_size = 8
else:
	batch_size = 1

eval_steps = 1000
#train_data_seen = torch.randn([35, 11188, 300])

do_train = True

for e in range(15):
	if not do_train:
		break
	losses = []
	if padded:
		num_steps = int(train_data_seen.shape[1] / batch_size) + 1
	else:
		num_steps = len(train_data_seen)
	for i in trange(num_steps):
		#print(model.rnns[2].rnn.weight_hh_l0)
		#print(model.coordinates)
		if i != 0 and i % eval_steps == 0:
			with torch.no_grad():
				_=model.eval()
				if padded:
					pred = model(val_data_seen.to(device)).argmax(dim=1)
				else:
					pred = []
					for i in range(len(val_data_seen)):
						pred.append(model(val_data_seen[i].unsqueeze(1).to(device)).argmax(dim=1).item())
			acc = accuracy_score(val_label_seen.numpy(), pred)
			print("Accuracy score on the evaluation split: {0:.3f}".format(acc))
			_=model.train()
			if acc > 0.8:
				#do_train = False
				break
		if padded:
			out = model(train_data_seen[:,i*batch_size:(i+1)*batch_size,:].to(device))
			loss = loss_fn(out, train_label_seen[i*batch_size:(i+1)*batch_size].to(device))
		else:
			out = model(train_data_seen[i].unsqueeze(1).to(device))
			loss = loss_fn(out, train_label_seen[i].view(-1).to(device))
		loss.backward()
		losses.append(loss.item())
		optimizer.step()
		#print(out)
		optimizer.zero_grad()
		#input()
	if e > 6:
		model.coordinates.requires_grad = False
		optimizer = optim.SGD([{"params":rnn.parameters(), "lr":0.1} for rnn in model.rnns])
	elif e > 3:
		model.coordinates.requires_grad = True
		optimizer = optim.SGD([{"params":model.parameters(), "lr":0.1, "weight_decay":1e-5}])

_=model.eval()

with torch.no_grad():
	if padded:
		pred = model(test_data_seen.to(device)).argmax(dim=1)
	else:
		pred = []
		for i in range(len(test_data_seen)):
			pred.append(model(test_data_seen[i].unsqueeze(1).to(device)).argmax(dim=1).item())


print(classification_report(test_label_seen.numpy(), pred))

