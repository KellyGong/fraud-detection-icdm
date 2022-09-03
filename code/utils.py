import torch
import numpy as np
import random as rd


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True


class EarlyStop:
	def __init__(self, interval=10):
		self.interval = interval
		self.evaluation_metric = -1
		self.no_improve_epoch = 0
	
	def update(self, evaluation_metric):
		if evaluation_metric > self.evaluation_metric:
			self.evaluation_metric = evaluation_metric
			self.no_improve_epoch = 0
		else:
			self.no_improve_epoch += 1
		if self.no_improve_epoch == self.interval:
			return True
		else:
			return False