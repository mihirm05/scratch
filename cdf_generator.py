import numpy as np
import matplotlib.pyplot as plt


def dice_throw(trials):
	"""
	trials: number of times the experiment needs to be repeated
	returns: dice output for experiments 
	"""
	return np.random.randint(low=1, high=7, size=(trials,1),dtype='l').T


def two_dice_throw(trials):
	"""
	trials: number of times the experiment needs to be repeated
	returns: dice output for experiments  
	"""
	d1 = dice_throw(trials)
	d2 = dice_throw(trials)
	return d1, d2
     

def two_dice_throw_sum(trials):
	"""
	trials: number of times the experiment needs to be repeated
	returns: sum of dice output for experiments  
	"""
	d1 = dice_throw(trials)
	d1 = dice_throw(trials)
	d2 = dice_throw(trials)
	return d1 + d2
    

def plt_hist(vals):
	"""
	vals: output of two_dice_throw_sum function
	returns: histogram of values 
	"""
	possible_die_rolls = np.arange(2,13)
	fig= plt.figure(figsize=(8, 4), dpi=100)
	plt.grid()
	plt.xlabel("sum of two dice")
	plt.ylabel("frequency")
	plt.title('Probability distribution of sum of two dice')
	plt.hist(np.squeeze(vals), bins=possible_die_rolls)
	plt.show()


def plt_cdf(vals, trials):
	"""
	vals: output of two_dice_throw_sum function
	returns: cdf of values 
	"""
	values, base = np.histogram(vals, bins=np.linspace(2,14,11))
	real = np.cumsum(values)/trials
	plt.grid()
	plt.plot(np.linspace(2,13,10), real)
	plt.ylabel('Probability')
	plt.xlabel('Sum of two dice')
	plt.title('Cumulative probability function of sum of two dice')
	plt.show()
	

if __name__ == "__main__":
	thousand_rolls = two_dice_throw_sum(1000)
	plt_hist(thousand_rolls)
	plt_cdf(thousand_rolls, 1000)

