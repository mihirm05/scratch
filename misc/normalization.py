import numpy as np 
import matplotlib.pyplot as plt 
import random 

def normalization(x):
    #print('centroid before: ',np.mean(x, 0))
	m = np.mean(x, 0)
	x = x - m
	print('mean before: ',m)
	print('distance before normalization: ',np.mean(np.hypot(x[:],0)))
	scale  = (np.sqrt(2)) / np.mean(np.hypot(x[:],0))
	print('scale is: ',scale)
	x = x*scale
    
	print('centroid after: ',np.mean(x,0))
	print('distance after normalization: ',np.mean(np.hypot(x[:],0)))
	print('normalized x is: \n',x)
	plt.scatter(x,np.linspace(-10,10,10),label='After')
	plt.legend()
	plt.grid()
	plt.show()
	return x


def main():
	data = []
	for i in range(10):
		data.append(np.random.randint(-10,10))
	print('data is: ',data)
	print('mean is: ',np.mean(data))
	print('var is: ',np.power(np.std(data),2))
	print(len(data))
	print(len(np.linspace(-10,10,10)))
	plt.scatter(data,np.linspace(-10,10,10),label='Before')
	normalization(data)
	
	
if __name__ == "__main__":
	main()




