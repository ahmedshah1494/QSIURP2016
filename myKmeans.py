import numpy as np 
import dispy
import multiprocessing as mp
def getDistance((x,y)):
	import numpy as np 
	x = np.matrix(map(float,x))
	y = np.matrix(map(float,y))
	d = x - y
	d = d.dot(d.T)
	d = d.getA()
	return sum(d)
# node1 is the local machine
node1 = dispy.NodeAllocate("0.0.0.0",port=9998)
# node 2 is a remote machine
node2 = dispy.NodeAllocate('10.33.49.230', port=9993)
node3 = dispy.NodeAllocate('10.33.49.17', port=9991)
node4 = dispy.NodeAllocate('10.33.49.32', port=9992)
nodeList = [node1,node2,node3,node4]
distCluster = dispy.JobCluster(getDistance,nodeList)

def kmeans_PP(K, dataPoints):
	centroids = []
	centroidIdx = np.random.choice(len(dataPoints))
	centroids.append(map(float,dataPoints[centroidIdx]))
	del dataPoints[centroidIdx]
	# dataPoints = np.delete(dataPoints[centroidIdx], centroidIdx, 0)

	while len(centroids) < K:
		print len(centroids)
		nearestCentroidDist = map(lambda x : map(lambda y: distCluster.submit((x,y)),centroids),dataPoints)
		nearestCentroidDist = map(lambda x : 1.0/(min(map(lambda j: j(),x)) ** 2),nearestCentroidDist)
		nearestCentroidDist = map(lambda x : 1.0/(min(map(lambda j: j(),x)) ** 2),nearestCentroidDist)
		print "Dist calculated"
		centroidProbs = map(lambda x : x/sum(nearestCentroidDist), nearestCentroidDist)
		centroidIdx = np.random.choice(len(dataPoints),p=map(lambda x : x[0], centroidProbs))
		print centroidIdx
		newCentroid = dataPoints[centroidIdx]
		del dataPoints[centroidIdx]
		# dataPoints = np.delete(dataPoints[centroidIdx], centroidIdx, 0)
		centroids.append(newCentroid)
	return centroids

if __name__ == '__main__':
	# data = np.loadtxt('files/DataSets/universal.Ntrain', delimiter=',')
	# data = data.reshape(1, -1)
	with open('files/DataSets/universal.Ntrain','r') as f:
		data = f.readlines()
	data = map(lambda x : x.split(','), data)
	kmeans_PP(1024, data)
	distCluster.close()