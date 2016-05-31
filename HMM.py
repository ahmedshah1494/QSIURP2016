from scipy.stats import multivariate_normal
import numpy as np
from sklearn import mixture
import os
import sys
import math
sys.setrecursionlimit(10000)
def getGaussianParams(obs):
	if len(obs) == 0:
		return ([0.0]*3,[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
	means = [0] * len(obs[0])
	for o in obs:
		for i in range(len(o)):
			means[i] += o[i]
	means = map(lambda x : float(x) / len(obs), means)
	# print means
	cov = []
	for i in range(len(obs[0])):
		vals = map(lambda x : x[i], obs)
		var = max(1.0, reduce(lambda x,y : (((x - means[i])**2)/(len(vals) - 1)) + y, vals))
		row = [0] * len(obs[0])
		row[i] = var
		cov.append(row)
	return (means,cov)

class hmm:
	def __init__(self, nstates):
		# super(hmm,self).__init__()
		self.nStates = nstates
		self.obsProbs = [0]*self.nStates
		self.transProbs = np.zeros((self.nStates, self.nStates))
		self.pi = [0]*self.nStates

	def getMemTable(self):
		T = {}
		for i in range(self.nStates):
			T[i] = {}
		return T

	def getObsProb(self,obs,state):
		(m,c) = self.obsProbs[state]
		# print obs, m, c
		return multivariate_normal.pdf(obs, mean=m, cov=c)
		# print gmm.score([obs])[0]
		# return np.e ** gmm.score([obs])[0]
	def bestPathScr(self, state, time, obsSeq, BPS):
		if time == 0:
			return self.pi[state] * self.getObsProb(obsSeq[time],state)
		# print BPS, state, time
		if BPS[state].get(time) != None:
			return BPS[state][time]
		else:
			prevStates = range(0, state + 1)
			# prevStates = range(max(0,state - 2), state + 1)
			BPS[state][time] = self.getObsProb(obsSeq[time],state) * max(map(lambda x : self.bestPathScr(x,time-1,obsSeq,BPS) * self.transProbs[x][state],prevStates))
			return BPS[state][time]

	def getBestParent(self, state,time,obsSeq,BPS):
		prevStates = range(max(0,state - 2), state + 1)
		return prevStates[np.argmax(map(lambda x : self.bestPathScr(x,time-1,obsSeq,BPS) * self.transProbs[x][state],prevStates))]

	def getBestState(self, obsSeq, time, BPS):
		if time == len(obsSeq) - 1:
			# print lambda x : self.bestPathScr(x,time,obsSeq,BPS), range(self.nStates)
			return np.argmax(map(lambda x : self.bestPathScr(x,time,obsSeq,BPS), range(self.nStates)))
		else:
			return self.getBestParent(self.getBestState(obsSeq, time + 1, BPS), time + 1,obsSeq,BPS)

	def train(self,nIters, obsSeqs):
		for it in range(nIters):
			sys.stdout.write("\r 		%d" % it)
			sys.stdout.flush()
			stateSeqs = []
			for j in range(len(obsSeqs)):
				obsSeq = obsSeqs[j]
				stateSeq = []
				obsSeq = filter(lambda x : x != [0.0,0.0,0.0], obsSeq)
				frac = len(obsSeq) / self.nStates 
				# bestState = np.argmax(map(bestPathScr(s,len(obs) - 1),range(self.nStates)))
				if (it > 0):
					bestBPS = 0
					bestPathScore = float("-infinity")
					for i in range(self.nStates):
						sys.stdout.write("\r 			state: %d 	obsSeq#: %d" % (i,j))
						sys.stdout.flush()
						BPS = self.getMemTable()
						pathScore = self.bestPathScr(i,len(obsSeq) - 1, obsSeq,BPS)
						# print pathScore, BPS
						if (pathScore > bestPathScore):
							bestPathScore = pathScore
							bestBPS = BPS
					# print bestPathScore
					# print bestBPS
				for i in range(len(obsSeq)):
					x = obsSeq[i]
					if (it == 0):
						# print i, i/frac, len(obs)
						stateSeq.append(min(i/frac,self.nStates-1))
					else:

						# print self.obsProbs
						# currState = stateSeq[-1] if len(stateSeq) > 0 else 0
						# nextStates = range(currState, currState + min(2,self.nStates - currState) + 1)
						# # print currState, nextStates, self.obsProbs
						# # print map(lambda (m,c) : multivariate_normal.pdf(x, mean=m, cov=c), self.obsProbs[nextStates[0]:nextStates[-1] + 1])
						# nextState = nextStates[np.argmax(map(lambda (m,c) : multivariate_normal.pdf(x, mean=m, cov=c), self.obsProbs[nextStates[0]:nextStates[-1] + 1]))]
						
						stateSeq.append(self.getBestState(obsSeq,i,bestBPS))
						# stateSeq.append(nextState)
				stateSeqs.append(stateSeq)
				# print stateSeq
			self.pi = [0] * self.nStates
			self.transProbs = np.zeros((self.nStates, self.nStates))
			O_S = []
			for i in range(self.nStates):
				O_S.append([])

			for j in range(len(stateSeqs)):
				ss = stateSeqs[j]
				self.pi[ss[0]] += 1

				for i in range(len(ss) - 1):
					# print ss[i], ss[i+1]
					self.transProbs[ss[i]][ss[i+1]] += 1

				for i in range(len(ss)):
					O_S[ss[i]].append(obsSeqs[j][i])

			for i in range(self.nStates):
				self.obsProbs[i] = getGaussianParams(O_S[i])
				# if (O_S[i] == []):
				# 	continue
				# self.obsProbs[i] = mixture.GMM(n_components=1,covariance_type='full',min_covar=1.0,verbose=1)
				# self.obsProbs[i].fit(O_S[i])

			# print self.transProbs
			self.transProbs = map(lambda x : map(lambda x2 : float(x2) / sum(x) if sum(x) > 0 else (0.0), x), self.transProbs)
			self.pi = map(lambda x : float(x) / sum(self.pi), self.pi)
	def test(self,obs):
		def alpha(state, time, observations, A):
			if time == 0:
				# print self.pi[state] * multivariate_normal.pdf(observations[0], mean=self.obsProbs[state][0], cov=self.obsProbs[state][1])
				return self.pi[state] * multivariate_normal.pdf(observations[0], mean=self.obsProbs[state][0], cov=self.obsProbs[state][1])
			if A[state].get(time) != None:
				return A[state][time]
			else:
				Sum = 0
				prevStates = []
				nPrevs = min(3,state+1)
				for i in range(nPrevs):
					s = state - i
					Sum += alpha(s,time - 1,observations,A) * self.transProbs[s][state] * multivariate_normal.pdf(observations[time], mean=self.obsProbs[state][0], cov=self.obsProbs[state][1])
				A[state][time] = Sum
				return Sum
		Sum = 0
		for s in range(self.nStates):
			Sum += alpha(s,len(obs)-1,obs,self.getMemTable())
		return Sum

def getObsSeq(d):
	files = os.listdir(d)
	files = filter(lambda x : '.feat' in x, files)
	obsSeqs = []
	for F in files:
		f = open(d+F, 'r')
		lines = f.readlines()
		f.close()
		lines = filter(lambda x : x.find("NaN") == -1, lines)
			# print filename
		lines = map(lambda x : map(lambda x2 : float(x2), x.split(',')), lines)
		obsSeqs.append(lines)
	return obsSeqs

# nStates = 7
# nIters = 2
# HMMs = {}
# alphabets = ['a','b','c']
# for c in alphabets:
# 	sys.stdout.write("\r%s %s" % ('training',c))
# 	sys.stdout.flush()
# 	HMMs[c] = hmm(nStates)
# 	HMMs[c].train(nIters,getObsSeq('characterdata/train/%s/' % c))

nStates = 7
nIters = 2
HMMs = {}
train_rooms = ['1030',"bathroom_lockers"]
test_rooms = ['2052','bathroom2_locker']
for c in train_rooms:
	sys.stdout.write("\r%s %s" % ('training',c))
	sys.stdout.flush()
	HMMs[c] = hmm(nStates)
	HMMs[c].train(nIters,getObsSeq('files/%s/' % c))
	

# f = open('HMM_results.txt','w')
# corrCount = 0
# totalCount = 0
# for c1 in test_rooms:
# 	obsSeqs = getObsSeq('files/%s/' % c1)
# 	actual = c1
# 	for i in range(len(obsSeqs)):
# 		sys.stdout.write("\r%s %s %d" % ("testing",c1,i))
# 		sys.stdout.flush()
# 		obsSeq = obsSeqs[i]
# 		pred = alphabets[np.argmax(map(lambda x : HMMs[x].test(obsSeq),alphabets))]
# 		if (actual == pred):
# 			corrCount +=1
# 		totalCount += 1
# 		f.write("%s %d %c\n" % (actual, i, pred))
# f.close()
# print float(corrCount)/totalCount

# f = open('HMM_results.txt')
# lines = f.readlines()
# lines = map(lambda x : x.split(), lines)
# charCorrect = {}
# charCount = {}
# for line in lines:
# 	if (charCorrect.get(line[0]) == None):
# 		charCorrect[line[0]] = 0
# 	if (charCount.get(line[0]) == None):
# 		charCount[line[0]] = 0
# 	charCount[line[0]] += 1
# 	if (line[0] == line[2]):
# 		charCorrect[line[0]] += 1

# for c in charCorrect:
# 	print c, float(charCorrect[c]) / charCount[c]





