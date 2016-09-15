'''
% Réalisée par: NGUYEN Thi Ngoc Tam
% Département: Master 2 MIAGE at University of Paris Dauphine
% Sujet: Construction des portefeuilles efficients dans le cadre des modèles de Markowitz, Sharpe, CAPM
% Date: 31/08/2016
'''

### %%%%%%%%%%%%%%%%%%%%%%%%% Libraries %%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy import matrix, power

from matplotlib import pyplot as plt
from matplotlib import ticker

import cvxopt as opt
from cvxopt import blas, solvers

from math import sqrt
import itertools 
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

risk_free_rate = 0.4

## %%%%%%%%%%%%%%% Définir tous les input csv (data set) %%%%%%%%%%%%%%%%
def codes():
	codes = []
	codes[0] = "Data/Apple"
	codes[1] = "Data/Microsoft"
	codes[2] = "Data/AMZN"
	codes[3] = "Data/GE"
	return codes

## %%%%%%%%%%%%%%%%% Lire file CSV %%%%%%%%%%%%%%%%%%%%
def readFile(fileName):
	return pd.read_csv(fileName)

data_apple = readFile(codes()[0] + '.csv')
data_micro = readFile(codes()[1] + '.csv')
data_amazon = readFile(codes()[2] + '.csv')
data_ge = readFile(codes()[3] + '.csv')

## %%%%%%%%%%%%%%%%% Utilisé pour calcul CAPM %%%%%%%%%%%%%%%%
data_sp500 = readFile('SP500' + '.csv')

## %%%%%%%%%%%%%%%%%%%% Calculer le tableau de rendement  %%%%%%%%%%%%%%%%%%%%%%
def calculReturn(data):
	rendement = []
	for i in range(len(data)-1, 0, -1):
		rendement.append((data["Adj Close"][i-1] - data["Adj Close"][i])*100 / data["Adj Close"][i])
	return rendement
	
### %%%%%%%%%%%%%%%%%%%%%%%%% Rendement chaque actif %%%%%%%%%%%%%%%%%%%%%%%%%
##- Valeur rendement hebdomadaire
##- Valeur rendement moyen

rendement_apple = calculReturn(data_apple)
rendement_micro = calculReturn(data_micro)
rendement_amazon= calculReturn(data_amazon)
rendement_ge = calculReturn(data_ge)
rendement_sp500 = calculReturn(data_sp500)

## %%%%%%%%%%%%%%%%%%%%%%%%% Calculer le rendement moyen %%%%%%%%%%%%%%%%%%%%%%%%%
def calculReturnMoyen(rendementList):
	return sum(rendementList)/float(len(rendementList))

rendementAppleMoyen = calculReturnMoyen(rendement_apple)
rendementMicroMoyen = calculReturnMoyen(rendement_micro)
rendementAmazonMoyen = calculReturnMoyen(rendement_amazon)
rendementGEMoyen = calculReturnMoyen(rendement_ge)
rendementSP500Moyen = calculReturnMoyen(rendement_sp500)

#print("rendement apple: \n", rendement_apple)
print("rendement Apple Moyen: \n", rendementAppleMoyen)
#print("rendement micro: \n", rendement_micro)
print("rendement Microsoft Moyen: \n", rendementMicroMoyen)
#print("rendement Amazon: \n", rendement_amazon)
print("rendement Amazon Moyen: \n", rendementAmazonMoyen)
#print("rendement GE: \n", rendement_ge)
print("rendement GE Moyen: \n", rendementGEMoyen)
#print("rendement SP500: \n", rendement_sp500)
print("rendement SP500 Moyen: \n", rendementSP500Moyen)

### %%%%%%%%%%%%%%%%%%%%%%%%% Rendement tous les actifs %%%%%%%%%%%%%%%%%%%%%%%%%
##- Rendement portefeuille avec les propartion (poids) des actifs

### Variances et covariance
##- Calcul variance de chaque actif
##	+ Calcul mean
##	+ Calcul variance

def calculMean(data):
	return np.mean(data)

def calculVariance(data, mean):
	sum = 0
	for i in range(0, len(data)):
		sum += (data[i] - mean)**2
	variance = sum/float(len(data))
	return variance

variance_rendement_sp500 = calculVariance(rendement_sp500, rendementSP500Moyen)

##- %%%%%%%%%%%%%%%%%%%%%%%%% Calculer covariance entre deux actifs %%%%%%%%%%%%%%%%%%%%%%%%%
def calculCoVar(data1, data2, mean1, mean2):
	sum = 0
	for i in range(0, len(data1)):
		sum1 = data1[i] - mean1
		sum2 = data2[i] - mean2
		sum += sum1*sum2
	coVar = sum / len(data1)
	return coVar

coVar_Apple_SP500 = calculCoVar(rendement_apple, rendement_sp500, rendementAppleMoyen, rendementSP500Moyen)
print("Covariance entre apple et microsoft: ", coVar_Apple_SP500)

coVar_Micro_SP500 = calculCoVar(rendement_micro, rendement_sp500, rendementMicroMoyen, rendementSP500Moyen)
print("Covariance entre apple et microsoft: ", coVar_Micro_SP500)

coVar_Amazon_SP500 = calculCoVar(rendement_amazon, rendement_sp500, rendementAmazonMoyen, rendementSP500Moyen)
print("Covariance entre apple et microsoft: ", coVar_Amazon_SP500)

coVar_GE_SP500 = calculCoVar(rendement_ge, rendement_sp500, rendementGEMoyen, rendementSP500Moyen)
print("Covariance entre apple et microsoft: ", coVar_GE_SP500)

beta_apple = coVar_Apple_SP500 / variance_rendement_sp500
print("The Beta for Apple and SP500: \n", beta_apple)

beta_micro = coVar_Micro_SP500 / variance_rendement_sp500
print("The Beta for Microsoft and SP500: \n", beta_micro)

beta_amazon = coVar_Amazon_SP500 / variance_rendement_sp500
print("The Beta for Apple and SP500: \n", beta_amazon)

beta_ge = coVar_GE_SP500 / variance_rendement_sp500
print("The Beta for Apple and SP500: \n", beta_ge)


print("The CAPM of Apple: \n", risk_free_rate + beta_apple * (rendementSP500Moyen - risk_free_rate))
print("The CAPM of Microsoft: \n", risk_free_rate + beta_micro * (rendementSP500Moyen - risk_free_rate))
print("The CAPM of Amazon: \n", risk_free_rate + beta_amazon * (rendementSP500Moyen - risk_free_rate))
print("The CAPM of GE: \n", risk_free_rate + beta_ge * (rendementSP500Moyen - risk_free_rate))


##- %%%%%%%%%%%%%%%%%%%%%%%%% Calculer covariance entre deux actifs en utilisant la fonction COV de Python %%%%%%%%%%%%%%%%%%%%%%%%%
rendement = [rendement_apple, rendement_micro, rendement_amazon, rendement_ge]
cov = np.cov(np.asmatrix(rendement))
print("cov: \n", cov)


## %%%%%%%%%%%%%%%%%%%%%%%%% Créer la matrice de variance et covariance %%%%%%%%%%%%%%%%%%%%%%%%%
def var_covar(codes):
	n = len(codes)
	matrice = [[0 for x in range(n)] for y in range(n)]
	data = []
	rendement = []
	mean = []
	
	for i in range(n):
		data.append(readFile(codes[i] + ".csv"))
		rendement.append(calculReturn(data[i]))
		mean.append(calculMean(rendement[i]))
		matrice[i][i] = calculVariance(rendement[i], mean[i])
	for i in range(n):
		for j in range(i+1,n):
			matrice[i][j] = calculCoVar(rendement[i], rendement[j], mean[i], mean[j])
		
	for i in range(1,n):
		for j in range(0,i):
			matrice[i][j] = calculCoVar(rendement[i], rendement[j], mean[i], mean[j])
	return matrice

matrice = var_covar(codes())
print("Matrice var et covariance: \n", np.asmatrix(matrice))
	
##- %%%%%%%%%%%%%%%%%%%%%%%%% Calculer variance de portefeuille avec les allocations des actifs (créer matrice) %%%%%%%%%%%%%%%%%%%%%%%%%
## Produces n random weights that sum to 1

def rand_weights(n):   
	#np.random.seed(123)
	k = np.random.rand(n)
	return k / sum(k)

p = np.array((rendementAppleMoyen,rendementMicroMoyen,rendementAmazonMoyen,rendementGEMoyen))

## %%%%%%%%%%%%%%%%%%%%%%%%% Créer des portefeuilles %%%%%%%%%%%%%%%%%%%%%%%%%
def random_portfolio(returns):
	#poids
	w = np.asmatrix(rand_weights(len(codes())))

	#rendement
	p = np.asmatrix((rendementAppleMoyen,rendementMicroMoyen,rendementAmazonMoyen,rendementGEMoyen))

	rendement_portefeuille = w * p.T
	
	#variance covariance matrix
	C = np.asmatrix(returns)

	#risque = np.sqrt(w * (w * C).T)
	risque = np.sqrt(w * C * w.T)
	
	#print("sigma: \n", risque)
	return rendement_portefeuille, risque

n_portfolios = 1000
means, stds = np.column_stack([random_portfolio(cov) for _ in range(n_portfolios)])

''' ### test
plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('Risque')
plt.ylabel('Rendement')
plt.title('Rendement et risque de 4 actifs: Apple, Microsoft, Amazon, and General Electric')

plt.show()
'''

#returns = np.asmatrix(rendement)

## %%%%%%%%%%%%%%%%%%%%%%%%% Trouver le minimum de risque à un rendement donnée %%%%%%%%%%%%%%%%%%%%%%%%%
def minRisk_givenReturn (cov, defaultRdt):
	sigma = opt.matrix(cov)
	P = sigma
	q = opt.matrix(np.zeros((4, 1)))

	G = opt.matrix((np.concatenate((np.array(-np.array(opt.matrix(p).T)), -np.identity(4)), 0)))
	h = opt.matrix(np.concatenate((-np.ones((1,1))*defaultRdt, np.zeros((4,1))), 0))

	A = opt.matrix(1.0, (1,4))
	b = opt.matrix(1.0)

	solvers.options['show_progress'] = False
	weights = np.asarray(solvers.qp(P, q, G, h, A, b)['x'])

	optRdt = (np.matrix(p.T) * weights).item(0)
	optRsk = sqrt((weights.T * np.matrix(P)) * weights)
	print ("les poids optimal: \n", weights.T)
	print("rendement: ", optRdt);
	print("risque: ", optRsk)

	return weights, optRdt, optRsk

r_min = 0.51
weights, optRdt, optRsk = minRisk_givenReturn (cov, r_min)
plt.plot(*zip([optRsk, optRdt]), marker='o', color='r', ls='')

## %%%%%%%%%%%%%%%%%%%%%%%%% Trouver la frontière efficiente %%%%%%%%%%%%%%%%%%%%%%%%%
def optimal_portfolio(returns):
	#sigma = opt.matrix()

    n = len(returns)

    returns = np.asmatrix(returns)
    numberExpectedReturn = 50
    solvers.options['show_progress'] = False
    
    N = 100
    mus = [numberExpectedReturn**(5.0 * t/N - 1.0) for t in range(N)]
    
    ## Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
   	
    ## Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
	
    ## Calcul de la frontière efficiente utilisant quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
				  
    ## Calcul risques et rendement pour la frontière
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
	
	
    ## Calculate the 2nd degree polynomial of the frontier curve
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
	
    ## Calcul le portefeuille optimal
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

weights, returns, risks = optimal_portfolio(np.asmatrix(rendement))

print("weights: \n", weights)

optimal_rendement = np.matrix(p.T) * weights
print("optimal_rendement: ", optimal_rendement);
print("optimal_risque: ", sqrt((weights.T * np.matrix(cov)) * weights))


plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('Risque')
plt.ylabel('Rendement')
plt.plot(risks, returns, 'y-o')
plt.title('Rendement et risque de 4 actifs: Apple, Microsoft, Amazon, and General Electric')

plt.show()

## %%%%%%%%%%%%%%%%%%%%%%%%% Exporter des graphes de variance %%%%%%%%%%%%%%%%%%%%%%%%%
def export_graph(data, mean, file_save):

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(data)

	N = 300
	rects = ax.patches

	ind = np.arange(N)  
	width = 0.5       

	ax.set_xlim(-width,len(ind))
	#ax.set_ylim(0,40)
	ax.set_ylabel("Valeur de l'actif")
	plt.xlabel('Temps')
	ax.set_title('Variance')

	ax.set_xticks(ind+width)

	#plt.setp(xtickNames, rotation=45, fontsize=5)
	plt.axhline(y=mean, color='red')
	return plt.savefig(file_save + ".png")
	
'''  ### test
export_graph(rendement_apple, rendementAppleMoyen, "variance_rendement_apple")
export_graph(rendement_micro, rendementMicroMoyen, "variance_rendement_microsoft")
export_graph(rendement_amazon, rendementAmazonMoyen, "variance_rendement_amazon")
export_graph(rendement_ge, rendementGEMoyen, "variance_rendement_ge")

'''

