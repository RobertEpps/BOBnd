def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sklearn as sk
import sklearn.gaussian_process as skg
import sklearn.neural_network as sknn
import sklearn.ensemble as sken
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import *
from sklearn.metrics import mean_squared_error

import numpy as np
from scipy.stats import norm
from scipy import optimize

import Plotting
import nDimensionalFunctions as ND

def randomX(dim):
    randarray = np.random.rand(dim)
    return [ii for ii in randarray]

class run:
    def __init__(self, modeltype='GPR', policy='UCB', surrogate='ackley', noise=0, dimensions=2, runlength = 25,
                 folderpath='', startRandSamples=5):
        self.modeltype = modeltype
        self.policy = policy
        self.surrogate = surrogate
        self.runlength = runlength
        self.filepath = folderpath + '\\Opt ' + surrogate + 'N' + str(noise) + '_' + modeltype + '_' + policy
        self.nStart = startRandSamples
        self.dim = dimensions
        self.noise = noise

        self.surrfunc = getattr(ND, surrogate)

    def buildStartRand(self):
        self.X = []
        self.Y = []
        self.MSE = []

        self.X_S = []
        self.Y_S = []

        for ii in range(self.nStart):
            tempX = randomX(self.dim)
            self.X.append(tempX)

            tempY = self.surrfunc(tempX, self.noise)
            self.Y.append(tempY)

            self.MSE.append(0)

    def singleOptimization(self):
        self.buildStartRand()

        while len(self.Y) < self.runlength:
            self.model = BeliefModels(self.X, self.Y).modelPicker(self.modeltype)
            tempX = Minimization(self.model, self.modeltype, self.policy, self.dim, Ytrain=self.Y).basefminSearch()
            tempY = self.surrfunc(tempX, self.noise)
            tempMSE = self.getMSE()

            self.X.append(tempX)
            self.Y.append(tempY)
            self.MSE.append(tempMSE)

            print(f'Experiment {len(self.Y)}')

        return self

    def dimensionScreen(self, dimensions=[2, 3, 4]):
        Y = []
        for dim in dimensions:
            self.dim = dim
            tempY = self.singleOptimization()
            Y.append(tempY)
        Plotting.saveDimensionLinePlots(Y, dimensions, self.surrogate, self.filepath)

    def getMSE(self, nTestSamples=100):
        if self.modeltype == 'RND':
            return 0

        # testX = []
        # actualY = []
        # predictedY = []
        SE = []
        for ii in range(nTestSamples):
            tempX = randomX(self.dim)
            # testX.append(tempX)

            tempYact = self.surrfunc(tempX, self.noise)
            # actualY.append(tempYact)

            tempYpred, tempYErr = Prediction(tempX, self.model).predictYYErr(self.modeltype)
            # predictedY.append(tempYpred)

            SE.append((tempYpred-tempYact) ** 2)
        return np.mean(SE)


class BeliefModels:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.dim = len(X[0])

    def modelPicker(self, modeltype):
        if modeltype == 'RND':
            model = []
        elif modeltype == 'GPR':
            model = self.gaussianProcessRegression()
        elif modeltype == 'GPR_EGS':
            model = self.gaussianProcessRegressionExhaustiveGridSearch()
        elif modeltype == 'MLPR':
            model = self.mLPRegression()
        elif modeltype == 'BRMLPR_EGS':
            best_estimator = self.model_mLPRegressionExhaustiveGridSearch()
            model = self.baggingRegressor(estimator=best_estimator)
        elif modeltype == "BRSVR":
            model = self.baggingRegressor(estimator = SVR())
        elif modeltype == 'BRSVR_EGS':
            best_estimator = self.model_SVRExhaustiveGridSearch()
            model = self.baggingRegressor(estimator=best_estimator)
        elif modeltype == "BRMLP":
            model = self.baggingRegressor(estimator = sknn.MLPRegressor())
        elif modeltype == "BRDTR":
            model = self.baggingRegressor(estimator = sk.tree.DecisionTreeRegressor())
        elif modeltype == 'BRDTR_EGS':
            best_estimator = self.model_DTRExhaustiveGridSearch()
            model = self.baggingRegressor(estimator=best_estimator)
        elif modeltype == 'NGBR':
            model = self.naturalGradientBoostedTreeRegressor()
        elif modeltype == 'WV_MSE':
            model = self.weightedVoteEnsembleMSE()
        elif modeltype == 'WV_MSE_EGS':
            model = self.weightedVoteEnsembleMSE_EGS()
        return model

    def gaussianProcessRegression(self):
        model = skg.GaussianProcessRegressor(alpha=0.01)
        model.fit(self.X, self.Y)
        return model

    def gaussianProcessRegressionExhaustiveGridSearch(self):
        param_grid = [
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [DotProduct(sigma_0) for sigma_0 in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [RBF(l) for l in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [Matern(l, nu=0.5) for l in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [Matern(l, nu=1.5) for l in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [Matern(l, nu=2.5) for l in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [RationalQuadratic(l, alpha=1e-4) for l in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [RationalQuadratic(l, alpha=1) for l in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [RationalQuadratic(l, alpha=1e4) for l in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [ExpSineSquared(l, periodicity=1e-4) for l in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [ExpSineSquared(l, periodicity=1) for l in np.logspace(-4, 4, 3)]},
            {'alpha': [a for a in np.logspace(-5, 1, 3)], 'kernel': [ExpSineSquared(l, periodicity=1e4) for l in np.logspace(-4, 4, 3)]}
        ]
        GSmodel = GridSearchCV(skg.GaussianProcessRegressor(), param_grid)
        GSmodel.fit(self.X, self.Y)
        model = GSmodel.best_estimator_
        model.fit(self.X, self.Y,)
        return model

    def mLPRegression(self):
        model = sknn.MLPRegressor()
        model.fit(self.X, self.Y)
        return model

    def model_mLPRegressionExhaustiveGridSearch(self):
        param_grid = {"activation": ["identity", "logistic", "tanh", "relu"],
                      "solver": ["lbfgs"], "alpha": [a for a in np.logspace(-6, 0, 3)]}
        GSmodel = GridSearchCV(sknn.MLPRegressor(), param_grid)
        GSmodel.fit(self.X, self.Y)
        model = GSmodel.best_estimator_
        return model

    def model_SVRExhaustiveGridSearch(self):
        param_grid = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'], "gamma": [a for a in np.logspace(-3, 0, 4)],
                      "C": [a for a in np.logspace(-1, 2, 4)]}
        GSmodel = GridSearchCV(SVR(), param_grid)
        GSmodel.fit(self.X, self.Y)
        model = GSmodel.best_estimator_
        return model

    def model_DTRExhaustiveGridSearch(self):
        param_grid = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                      'splitter': ['best', 'random'], 'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 8, 32]}
        GSmodel = GridSearchCV(sk.tree.DecisionTreeRegressor(), param_grid)
        GSmodel.fit(self.X, self.Y)
        model = GSmodel.best_estimator_
        return model

    def baggingRegressor(self, estimator=SVR()):
        self.estimator=SVR
        model = sken.BaggingRegressor(estimator=estimator)
        model.fit(self.X, self.Y)
        return model

    def weightedVoteEnsembleMSE(self):
        ensemble = [skg.GaussianProcessRegressor(),
                    self.baggingRegressor(estimator = SVR()),
                    self.baggingRegressor(estimator = sknn.MLPRegressor()),
                    self.baggingRegressor(estimator = sk.tree.DecisionTreeRegressor())]
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2)

        if len(Y_test) < 2:
            weights = [0.25, 0.25, 0.25, 0.25]
        else:
            recipMSE = []
            for ii, member in enumerate(ensemble):
                member.fit(X_train, Y_train)
                Y_pred = member.predict(X_test)
                recipMSE.append(1/mean_squared_error(Y_test, Y_pred))
            weights = [x/sum(recipMSE) for x in recipMSE]
        model = [ensemble, weights]
        print(weights)
        return model

    def weightedVoteEnsembleMSE_EGS(self):
        ensemble = [BeliefModels(self.X, self.Y).modelPicker('GPR_EGS'),
                    BeliefModels(self.X, self.Y).modelPicker('BRMLPR_EGS'),
                    BeliefModels(self.X, self.Y).modelPicker('BRDTR_EGS'),
                    BeliefModels(self.X, self.Y).modelPicker('BRSVR_EGS')]
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.2)

        if len(Y_test) < 2:
            weights = [0.25, 0.25, 0.25, 0.25]
        else:
            recipMSE = []
            for ii, member in enumerate(ensemble):
                member.fit(X_train, Y_train)
                Y_pred = member.predict(X_test)
                recipMSE.append(1/mean_squared_error(Y_test, Y_pred))
            weights = [x/sum(recipMSE) for x in recipMSE]
        model = [ensemble, weights]
        print(weights)
        return model

class DecisionPolicies:
    def __init__(self, model, modeltype, policy):
        self.model = model
        self.modeltype = modeltype
        self.policy = policy

    def policyPicker(self, X, Ymax=0):
        if self.policy == 'UCB':
            self.Y, self.YErr = Prediction(X, self.model).predictYYErr(self.modeltype)
            value = self.upperConfidenceBounds()
        elif self.policy == 'EI':
            self.Y, self.YErr = Prediction(X, self.model).predictYYErr(self.modeltype)
            z = (self.Y - Ymax) / self.YErr
            value = (self.Y - Ymax) * norm.cdf(z) + self.YErr * norm.pdf(z)
        return -value

    def upperConfidenceBounds(self):
        lam = 1 / (2 ** 0.5)
        value = self.Y + lam * self.YErr
        return value

class Prediction:
    def __init__(self, X, model):
        self.X = X
        self.model = model

    def predictYYErr(self, modeltype):
        self.modeltype = modeltype
        if self.modeltype in ['GPR', 'GPR_EGS']:
            result = self.model.predict([self.X], return_std=True)
            Y = result[0]
            YErr = result[1]

        elif self.modeltype == "MLPR":
            result = self.model.predict([self.X])

        elif self.modeltype in ['BRSVR', 'BRMLP', 'BRDTR', 'BRMLPR_EGS', 'BRSVR_EGS', 'BRDTR_EGS']:
            result = self.model.predict([self.X])
            Y = result[0]
            resultmembers = [x.predict([self.X]) for x in self.model.estimators_]
            YErr = np.std(resultmembers)

        elif self.modeltype == 'NGBR':
            Y = self.model.predict([self.X])
            YDist = self.model.pred_dist([self.X])
            YErr = YDist.params['scale']

        elif self.modeltype == 'WV_MSE':
            ensemble = self.model[0]
            weights = self.model[1]
            membertypes = ['GPR', 'BRSVR', 'BRMLP', 'BRDTR']
            Y = 0
            YErr = 0
            for ii, membertype in enumerate(membertypes):
                tempY, tempYErr = Prediction(self.X, ensemble[ii]).predictYYErr(membertype)
                Y = tempY*weights[ii]
                YErr = tempYErr * weights[ii]

        elif self.modeltype == 'WV_MSE_EGS':
            ensemble = self.model[0]
            weights = self.model[1]
            membertypes = ['GPR_EGS', 'BRSVR_EGS', 'BRMLPR_EGS', 'BRDTR_EGS']
            Y = 0
            YErr = 0
            for ii, membertype in enumerate(membertypes):
                tempY, tempYErr = Prediction(self.X, ensemble[ii]).predictYYErr(membertype)
                Y = tempY*weights[ii]
                YErr = tempYErr * weights[ii]

        return Y, YErr

class Minimization:
    def __init__(self, model, modeltype, policy, dim, Ytrain):
        self.model = model
        self.modeltype = modeltype
        self.policy = policy
        self.dim = dim
        self.Ytrain = Ytrain

    def basefminSearch(self):
        if self.modeltype == 'RND':
            return np.random.rand(self.dim)

        Ymax = np.max(self.Ytrain)
        f = lambda X: DecisionPolicies(self.model, self.modeltype, self.policy).policyPicker(X, Ymax)

        bnds = []
        for ii in range(self.dim):
            bnds.append((0, 1))

        result = optimize.minimize(f, np.random.rand(self.dim), method='Nelder-Mead', bounds=bnds)
        newX = result.x
        return newX