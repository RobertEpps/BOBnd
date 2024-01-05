import numpy as np
import matplotlib.pyplot as plt
import nDimensionalFunctions as ND

def save2DSurfPlot(stepsize=0.1, surrogate='ackley', folderpath=''):
    func = getattr(ND, surrogate)

    x = np.arange(0, 1, stepsize)
    X, Y = np.meshgrid(x, x)

    Z = np.zeros((len(X), len(Y)))

    for ii in range(len(X)):
        for jj in range(len(X[ii])):
            Z[ii,jj] = func([X[ii,jj], Y[ii,jj]], normY=True)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap='gist_stern')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Response')
    plt.title(surrogate)
    plt.savefig(folderpath+'\\'+surrogate+'.png', dpi=400)
    plt.close()
    np.savetxt(folderpath+'\\'+surrogate+'.txt', Z, delimiter=',', fmt='%1.6f')

def save2DSurfPlotwNoise(stepsize=0.1, surrogate='ackley', folderpath='', noise=0):
    func = getattr(ND, surrogate)

    x = np.arange(0, 1, stepsize)
    X, Y = np.meshgrid(x, x)

    Z = np.zeros((len(X), len(Y)))

    for ii in range(len(X)):
        for jj in range(len(X[ii])):
            Z[ii,jj] = func([X[ii,jj], Y[ii,jj]], normY=True, noise=noise)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap='gist_stern')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Response')
    plt.title(surrogate)
    plt.savefig(folderpath+'\\'+surrogate+'Noise' + str(noise*100) + '.png', dpi=400)
    plt.close()

def saveDimensionLinePlots(Y, dimensions, surrogate='ackley', filepath=''):
    fig, ax = plt.subplots()
    legendvals = [str(dim)+'D' for dim in dimensions]
    x = np.arange(1, len(Y[0])+1)
    for ii in range(len(legendvals)):
        plt.plot(x, Y[ii], label=legendvals[ii])
    plt.xlabel('Sample Number')
    plt.ylabel('Response')
    plt.title(surrogate)
    plt.legend()
    plt.savefig(filepath + '.png', dpi=400)
    plt.close()

    fig, ax = plt.subplots()
    legendvals = [str(dim) + 'D' for dim in dimensions]
    x = np.arange(1, len(Y[0]) + 1)
    for ii in range(len(legendvals)):
        besty = 0
        bestY = []
        for y in Y[ii]:
            if y > besty:
                besty = y
            bestY.append(besty)


        plt.plot(x, bestY, label=legendvals[ii])
    plt.xlabel('Sample Number')
    plt.ylabel('Best Response')
    plt.title(surrogate)
    plt.legend()
    plt.savefig(filepath + '_best.png', dpi=400)
    plt.close()

def saveNoiseLinePlots(Y, noises, beleifmodel='GPR', dimension=3, randsampnum=10, surrogate='ackley', filepath='', ylabel='Best Response'):
    fig, ax = plt.subplots()
    legendvals = [str(noise) + ' Noise' for noise in noises]
    x = np.arange(1, len(Y[0])+1)
    for ii in range(len(legendvals)):
        plt.plot(x, Y[ii], label=legendvals[ii])
    plt.xlabel('Sample Number')
    plt.ylabel(ylabel)
    plt.title(surrogate + ', ' + beleifmodel +  ', ' + str(dimension)+'D, ' + str(randsampnum) + ' Random Samples')
    plt.legend()
    plt.savefig(filepath + '.png', dpi=400)
    plt.close()

def saveBeliefLinePlots(Y, beliefmodels, noise=0, dimension=3, randsampnum=10, surrogate='ackley', filepath='', ylabel='Best Response'):
    fig, ax = plt.subplots()
    legendvals = [beliefmodel for beliefmodel in beliefmodels]
    x = np.arange(1, len(Y[0])+1)
    for ii in range(len(legendvals)):
        plt.plot(x, Y[ii], label=legendvals[ii])
    plt.xlabel('Sample Number')
    plt.ylabel(ylabel)
    plt.title(surrogate + ', ' + str(noise) + ' Noise, ' + str(dimension)+'D, ' + str(randsampnum) + ' Random Samples')
    plt.legend()
    plt.savefig(filepath + '.png', dpi=400)
    plt.close()

def saveSurrLinePlots(Y, surrogates, beliefmodel='GPR', noise=0, dimension=3, randsampnum=10, surrogate='ackley', filepath='', ylabel='Best Response'):
    fig, ax = plt.subplots()
    legendvals = [surrogate for surrogate in surrogates]
    x = np.arange(1, len(Y[0])+1)
    for ii in range(len(legendvals)):
        plt.plot(x, Y[ii], label=legendvals[ii])
    plt.xlabel('Sample Number')
    plt.ylabel(ylabel)
    plt.title(beliefmodel + ', ' + str(noise) + ' Noise, ' + str(dimension)+'D, ' + str(randsampnum) + ' Random Samples')
    plt.legend()
    plt.savefig(filepath + '.png', dpi=400)
    plt.close()