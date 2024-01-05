import BayesianOptimization as BO
import Plotting as Plt
import numpy as np

folderpath = './demo_data'

dimensions = [2, 3, 4]
Y_set = []
for dim in dimensions:
    print(f'Starting {dim} Dimension Campaign')
    res = BO.run(modeltype='GPR', policy='EI', surrogate='trid', noise=0.05,  runlength=50,
                 folderpath=folderpath, startRandSamples=3, dimensions=dim).singleOptimization()

    Y = np.array(res.Y)
    X = np.array(res.X)
    data = np.append(X, Y.reshape(-1,1), axis=1)
    filename = 'run_dim' + str(dim) + '.txt'
    np.savetxt(folderpath + '/' + filename, data, fmt='%10.9f')

    Y_set.append(res.Y)

Plt.saveDimensionLinePlots(Y_set, dimensions=dimensions, surrogate='trid',
                           filepath=folderpath + '/plot.png')