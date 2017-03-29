The deep method is based on torch framework and you need to use gpu to run the code. Install torch, cutorch, nn, cunn, cudnn before running the code. Just cd in the directory and th train_(conv/fc/lstm).lua to run different models. The data for fc is in './data/pca/' and the data for other two methods is in'./data/mean'.


Use 'python fourier.py' in the 'fourier' directory to run the fourier learning code. You only need the numpy lib. The data is in './data/mean'.

Use 'python SVM.py' in the 'svm' directory to run the svm learning code. You only need the numpy lib. The data is in './data/pca/'