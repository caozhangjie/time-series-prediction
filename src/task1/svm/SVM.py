#author: Zhangjie Cao

from numpy import *  
import time  

class SVM:  
    def __init__(self, data, labels, C, terminate_c):  
        self.train_data = data
        self.train_label = labels 
        self.b = 0
        self.terminate_c = terminate_c
        self.alphas = mat(zeros((self.train_data.shape[0] , 1))) 
        self.C = C 
        self.error_cache = mat(zeros((self.train_data.shape[0] , 2)))  
        self.kernel_mat = mat(zeros((self.train_data.shape[0] , self.train_data.shape[0])))
        for i in xrange(self.train_data.shape[0] ):  
            self.kernel_mat[:, i] = self.train_data * (self.train_data[i, :]).T  
  
  
def geta(svm, alpha_i, error_i):  
    svm.error_cache[alpha_i] = [1, error_i]
    candidateAlphaList = nonzero(svm.error_cache[:, 0].A)[0]  
    maxStep = 0; alpha_j = 0; error_j = 0   
    if len(candidateAlphaList) > 1:  
        for alpha_k in candidateAlphaList:  
            if alpha_k == alpha_i:   
                continue  
            error_k = float(multiply(svm.alphas, svm.train_label).T * svm.kernel_mat[:, alpha_k] + svm.b) - float(svm.train_label[alpha_k])  
            if abs(error_k - error_i) > maxStep:  
                maxStep = abs(error_k - error_i)  
                alpha_j = alpha_k  
                error_j = error_k  
    else:             
        alpha_j = alpha_i  
        while alpha_j == alpha_i:  
            alpha_j = int(random.uniform(0, svm.train_data.shape[0]))  
        error_j = float(multiply(svm.alphas, svm.train_label).T * svm.kernel_mat[:, alpha_j] + svm.b) - float(svm.train_label[alpha_j])
      
    return alpha_j, error_j  
  
  
def train_in(svm, alpha_i):  
    error_i = float(multiply(svm.alphas, svm.train_label).T * svm.kernel_mat[:, alpha_i] + svm.b) - float(svm.train_label[alpha_i]) 
    if (svm.train_label[alpha_i] * error_i < -svm.terminate_c) and (svm.alphas[alpha_i] < svm.C) or (svm.train_label[alpha_i] * error_i > svm.terminate_c) and (svm.alphas[alpha_i] > 0):
        alpha_j, error_j = geta(svm, alpha_i, error_i)  
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()
        if svm.train_label[alpha_i] != svm.train_label[alpha_j]:  
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])  
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])  
        else:  
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)  
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])  
        if L == H:  
            return 0
        eta = 2.0 * svm.kernel_mat[alpha_i, alpha_j] - svm.kernel_mat[alpha_i, alpha_i] - svm.kernel_mat[alpha_j, alpha_j]  
        if eta >= 0:  
            return 0
        svm.alphas[alpha_j] -= svm.train_label[alpha_j] * (error_i - error_j) / eta
        if svm.alphas[alpha_j] > H:  
            svm.alphas[alpha_j] = H  
        if svm.alphas[alpha_j] < L:  
            svm.alphas[alpha_j] = L    
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:  
            svm.error_cache[alpha_j] = [1, float(multiply(svm.alphas, svm.train_label).T * svm.kernel_mat[:, alpha_j] + svm.b) - float(svm.train_label[alpha_j])]
            return 0
        svm.alphas[alpha_i] += svm.train_label[alpha_i] * svm.train_label[alpha_j] * (alpha_j_old - svm.alphas[alpha_j])
        b1 = svm.b - error_i - svm.train_label[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernel_mat[alpha_i, alpha_i] - svm.train_label[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernel_mat[alpha_i, alpha_j]  
        b2 = svm.b - error_j - svm.train_label[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernel_mat[alpha_i, alpha_j] - svm.train_label[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernel_mat[alpha_j, alpha_j]  
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):  
            svm.b = b1  
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):  
            svm.b = b2  
        else:  
            svm.b = (b1 + b2) / 2.0
        svm.error_cache[alpha_j] = [1, float(multiply(svm.alphas, svm.train_label).T * svm.kernel_mat[:, alpha_j] + svm.b) - float(svm.train_label[alpha_j])]
        svm.error_cache[alpha_i] = [1, float(multiply(svm.alphas, svm.train_label).T * svm.kernel_mat[:, alpha_i] + svm.b) - float(svm.train_label[alpha_i])]
  
        return 1  
    else:  
        return 0


def train(train_data, train_label, C, terminate_c, iteration):  
    start_time = time.time()  
    svm = SVM(train_data, train_label, C, terminate_c)  
       
    entire_set = True  
    pairs_changed = 0  
    iter_num = 0  

    while (iter_num < iteration) and ((pairs_changed > 0) or entire_set):  
        pairs_changed = 0  
        if entire_set:  
            for i in xrange(svm.train_data.shape[0]):  
                pairs_changed += train_in(svm, i)   
            iter_num += 1
        else:  
            non_bound = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]  
            for i in non_bound:
                pairs_changed += train_in(svm, i)  
            iter_num += 1  
        if entire_set:  
            entire_set = False  
        elif pairs_changed == 0:  
            entire_set = True  
  
    print 'training time: %fs' % (time.time() - start_time)  
    return svm  


def test(svm, test_data, test_label):   
    sv_index = nonzero(svm.alphas.A > 0)[0]
    sv = svm.train_data[sv_index]  
    sv_label = svm.train_label[sv_index]  
    sv_alphas = svm.alphas[sv_index]  
    match = 0  
    for i in xrange(test_data.shape[0]):  
        predict = (sv * (test_data[i, :]).T).T * multiply(sv_label, sv_alphas) + svm.b  
        if sign(predict) == sign(test_label[i]):  
            match += 1  
    return float(match) / test_data.shape[0]


if __name__ == "__main__":
    train_data = []  
    train_label = []  
    test_data = []
    test_label = []
    file_train = open('../data/pca/train.csv')  
    for line in file_train.readlines():  
        line = line.strip().split(',')
        train_data.append([])
        for val in line:
            train_data[-1].append(float(val))
        train_label.append(float(line[-1]))  
    train_data = mat(train_data)  
    train_label = mat(train_label).T 
    
    file_test = open('../data/pca/test.csv')  
    for line in file_test.readlines():  
        line = line.strip().split(',')
        test_data.append([])
        for val in line:
            test_data[-1].append(float(val))
        test_label.append(float(line[-1]))  
     
    test_data = mat(test_data)
    test_label = mat(test_label).T
    
    svm = train(train_data, train_label, 0.6, 0.001, 50)  
    accuracy = test(svm, test_data, test_label)     
    print 'The classify accuracy is: %.3f%%' % (accuracy * 100)  