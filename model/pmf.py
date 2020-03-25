# encoding:utf-8
import sys

sys.path.append("..")

from prettyprinter import cpprint, set_default_style
# set_default_style('light')
import numpy as np
from mf import MF
from utility import tools


class PMF(MF):
    """
    docstring for PMF
    原始名称 FunkSVDwithR 
    implement the FunkSVD with regularization
    http://sifter.org/~simon/journal/20061211.html
    """

    def __init__(self):  # 
        super(PMF, self).__init__()
        self.config.gamma = 0.9  # Momentum
        self.config.isEarlyStopping = True
        # self.init_model()

    # def init_model(self):
    # 	super(PMF, self).init_model()

    def train_model(self, k):
        super(PMF, self).train_model(k)
        iteration = 0
        p_delta, q_delta = dict(), dict()
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                pred = self.predict(user, item)
                # pred = tools.sigmoid(pred)
                error = rating - pred  # self.predict(user,item)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]
                # update latent vectors 

                if not u in p_delta:
                    p_delta[u] = np.zeros(self.config.factor)
                if not i in q_delta:
                    q_delta[i] = np.zeros(self.config.factor)

                # 相对于 funk_svd 增加了 lamda 避免过拟合
                p_delta[u] = self.config.lr * (-error * q + self.config.lambdaP * p) + self.config.gamma * p_delta[u]
                q_delta[i] = self.config.lr * (-error * p + self.config.lambdaQ * q) + self.config.gamma * q_delta[i]
                self.P[u] -= p_delta[u]
                self.Q[i] -= q_delta[i]

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum()

            iteration += 1
            if self.isConverged(iteration):
                iteration = self.config.maxIter
                break


if __name__ == '__main__':
    # print(bmf.predict_model_cold_users())
    # coldrmse = bmf.predict_model_cold_users()
    # print('cold start user rmse is :' + str(coldrmse))
    # bmf.show_rmse()

    # 补充计时 、加速测试 、数据集打印 代码
    print("=== START TIMING"); from time import time , sleep; start_time = time()

    rmses = []
    maes = []
    bmf = PMF()
    config = bmf.config
    fold = config.k_fold_num # 原有 fold num 配置

    # 加速测试
    fold = 1 

    # print(bmf.rg.trainSet_u[1])
    for i in range(fold):
        bmf.train_model(i)
        rmse, mae = bmf.predict_model()
        print("current best rmse is %0.5f, mae is %0.5f" % (rmse, mae))
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / fold
    mae_avg = sum(maes) / fold
    print("[%s] the rmses are %s" % (config.dataset_name, rmses))
    print("[%s] the maes are %s" % (config.dataset_name, maes))
    print("the average of rmses in [%s] is %s " % (config.dataset_name, rmse_avg))
    print("the average of maes in  [%s] is %s " % (config.dataset_name, mae_avg))

    ## TIMING END; 
    end_time = time(); print("=== total run minutes: " , (end_time - start_time) / 60 )
    