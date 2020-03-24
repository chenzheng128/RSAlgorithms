# encoding:utf-8
import sys
sys.path.append("/home/songxuan/python/RSA/")

import time as t
import numpy as np
from mf import MF
from reader.trust import TrustGetter


class SocialRec(MF):
    """
    docstring for SocialRec

    Ma H, Yang H, Lyu M R, et al. Sorec: social recommendation using probabilistic matrix factorization[C]//Proceedings of the 17th ACM conference on Information and knowledge management. ACM, 2008: 931-940.

    """

    def __init__(self):
        super(SocialRec, self).__init__()
        # self.config.lr=0.0001
        self.config.alpha = 0.1
        self.config.lambdaZ = 0.01
        self.tg = TrustGetter()
        # self.init_model()

    def init_model(self, k):
        super(SocialRec, self).init_model(k)
        self.Z = np.random.rand(self.rg.get_train_size()[0], self.config.factor) / (
                self.config.factor ** 0.5)  # latent user social matrix

    def train_model(self, k):
        super(SocialRec, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            # tempP=np.zeros((self.rg.get_train_size()[0], self.config.factor))
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]

                followees = self.tg.get_followees(user)
                zs = np.zeros(self.config.factor)
                for followee in followees:
                    if self.rg.containsUser(user) and self.rg.containsUser(followee):
                        vminus = len(self.tg.get_followers(followee))  # ~ d - (k)
                        uplus = len(self.tg.get_followees(user))  # ~ d + (i)
                        import math
                        try:
                            weight = math.sqrt(vminus / (uplus + vminus + 0.0))
                        except ZeroDivisionError:
                            weight = 1
                        zid = self.rg.user[followee]
                        z = self.Z[zid]
                        err = weight - z.dot(p)
                        self.loss += err ** 2
                        zs += -1.0 * err * p
                        self.Z[zid] += self.config.lr * (self.config.alpha * err * p - self.config.lambdaZ * z)

                self.P[u] += self.config.lr * (error * q - self.config.alpha * zs - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + self.config.lambdaZ * (self.Z * self.Z).sum()

            iteration += 1
            if self.isConverged(iteration):
                break


if __name__ == '__main__':
    # 计时 、加速测试 、数据集打印 代码
    print("=== START TIMING"); from time import time , sleep; start_time = time()

    rmses = []
    maes = []
    tcsr = SocialRec()
    config = tcsr.config
    fold = config.k_fold_num
    
    # 加速测试
    # fold = 1 
    # config.lr = 0.02
    # config.threshold = 1000 # delta_loss 低于此值时结束

    # print(bmf.rg.trainSet_u[1])
    for i in range(fold):
        print('the %dth cross validation training' % i)
        tcsr.train_model(i)
        rmse, mae = tcsr.predict_model()
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / fold
    mae_avg = sum(maes) / fold
    print("[%s] the rmses are %s" % (config.dataset_name, rmses))
    print("[%s] the maes are %s" % (config.dataset_name, rmses))
    print("the average of rmses in [%s] is %s " % (config.dataset_name, rmse_avg))
    print("the average of maes in  [%s] is %s " % (config.dataset_name, mae_avg))
    

    ## TIMING END; 
    end_time = time(); print("=== total run minutes: " , (end_time - start_time) / 60 )