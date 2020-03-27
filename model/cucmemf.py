# encoding:utf-8
import sys
import os
sys.path.append("..")
import numpy as np
from mf import MF
from prettyprinter import cpprint
from collections import defaultdict
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp
from utility import util
from utility.sim.movielens import SimBase
from utility.sim.douban_movie import SimDoubanMovie

class CUCMEMF(MF):
    """
    docstring for CUCMEMF: CUC Multivariate Meta Matrix Fraction
    user_near_num=50 # 用户近邻数量
    item_near_num=50 # 物品近邻数量
    load_save_sim=False # 运行模型一次后，可加载原有保存的 sim pickle 数据，便于加快测试；需要至少执行一次，才能设置为 True
    """

    def __init__(self, user_near_num=50, item_near_num=50, load_save_sim=False):
        """
        user_near_num=50 # 用户近邻数量
        item_near_num=50 # 物品近邻数量
        load_save_sim=False # 运行模型一次后，可加载原有保存的 sim pickle 数据，便于加快测试；需要至少执行一次，才能设置为 True
        """

        super(CUCMEMF, self).__init__()
        # self.config.lr=0.001
        self.config.lambdaU = 0.002
        self.config.lambdaI = 0.001

        self.config.lambdaP = 0.02
        self.config.lambdaQ = 0.03
        self.config.lambdaB = 0.01

        self.user_near_num = user_near_num
        self.item_near_num = item_near_num
        self.load_save_sim = load_save_sim
        # self.init_model()

    def init_model(self, k):
        super(CUCMEMF, self).init_model(k)
        np.random.seed(seed=self.config.random_state)
        self.Bu = np.random.rand(self.rg.get_train_size()[0])  # bias value of user
        np.random.seed(seed=self.config.random_state)
        self.Bi = np.random.rand(self.rg.get_train_size()[1])  # bais value of item

        self.build_user_item_sim_CF(k)

    # construct the u-u,i-i similarity matirx and their's k neighbors
    def build_user_item_sim_CF(self, k):
        """
        k - foldnum
        此函数重构至 SimMovieLens 中；
        """

        sim_model = SimBase()
        # if self.config.dataset_name == "lastfm":
        #     sim_model = SimBase
        #     print ("loading SimClass %s ..." % SimBase.__name__)
        # else:
        #     # TODO
        #     raise Exception ("not suport %s dataset yet" % self.config.dataset_name)
        
        # 建立 user item 相似度数据
        sim_model.build_user_item_sim(k, 
            user_near_num=self.user_near_num, 
            item_near_num=self.item_near_num, 
            load_save_sim=self.load_save_sim
            )
        self.sim_model = sim_model

    def train_model(self, k):
        super(CUCMEMF, self).train_model(k) # 此时会 加载 init_mode()
        print('training model...')
        iteration = 0
        # faflag=True
        while iteration < self.config.maxIter:
            self.loss = 0
            self.u_near_total_dict = defaultdict()
            self.i_near_total_dict = defaultdict()
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line
                u = self.rg.user[user]
                i = self.rg.item[item]

                error = rating - self.predict(user, item)
                self.loss += error ** 2
                p, q = self.P[u], self.Q[i]

                # 从 sim_model 中 获取 k neighbors of user and item
                matchUsers = self.sim_model.user_k_neibor[user]
                matchItems = self.sim_model.item_k_neibor[item]

                u_near_sum, u_near_total, s = np.zeros((self.config.factor)), 0.0, 0.0
                for suser in matchUsers.keys():
                    near_user, sim_value = suser, matchUsers[suser]
                    if sim_value != 0.0:
                        s += sim_value
                        pn = self.P[self.rg.user[near_user]]
                        u_near_sum += sim_value * (pn - p)
                        u_near_total += sim_value * ((pn - p).dot(pn - p))
                if s != 0.0:
                    u_near_sum /= s

                i_near_sum, i_near_total, ss = np.zeros((self.config.factor)), 0.0, 0.0
                for sitem in matchItems:
                    near_item, sim_value = sitem, matchItems[sitem]
                    if sim_value != 0.0:
                        ss += sim_value
                    qn = self.Q[self.rg.item[near_item]]
                    i_near_sum += sim_value * (qn - q)
                    i_near_total += sim_value * ((qn - q).dot(qn - q))
                if ss != 0.0:
                    i_near_sum /= ss

                if u not in self.u_near_total_dict:
                    self.u_near_total_dict[u] = u_near_total
                if i not in self.i_near_total_dict:
                    self.i_near_total_dict[i] = i_near_total

                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * self.Bu[u])
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * self.Bi[i])

                self.P[u] += self.config.lr * (error * q - self.config.lambdaU * u_near_sum - self.config.lambdaP * p)
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaI * i_near_sum - self.config.lambdaQ * q)

                self.loss += 0.5 * (self.config.lambdaU * u_near_total + self.config.lambdaI * i_near_total)

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + self.config.lambdaB * ((self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum())

            iteration += 1
            if self.isConverged(iteration):
                break

    # test cold start users among test set
    def predict_model_cold_users_improved(self):
        res = []
        for user in self.rg.testColdUserSet_u.keys():
            for item in self.rg.testColdUserSet_u[user].keys():
                rating = self.rg.testColdUserSet_u[user][item]
                pred = self.predict_improved(user, item)
                # denormalize
                pred = denormalize(pred, self.config.min_val, self.config.max_val)
                pred = self.checkRatingBoundary(pred)
                res.append([user, item, rating, pred])
        rmse = Metric.RMSE(res)
        return rmse

def print_hyper_params(model):
    """
    打印本模型的 超参
    """
    print("%s.py HYPER_PARAMS lamdaP = %s, lamdaQ = %s config.factor = %s " % (model.__class__.__name__,  
            model.config.lambdaP, model.config.lambdaQ, model.config.factor))
    print("%s.py HYPER_PARAMS lamdaB = %s, lamdaU = %s lamdaI = %s " % (model.__class__.__name__,  
            model.config.lambdaB, config.lambdaU, config.lambdaI))
    print ("%s.py HYPER_PARAMS user_near_num = %s item_near_num = %s " % (model.__class__.__name__,  
            model.user_near_num,  model.item_near_num))

if __name__ == '__main__':


    print("=== START TIMING"); from time import time , sleep; start_time = time()

    rmses = []
    maes = []

    # 2 个 邻居 超参定义
    usernum=50
    itemnum=50
    # 第一次运行时应执行此句，生成可加载的缓存
    # model = CUCMEMF(user_near_num=usernum, item_near_num=itemnum)
    # 第二次运行，在运行上面语句 1 次后，即可加载缓存调试, 加载已经保存好的 sim 数据
    model = CUCMEMF(user_near_num=usernum, item_near_num=itemnum, load_save_sim=True)
    config = model.config

    # 设置调试数据集
    # config.dataset_name = "db"

    fold = 5 
    # 加速调试
    fold = 1 

    print("### 参数影响性分析： 超参 调整 ")
    config.isEarlyStopping = True
    config.threshold = 1 # 判断收敛 
    config.maxIter = 100
    # config.lambdaP = 0.02
    # config.lambdaQ = 0.002 # 正则化项目 
    # # model.item_near_num = usernum # item 近邻
    # # model.user_near_num = itemnum # user 近邻
    print_hyper_params(model) # 打印 超参 信息

    # print(bmf.rg.trainSet_u[1])
    for i in range(fold):
        print('the %dth cross validation training' % i)
        model.train_model(i)
        rmse, mae = model.predict_model()
        rmses.append(rmse)
        maes.append(mae)
    
    print_hyper_params(model)

    rmse_avg = sum(rmses) / fold
    mae_avg = sum(maes) / fold
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)

    print("the average of rmses in %s is %s " % (config.dataset_name, rmse_avg))
    print("the average of maes in %s is %s " % (config.dataset_name, mae_avg))

    ## TIMING END; 
    end_time = time(); print("=== total run minutes: " , (end_time - start_time) / 60 )