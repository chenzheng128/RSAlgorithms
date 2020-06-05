# encoding:utf-8
"""
计算并维护 social 与 meta 的 user , item 相似度

DONE:
 * 依据 user 的某一附加属性（社交、 group 等）， 计算 user 相似度
 * 依据 item 的某一附加属性（type、actor 等）， 计算 item 相似度
TODO:  待完成
1. 抽象并继承 sim_base 父类

"""

import sys,os
sys.path.append("../..")
from math import sqrt
import numpy as np
# from mf import MF
from prettyprinter import cpprint
from collections import defaultdict
from random import randint
from random import shuffle,choice
import gensim.models.word2vec as w2v
from prettyprinter import cpprint
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp, cosine, jaccard_sim
from utility import util
from reader.rating import RatingGetter
from reader.meta import MetaGetter
from configx.configx import ConfigX
from tqdm import tqdm

class SimGe():

    def __init__(self):
        super(SimGe, self).__init__()
        self.config = ConfigX()

        self.config.walkCount = 30
        self.config.walkLength = 20
        self.config.walkDim = 20
        self.config.winSize = 5
        self.config.topK = 50

        self.user_sim = SimMatrix()
        self.item_sim = SimMatrix()
        self.user_k_neibor = defaultdict(dict)
        self.item_k_neibor = defaultdict(dict)
        
        
    def check_dataset(self):
        super(SimGe, self).check_dataset()
        # if config.dataset_name != 'db':
        #     print("WARN: 注意 config.dataset_name 未设置为 'db' - douban movie")
        # # config.dataset_name = 'ml'
        # # sys.exit()

    def build_user_item_sim(self, kfold, user_near_num=50, item_near_num=50, load_save_sim=False):
        """
        获取 user 与 item 的相似性
        load_save_sim: 加载原有保存数据，提高测试速度
        """

        # 目前仅使用一个 SimCF 
        # TODO: 下一步要混合多个Sim
        self.build_user_item_sim_CF(kfold, 
            user_near_num=user_near_num, 
            item_near_num=item_near_num,
            load_save_sim=load_save_sim
            )

    def build_user_item_sim_CF(self, kfold, user_near_num=50, item_near_num=50, load_save_sim=False):

        self.rg = RatingGetter(kfold) 
        self.mg = MetaGetter(kfold) 

        from collections import defaultdict

        
        # compute item-item similarity matrix
        print('构建 item-item 相似度矩阵  ...')
        if load_save_sim:
            self.item_sim = util.load_data('../data/sim/%s_08_ii_gemf_cv0.pkl' % self.config.dataset_name)
        else:
            # 封装 item 相似度计算
            self.item_sim = self.mg.getSimMatrix(jaccard_sim)
            util.save_data(self.item_sim, '../data/sim/%s_08_ii_gemf_cv0.pkl' % self.config.dataset_name)

        # compute the k neighbors of item
        if load_save_sim:
            self.item_k_neibor = util.load_data(
                '../data/neibor/%s_08_ii_%s_neibor_gemf_cv0.pkl'  % (self.config.dataset_name, item_near_num))
        for item in self.mg.item:
            matchItems = sorted(self.item_sim[item].items(), key=lambda x: x[1], reverse=True)[
                         :item_near_num]
            matchItems = matchItems[:item_near_num]
            self.item_k_neibor[item] = dict(matchItems)
        util.save_data(self.item_k_neibor,
                       '../data/neibor/%s_08_ii_%s_neibor_gemf_cv0.pkl' % (self.config.dataset_name, item_near_num))

        # compute user-user similarity matrix
        print('构建 user-user 相似度矩阵 ...')
        if load_save_sim:
        # if True:
            self.user_sim = util.load_data('../data/sim/%s_08_uu_gemf_cv0.pkl' % self.config.dataset_name)
        else:
            itemNet = {}
            for item in self.rg.trainSet_i:
                if len(self.rg.trainSet_i[item])>1:
                    itemNet[item] = self.rg.trainSet_i[item]

            filteredRatings = defaultdict(list)

            for item in itemNet:
                for user in itemNet[item]:
                    if itemNet[item][user] > 0:
                        filteredRatings[user].append(item)

            self.CUNet = defaultdict(list)

            for user1 in tqdm(filteredRatings):
                s1 = set(filteredRatings[user1])
                for user2 in filteredRatings:
                    if user1 != user2:
                        s2 = set(filteredRatings[user2])
                        weight = len(s1.intersection(s2))
                        if weight > 0:
                            self.CUNet[user1]+=[user2]
            
            print('Generating random deep walks...')
            self.walks = []
            self.visited = defaultdict(dict)
            for user in tqdm(self.CUNet):
                for t in range(self.config.walkCount):
                    path = [str(user)]
                    lastNode = user
                    for i in range(1,self.config.walkLength):
                        nextNode = choice(self.CUNet[lastNode])
                        count=0
                        while(nextNode in self.visited[lastNode]):
                            nextNode = choice(self.CUNet[lastNode])
                            #break infinite loop
                            count+=1
                            if count==self.config.walkLength: # 10
                                break
                        path.append(str(nextNode))
                        self.visited[user][nextNode] = 1
                        lastNode = nextNode
                    self.walks.append(path)

            self.model = w2v.Word2Vec(self.walks, size=self.config.walkDim, window=5, min_count=0, iter=3)

            self.topKSim = defaultdict(dict)
            i = 0
            for u1 in tqdm(self.CUNet):
                sims = {}
                for u2 in self.CUNet:
                    if user1 != user2:
                        if self.user_sim.contains(u1,u2):
                            continue
                        wu1 = self.model[str(u1)]
                        wu2 = self.model[str(u2)]
                        sims[u2]=cosine(wu1,wu2) #若为空咋整
                        self.user_sim.set(u1,u2,sims[u2])
                i += 1
                if i % 200 == 0:
                    print('progress:', i, '/', len(self.CUNet))
            if not os.path.exists('../data/sim'):
                os.makedirs('../data/sim')
                print('../data/sim folder has been established.')
            util.save_data(self.user_sim, '../data/sim/%s_08_uu_gemf_cv0.pkl'  % self.config.dataset_name)

        # compute the k neighbors of user
        if load_save_sim:
            self.user_k_neibor = util.load_data(
                '../data/neibor/%s_08_uu_%s_neibor_gemf_cv0.pkl'  % (self.config.dataset_name, user_near_num))
        for user in self.rg.user:
            self.topKSim[u1] = sorted(sims.items(), key=lambda d: d[1], reverse=True)[:self.config.topK]
            self.topKSim[u1] = self.topKSim[u1][:user_near_num]
            self.user_k_neibor[user] = dict(self.topKSim[u1])

        if not os.path.exists('../data/neibor'):
            os.makedirs('../data/neibor')
            print('../data/neibor folder has been established.')
            
        util.save_data(self.user_k_neibor,
                    '../data/neibor/%s_08_uu_%s_neibor_gemf_cv0.pkl' % (self.config.dataset_name, user_near_num))


    
if __name__ == '__main__':
    sim_ge = SimGe()
    sim_ge.build_user_item_sim_CF(0)