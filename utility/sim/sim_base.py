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
from prettyprinter import cpprint
from utility.matrix import SimMatrix
from utility.similarity import pearson_sp, cosine, jaccard_sim
from utility import util
from reader.rating import RatingGetter
from reader.meta import MetaGetter
from configx.configx import ConfigX
from tqdm import tqdm

class SimBase():

    def __init__(self):
        super(SimBase, self).__init__()

        self.config = ConfigX()
        self.user_sim = SimMatrix()
        self.item_sim = SimMatrix()
        self.user_k_neibor = defaultdict(dict)
        self.item_k_neibor = defaultdict(dict)
        
        
    def check_dataset(self):
        super(SimBase, self).check_dataset()
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
            self.item_sim = util.load_data('../data/sim/%s_08_ii_cucmemf_cv0.pkl' % self.config.dataset_name)
        else:
            # 封装 item 相似度计算
            self.item_sim = self.mg.getSimMatrix(jaccard_sim)
            util.save_data(self.item_sim, '../data/sim/%s_08_ii_cucmemf_cv0.pkl' % self.config.dataset_name)

        # compute the k neighbors of item
        if load_save_sim:
            self.item_k_neibor = util.load_data(
                '../data/neibor/%s_08_ii_%s_neibor_cucmemf_cv0.pkl'  % (self.config.dataset_name, item_near_num))
        for item in self.mg.item:
            matchItems = sorted(self.item_sim[item].items(), key=lambda x: x[1], reverse=True)[
                         :item_near_num]
            matchItems = matchItems[:item_near_num]
            self.item_k_neibor[item] = dict(matchItems)
        util.save_data(self.item_k_neibor,
                       '../data/neibor/%s_08_ii_%s_neibor_cucmemf_cv0.pkl' % (self.config.dataset_name, item_near_num))

        # compute user-user similarity matrix
        print('构建 user-user 相似度矩阵 ...')
        if load_save_sim:
        # if True:
            self.user_sim = util.load_data('../data/sim/%s_08_uu_cucmemf_cv0.pkl' % self.config.dataset_name)
        else:
            for u1 in tqdm(self.rg.user):
                for u2 in self.rg.user:
                    if u1 != u2:
                        if self.user_sim.contains(u1, u2):
                            continue
                        # 皮尔逊相似度？ 修改为余弦相似度?； 
                        sim = pearson_sp(self.rg.get_row(u1), self.rg.get_row(u2))
                        sim = round(sim, 5)
                        self.user_sim.set(u1, u2, sim)
            if not os.path.exists('../data/sim'):
                os.makedirs('../data/sim')
                print('../data/sim folder has been established.')
            util.save_data(self.user_sim, '../data/sim/%s_08_uu_cucmemf_cv0.pkl'  % self.config.dataset_name)

        # compute the k neighbors of user
        if load_save_sim:
            self.user_k_neibor = util.load_data(
                '../data/neibor/%s_08_uu_%s_neibor_cucmemf_cv0.pkl'  % (self.config.dataset_name, user_near_num))
        for user in self.rg.user:
            matchUsers = sorted(self.user_sim[user].items(), key=lambda x: x[1], reverse=True)[kfold:user_near_num]
            matchUsers = matchUsers[:user_near_num]
            self.user_k_neibor[user] = dict(matchUsers)

        if not os.path.exists('../data/neibor'):
            os.makedirs('../data/neibor')
            print('../data/neibor folder has been established.')
            
        util.save_data(self.user_k_neibor,
                    '../data/neibor/%s_08_uu_%s_neibor_cucmemf_cv0.pkl' % (self.config.dataset_name, user_near_num))


    
if __name__ == '__main__':
    sim_base = SimBase()
    sim_base.build_user_item_sim_CF(0)