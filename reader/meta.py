# encoding:utf-8
import sys
sys.path.append("..")
import os, logging
from collections import defaultdict
import numpy as np
from configx.configx import ConfigX
from configx.configc import ConfigCUC
from utility.similarity import pearson_sp, cosine, jaccard_sim
from utility.matrix import SimMatrix
from utility.tools import normalize
from tqdm import tqdm

class MetaGetter(object):
    """
    docstring for MetaGetter
    read rating data and save the global parameters
    """

    def __init__(self, k):
        """
        k - fold number 运行 第几个 fold
        """
        super(MetaGetter, self).__init__()
        self.log = logging.getLogger('reader.MetaGetter')
        self.config = ConfigX()
        self.configc = ConfigCUC()
        self.k_current = k
        self.meta = {}     #key: id，value: 流水号
        self.item = {}     #key: id，value: 流水号
        self.all_Meta = {}  # 相当于 meta
        self.all_Item = {}  # 相当于 item
        self.id2meta = {}  #key: 流水号，value: id
        self.id2item = {}  #key: 流水号，value: id
        self.dataSet_m = defaultdict(dict)
        self.trainSet_m = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_m = defaultdict(dict)  # used to store the test set by hierarchy meta:[item,rating]
        self.testSet_i = defaultdict(dict)  # used to store the test set by hierarchy item:[meta,rating]
        self.testColdMetaSet_m = defaultdict(dict)  # cold start metas in test set
        self.trainHotMetaSet = []  # hot metas in train set
        self.trainSetLength = 0
        self.testSetLength = 0

        self.metaMeans = {}  # used to store the mean values of metas's meta
        self.itemMeans = {}  # used to store the mean values of items's meta
        self.globalMean = 0

        self.generate_data_set()  # generate train and test set
        self.getDataSet()
        self.get_data_statistics()
        self.get_cold_start_metas()

        self.log.debug(" 准备好各种结构，便于使用 ")
        self.log.debug("all_Meta len: %s " % len(self.all_Meta))
        self.log.debug("all_Item len: %s " % len(self.all_Item))
        self.log.debug("meta len: %s " % len(self.meta))
        self.log.debug("item len: %s " % len(self.item))
        self.log.debug("id2meta len: %s " % len(self.id2meta))
        self.log.debug("id2item len: %s " % len(self.id2item))
        self.log.debug("嵌套 dict trainSet_m len: %s " % len(self.trainSet_m))
        self.log.debug("嵌套 dict trainSet_i len: %s " % len(self.trainSet_i))


    def generate_data_set(self):
        for index, line in enumerate(self.trainSet()):
            # if i > testsize: break;
            i, m, r = line
            # print(u,i,r)
            if not m in self.meta:
                self.meta[m] = len(self.meta)
                self.id2meta[self.meta[m]] = m
            if not i in self.item:
                self.item[i] = len(self.item)
                self.id2item[self.item[i]] = i

            self.trainSet_m[m][i] = r
            self.trainSet_i[i][m] = r
            self.trainSetLength = index + 1
        self.all_Meta.update(self.meta)
        self.all_Item.update(self.item)

        # print(self.trainSetLength)
        # print(self.testSetLength)
        pass

    # for cross validation
    def trainSet(self):
        k = self.k_current
        for i in range(self.config.k_fold_num):
            if i != k:
                data_path = self.config.meta_path
                # if not os.path.exists
                if not os.path.isfile(data_path):
                    print("meta data %s is missing!" % self.config.meta_path)
                    sys.exit()
                with open(data_path, 'r') as f:
                    for index, line in enumerate(f):
                        i, m, r = line.strip('\r\n').split(self.config.sep)
                        # r = normalize(float(r))  # scale the rating score to [0-1]
                        yield (int(float(i)), int(float(m)), float(r))

    def getDataSet(self):
        with open(self.config.meta_path, 'r') as f:
            for index, line in enumerate(f):
                i, m, r = line.strip('\r\n').split(self.config.sep)
                self.dataSet_m[int(i)][int(m)] = float(r)

    def get_train_size(self):
        return (len(self.item), len(self.meta))

    # get cold start metas in test set
    def get_cold_start_metas(self):
        for meta in self.testSet_m.keys():
            rating_length = len(self.trainSet_m[meta])
            if rating_length <= self.config.coldMetaRating:
                self.testColdMetaSet_m[meta] = self.testSet_m[meta]
        # print('cold start metas count', len(self.testColdMetaSet_m))

    def get_data_statistics(self):

        total_rating = 0.0
        total_length = 0
        for u in self.meta:
            u_total = sum(self.trainSet_m[u].values())
            u_length = len(self.trainSet_m[u])
            total_rating += u_total
            total_length += u_length
            self.metaMeans[u] = u_total / float(u_length)

        for i in self.item:
            self.itemMeans[i] = sum(self.trainSet_i[i].values()) / float(len(self.trainSet_i[i]))

        if total_length == 0:
            self.globalMean = 0
        else:
            self.globalMean = total_rating / total_length

    def containsMeta(self, u):
        'whether meta is in training set'
        if u in self.meta:
            return True
        else:
            return False

    def containsItem(self, i):
        'whether item is in training set'
        if i in self.item:
            return True
        else:
            return False

    def containsMetaItem(self, meta, item):
        if meta in self.trainSet_m:
            if item in self.trainSet_m[meta]:
                # print(meta)
                # print(item)
                # print(self.trainSet_m[meta][item])
                return True
        return False

    def get_row(self, u):
        return self.trainSet_m[u]

    def get_col(self, c):
        return self.trainSet_i[c]

    def meta_rated_items(self, u):
        return self.trainSet_m[u].keys()

    def print_data_stats(self):
        print("%s.py meta = %s item = %s " % (model.__class__.__name__,  
            self.trainSet))


    def getSimMatrix(self, sim_func=pearson_sp):
        self.log.info("gettting sim matrix with '%s()' ... (will take some time) " % sim_func.__name__)
        sim_matrix = SimMatrix()
        count = 0
        for i1 in tqdm(self.item):
            for i2 in (self.item):
                if i1 != i2:
                    if sim_matrix.contains(i1, i2):
                        continue
                    a,b  = self.get_col(i1), self.get_col(i2)
                    # 皮尔逊相似度？ 修改为余弦相似度； 
                    # sim = pearson_sp(a, b)
                    # 计算 jaacard
                    sim = sim_func(a.keys(), b.keys())
                    # if sim1 != 0 or sim2 != 0 or sim3 != 0: 
                        # print (i1, a, i2, b, sim1, sim2, sim3)
                    # sim = sim1
                    sim = round(sim, 5)
                    if sim!=0:
                    #     self.log.debug("sim: %s -- item %s item %s " % (sim, i1, i2))
                        sim_matrix.set(i1, i2, sim)
                        count +=1
                    # if count > 10: 
                    #     break; # 测试早期停止数据
        self.log.info("'%s()' get %s sims " % (sim_func.__name__, sim_matrix.size()))
        return sim_matrix


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,  format='%(name)s - %(levelname)s - %(message)s')
    # logging.basicConfig(level=logging.INFO,  format='%(name)s - %(levelname)s - %(message)s')
    mg = MetaGetter(0)
    # for ind, entry in enumerate(mg.trainSet()):
    # 	if ind<5:
    # 		# print(entry)
    #         i, m, r = entry
    #         print(ind, m)
    mg.getSimMatrix(jaccard_sim)
    #         print(meta)
    # for i, m, r in enumerate(mg.trainSet()):
    #      print(m)

    # mg.generate_data_set()

    # print(mg.trainSet_m[52])
    # print(mg.trainSet_m[10])
