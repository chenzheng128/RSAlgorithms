# encoding:utf-8

"""
Author: zhchen

!!! 注意 ！！！
  第一次执行前要备份好原始 dataset， 否则原始数据会被 sample 覆盖 
# 先备份好原有数据集（ 仅执行一次）
# cd data
# cp ft_ratings.txt ft_ratings.txt.orig
# cp ft_trust.txt ft_trust.txt.orig
!!! 数据必须处理为备份为.orig文件才可正常运行!!!

读取 config 中的数据集文件， 并 sample 成不同的大小，便于 DEBUG 测试程序

Usage:

 修改语句中的 1000 数值 即可: new_data = pd.read_csv(dpath + ".1000", sep=' ', header=None)
"""
import sys
from cross_validation import split_5_folds

sys.path.append("..")
import pandas as pd
from configx.configx import ConfigX

config = ConfigX()

print (config.trust_path)
print (config.rating_path)

data = pd.read_csv(config.rating_path, sep=' ', header=None)
# the number of links
print("rating: ", len(data))
print("user num: ", len(data[0].unique()))
print("item num: ", len(data[1].unique()))



def sample_data_file(dpath, config, pd, sample_num=-1, newfile=False):
    """
    dpath 要 sample 的数据文件路径
    sample_num 要 sample 的数据数量; 建议 1000 以上； -1 代表恢复原始数据
    """
    # 读取原始的备份数据集
    data = pd.read_csv(dpath + ".orig", sep=' ', header=None)

    # 生成不同的 sample 数据集
    print ("\n### 读入原始数据集： len(data)=", len(data) ); 
    #  if len(data) == 7090: data.to_csv("../data/ft-0.csv.bak", sep=' ', header=None, index=None)
    #  5kdata = data.sample(5000) # 采样语句 
    # for x in range(1, 5):
    #     new_num = x * 1000
    #     new_data = data.sample(new_num)
    #     if len(new_data) == new_num: 
    #         newpath = dpath + "." + str(new_num)
    #         new_data.to_csv(newpath, sep=' ', header=None , index=None)
    #         print ("  generate %d sample dataset %s ..." % (new_num, newpath ) )

    # 读入 sample 数据集， 需要哪个数据集, 就调入使用哪个。 )
    # 尝试 1000 左右； 太小的数据集 500, 100 , 可能因为随机量太小，导致没有可用 user/item rating。 因而 mae 为 0 
    if (sample_num == -1): # -1 时恢复原始数据
        sample_num = len(data)
    new_data = data.sample(sample_num)
    sample_data = new_data 

    print ("\n### 启用 samples 数据集： len(sample_data)=", len(sample_data) ); 
    sample_data.to_csv(dpath, sep=' ', header=None, index=None) # 启用 data3 500 条
    if newfile:
        sample_data.to_csv(dpath + "." + str(sample_num), sep=' ', header=None, index=None) # 启用 data3 500 条
    data = sample_data

    print("rating: ", len(data))
    print("user num: ", len(data[0].unique()))
    print("item num: ", len(data[1].unique()))


print ("\n### sample rating 文件 ...")
dpath = config.rating_path
print(dpath)
sample_data_file(dpath, config, pd, sample_num=1000, newfile=True)

print ("\n### sample trust 文件 ...")
dpath = config.trust_path
print(dpath)
sample_data_file(dpath, config, pd, sample_num=0, newfile=False)

print ("\n### 进行 5 folds ...")
split_5_folds(config)
