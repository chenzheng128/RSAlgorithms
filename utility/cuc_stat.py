# encoding:utf-8
import sys

sys.path.append("..")

import pandas as pd
from configx.configx import ConfigX
from configx.configc import ConfigCUC
import logging

# create logger
log = logging.getLogger('utility.cuc_stat.py')
# logging.s


def print_data_file_stats(dpath):
    """
    打印数据文件统计
    """
    logging.basicConfig(level=logging.DEBUG,  format='%(name)s - %(levelname)s - %(message)s')
    
    log.info("== 数据集评分统计信息 %s ==" % (dpath))
    log.info(" m    *     n            数据量   稀疏度 (%) ")
    data = pd.read_csv(dpath, sep=' ', header=None)
    m = len(data[0].unique())
    n = len(data[1].unique())
    count = len(data)
    log.info("%s * %8s  %10s  %6f  " % (m , n, count, float(count)/(m*n)*100 ))
    
if __name__ == '__main__':
    config = ConfigX()
    config = ConfigCUC()
    print_data_file_stats(config.rating_path)
    print_data_file_stats(config.trust_path)


