# encoding:utf-8
from configx.configx import ConfigX

class ConfigCUC(ConfigX):
    """
    docstring for ConfigCUC
    
    暂未启用 
    
    新增的全局变量类， 避免对原有的全局变量修改过多
    一般仅应用在新的 CUC model 模型中心

    configurate the global parameters and hyper parameters

    """

    def __init__(self):
        super(ConfigCUC, self).__init__()
     
        # # 新增 多元信息 path 
        # # self.meta_path = '../data/%s_meta.txt' % self.dataset_name  # the raw meta data file
        # # 计算出并使用的物品相似度 (KNN) 文件 ； 一般由 item_sim_path1( CF 相似度 )  + item_sim_path2 （属性 相似度) 计算
        # self.item_sim_path_final =  '../data/sim/%s_item_sim_final.txt'  % self.dataset_name  
        # self.item_sim_path_prefix = '../data/sim/%s_item_sim'            % self.dataset_name  # + 1.txt 2.txt ..."
        # # 计算出并使用的物品相似度 (KNN) 文件 ； 一般由 user_sim_path1( CF 相似度 )  + item_sim_path2 （社交/属性 相似度) 计算
        # self.user_sim_path_final =  '../data/sim/%s_user_sim_final.txt'  % self.dataset_name  
        # self.user_sim_path_prefix = '../data/sim/%s_user_sim'            % self.dataset_name  # + 1.txt 2.txt ..."

        # # movielens 数据集的 meta 信息
        # self.sim_ml_user_metas = ["Movie", "User", "Age", "Occupation"] # movielens user meta
        # self.sim_ml_item_metas = ["User", "Genre"] # movielens item meta
        
        # # douban movie 数据集的 meta 信息
        # self.sim_db_movie_user_metas = ["Movie", "Group"] # douban movie item meta
        # self.sim_db_movie_item_metas = ["Actor", "Director", "Type"] # douban movie item meta
        
        # # self.item_sim_ml_meta = "rating"; 

        # self.item_sim_mixs = ["cf", 'meta', 'cf-meta']
        # # self.item_sim_mix = ""
