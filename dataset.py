import torch
import numpy as np
import os
import scipy.sparse as sp
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    #用来初始化训练数据集的模块
    def __init__(self,flags_obj):
        self.path = flags_obj.path #数据集目录
        self.name = flags_obj.dataset_name #数据集名称
        self.__decode_relation(flags_obj) #解码出交互类型
        self.__load_size() #加载数据集大小
        self.__create_relation_matrix() #创建交互矩阵
        self.__calculate_user_behaviour() #计算不同交互类型下用户和物品度数
        self.__generate_ground_truth() #生成真实值
        self.__generate_train_matrix() #生成训练集的交互矩阵
        self.__load_item_graph() #加载物品之间的关系图
        self.cnt = 0 #待读取的样本集编号
        self.__read_train_data(self.cnt) #读取训练数据

    def __decode_relation(self, flags_obj):
        #解码交互类型
        self.relation = flags_obj.relation #交互类型列表

    def __load_size(self):
        #加载用户和物品数量
        with open(os.path.join(self.path, self.name, 'data_size.txt')) as f:
            data = f.readline()
            user_num, item_num = data.strip().split()
            self.user_num = int(user_num) #用户数量
            self.item_num = int(item_num) #物品数量

    def __load_item_graph(self):
        #加载物品之间的关系图
        self.item_graph = {} #各交互类型下的物品关系图
        self.item_graph_degree = {} #物品关系图中各节点的度
        for tmp_relation in self.relation:
            self.item_graph[tmp_relation] = torch.load(os.path.join(self.path, self.name, 'item_'+tmp_relation+'.pth'))
            #torch中的一维张量(也就是行向量)直接转置是不能生成列向量的 必须通过unsqueeze()方法
            self.item_graph_degree[tmp_relation] = self.item_graph[tmp_relation].sum(dim=1).float().unsqueeze(-1)

    def __create_relation_matrix(self):
        #为每种交互创建交互矩阵
        self.relation_dict = {} #各交互类型下的交互矩阵
        for i in range(len(self.relation)):
            index = []
            with open(os.path.join(self.path, self.name, self.relation[i]+'.txt')) as f:
                data = f.readlines()
                for row in data:
                    user, item = row.strip().split()
                    user, item = int(user), int(item)
                    index.append([user,item])
            index_tensor = torch.LongTensor(index)
            lens, _ = index_tensor.shape
            #创建交互矩阵
            #sparse本来是用来压缩稀疏矩阵的 第一个参数是索引 标注元素位置 第二个参数是值 表示索引对应的值 第三个参数是尺寸 也就是压缩前的矩阵尺寸
            #在这里 把交互信息作为索引 把1作为值 把用户和物品数量作为矩阵的行数和列数 就可以利用交互信息来生成交互矩阵了
            self.relation_dict[self.relation[i]] = torch.sparse.FloatTensor(index_tensor.t(), torch.ones(lens, dtype=torch.float), torch.Size([self.user_num,self.item_num]))

    def __calculate_user_behaviour(self):
        #计算不同交互类型下用户和物品度数
        for i in range(len(self.relation)):
            if i==0:
                #计算目标行为下用户和物品的度数
                #to_dense()是将压缩后的矩阵进行还原的方法
                user_behaviour = self.relation_dict[self.relation[i]].to_dense().sum(dim=1).unsqueeze(-1)
                item_behaviour = self.relation_dict[self.relation[i]].to_dense().t().sum(dim=1).unsqueeze(-1)
            else:
                #计算辅助行为下用户/物品的度数 然后拼接在user_behaviour/item_behaviour后面
                user_behaviour = torch.cat((user_behaviour, self.relation_dict[self.relation[i]].to_dense().sum(dim=1).unsqueeze(-1)), dim=1)
                item_behaviour = torch.cat((item_behaviour, self.relation_dict[self.relation[i]].to_dense().t().sum(dim=1).unsqueeze(-1)), dim=1)
        self.user_behaviour_degree = user_behaviour
        self.item_behaviour_degree = item_behaviour

    def __generate_ground_truth(self):
        #在计算损失函数时 要将模型预测值和真实值进行比对
        #这一步的目的就是生成真实值
        row_data = []
        col = []
        with open(os.path.join(self.path, self.name, 'train.txt')) as f:
            data = f.readlines()
            for row in data:
                user, item = row.strip().split()
                user, item = int(user), int(item)
                row_data.append(user)
                col.append(item)
        row_data = np.array(row_data)
        col = np.array(col)
        values = np.ones(len(row_data), dtype=float)
        self.ground_truth = sp.csr_matrix((values, (row_data, col)), shape=(self.user_num, self.item_num)) #标签(是否交互)
        self.checkins = np.concatenate((row_data[:, None], col[:, None]), axis=1) #样本(用户 物品)

    def __generate_train_matrix(self):
        #生成训练集的交互矩阵
        index = []
        with open(os.path.join(self.path, self.name, 'train.txt')) as f:
            data = f.readlines()
            for row in data:
                user, item = row.strip().split()
                user, item = int(user), int(item)
                index.append([user, item])
        index_tensor = torch.LongTensor(index)
        lens, _ = index_tensor.shape
        self.train_matrix = torch.sparse.FloatTensor(index_tensor.t(), torch.ones(lens, dtype=torch.float), torch.Size([self.user_num, self.item_num]))

    def __read_train_data(self, i):
        #读取样本集
        tmp_array = []
        with open(os.path.join(self.path, self.name, 'sample_file', 'sample_'+str(i)+'.txt')) as f:
            data = f.readlines()
            for row in data:
                user, pid, nid = row.strip().split()
                user, pid, nid = int(user), int(pid), int(nid)
                tmp_array.append([user, pid, nid])
        self.train_tmp = torch.LongTensor(tmp_array) #当前读取的样本集

        print('Read Epoch{} Train Data over!'.format(i))

    def newit(self):
        #按照编号读取样本集并自增编号
        self.cnt += 1
        self.__read_train_data(self.cnt)

    def __getitem__(self, index):
        #返回一个样本的用户编号和正负样本编号
        return self.train_tmp[index, 0].unsqueeze(-1), self.train_tmp[index, 1:]

    def __len__(self):
        #返回训练集长度
        return len(self.checkins)

class TestDataset(Dataset):
    def __init__(self, flags_obj, trainset, task='test'):
        self.path = flags_obj.path #数据集目录
        self.name = flags_obj.dataset_name #数据集名称
        self.train_mask = trainset.ground_truth #训练集样本真实值
        self.user_num, self.item_num = trainset.user_num, trainset.item_num #用户数量 物品数量
        self.task = task #任务(训练/测试)
        self.__read_testset() #读取测试集

    def __read_testset(self):
        row = [] #用户
        col = [] #物品
        with open(os.path.join(self.path, self.name, self.task+'.txt')) as f:
            data = f.readlines()
            for line in data:
                user, item = line.strip().split()
                user, item = int(user), int(item)
                row.append(user)
                col.append(item)
        row = np.array(row)
        col = np.array(col)
        values = np.ones(len(row), dtype=float)
        self.checkins = np.concatenate((row[:, None], col[:, None]), axis=1) #样本(用户 物品)
        self.ground_truth = sp.csr_matrix((values, (row, col)), shape=(self.user_num, self.item_num)) #标签(是否交互)

    def __getitem__(self, index):
        #返回一个用户在测试集中交互过的物品与在训练集中交互过的物品
        return index, torch.from_numpy(self.ground_truth[index].toarray()).squeeze(), torch.from_numpy(self.train_mask[index].toarray()).float().squeeze()

    def __len__(self):
        #返回用户数量
        return self.user_num
