import os
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sparse
import torch.nn.functional as F
import time
from torch.utils.data._utils.pin_memory import pin_memory
class model_base(nn.Module):
    #基本模型 其他具体模型的父类
    def __init__(self, flags_obj, trainset, device):
        super().__init__()
        self.embed_size = flags_obj.embedding_size #嵌入尺寸
        self.L2_norm = flags_obj.L2_norm #L2系数
        self.device = device #设备
        self.user_num = trainset.user_num #训练集用户数量
        self.item_num = trainset.item_num #训练集物品数量
        if flags_obj.create_embeddings==True:
            #如果创建新的嵌入 则用Xavier初始化
            self.item_embedding = nn.Parameter(torch.FloatTensor(self.item_num,self.embed_size)) #物品嵌入矩阵 一行对应一个物品的嵌入
            nn.init.xavier_normal_(self.item_embedding) #Xavier初始化
            self.user_embedding = nn.Parameter(torch.FloatTensor(self.user_num,self.embed_size)) #用户嵌入矩阵 一行对应一个用户的嵌入
            nn.init.xavier_normal_(self.user_embedding) #Xavier初始化
        else:
            #如果使用预训练嵌入 则加载预训练模型
            if flags_obj.pretrain_frozen==False:
                self.item_embedding = nn.Parameter(torch.load("dataset/Tmall/item_embedding").to(self.device))
                self.user_embedding = nn.Parameter(torch.load("dataset/Tmall/user_embedding").to(self.device))

    def propagate(self,*args,**kwargs):
        #传播 应该是个虚拟方法 给别人继承的
        '''
        raw embeddings -> embeddings for predicting
        return (user's,POI's)
        '''
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        #预测 应该是个虚拟方法 给别人继承的
        '''
        embeddings of targets for predicting -> scores
        return scores
        '''
        raise NotImplementedError

    def regularize(self, user_embeddings, item_embeddings):
        #正则化项
        '''
        embeddings of targets for predicting -> extra loss(default: L2 loss...)
        '''
        return self.L2_norm*((user_embeddings**2).sum()+(item_embeddings**2).sum())

    def forward(self, users, items):
        #前向传播
        #读取用户和物品嵌入
        #把用户嵌入和给定的所有物品嵌入一一求内积
        #然后再把L2给算出来
        users_feature, item_feature = self.propagate()
        item_embeddings = item_feature[items]
        user_embeddings = users_feature[users].expand(-1,items.shape[1],-1)
        pred = self.predict(user_embeddings, item_embeddings)
        L2_loss = self.regularize(user_embeddings, item_embeddings)
        return pred, L2_loss

    def evaluate(self, users):
        #评估 应该是个虚拟方法 给别人继承的
        #测试的时候才有用
        '''
        just for testing, compute scores of all POIs for `users` by `propagate_result`
        '''
        raise NotImplementedError

class MBGCN(model_base):
    #MBGCN模型
    def __init__(self, flags_obj, trainset, device):
        super().__init__(flags_obj, trainset, device)
        self.relation_dict = trainset.relation_dict #各交互类型下的交互矩阵
        self.mgnn_weight = flags_obj.mgnn_weight #各行为固有影响力
        self.item_graph = trainset.item_graph #物品之间的关系图
        self.train_matrix = trainset.train_matrix.to(self.device) #训练集的交互矩阵
        self.relation = trainset.relation #交互类型
        self.lamb = flags_obj.lamb #λ系数
        self.K = flags_obj.K #潜在偏好因子个数
        self.item_graph_degree = trainset.item_graph_degree #物品之间关系图各节点的度
        self.user_behaviour_degree = trainset.user_behaviour_degree.to(self.device) #不同交互类型下用户节点度数
        self.message_drop = nn.Dropout(p=flags_obj.message_dropout) #消息dropout
        self.train_node_drop = nn.Dropout(p=flags_obj.node_dropout) #训练集节点dropout
        self.node_drop = nn.ModuleList([nn.Dropout(p=flags_obj.node_dropout) for _ in self.relation_dict]) #所有交互类型的节点dropout
        self.__to_gpu() #送到gpu
        self.__param_init() #参数初始化


    def __to_gpu(self):
        #送到gpu
        for key in self.relation_dict:
            self.relation_dict[key] = self.relation_dict[key].to(self.device)
        for key in self.item_graph:
            self.item_graph[key] = self.item_graph[key].to(self.device)
        for key in self.item_graph_degree:
            self.item_graph_degree[key] = self.item_graph_degree[key].to(self.device)

    def __decode_weight(self):
        #计算基于用户的各行为权重 这个函数似乎没被调用
        weight = nn.functional.softmax(self.mgnn_weight).unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        #按列除
        self.user_behaviour_weight = self.user_behaviour_degree.float() / (total_weight + 1e-8)

    def __param_init(self):
        #参数初始化
        #各行为权重初始化(初始化之后就可以优化了!)
        self.mgnn_weight = nn.Parameter(torch.FloatTensor(self.mgnn_weight))
        #物品预测得分时的参数矩阵
        self.item_behaviour_W = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.embed_size * 2, self.embed_size * 2)) for _ in self.mgnn_weight])
        for param in self.item_behaviour_W:
            nn.init.xavier_normal_(param)
        #物品间传播的参数矩阵
        self.item_propagate_W = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size)) for _ in self.mgnn_weight])
        for param in self.item_propagate_W:
            nn.init.xavier_normal_(param)
        #用户物品间传播的参数矩阵
        self.W = nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size))
        nn.init.xavier_normal_(self.W)

    def forward(self, user, item):
        #前向传播
        #进行了一遍节点dropout
        indices = self.train_matrix._indices() #索引 即[用户id 物品id]
        values = self.train_matrix._values() #值 是否交互
        values = self.train_node_drop(values) #节点dropout 有的交互记录删掉了
        train_matrix = torch.sparse.FloatTensor(indices, values, size=self.train_matrix.shape)
        #计算基于用户的各行为权重
        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)
        for i, key in enumerate(self.relation_dict):
            #又跑了一遍节点dropout 不过是针对不通交互类型的
            indices = self.relation_dict[key]._indices()
            values = self.relation_dict[key]._values()
            values = self.node_drop[i](values)
            tmp_relation_matrix = torch.sparse.FloatTensor(indices, values, size=self.relation_dict[key].shape)
            #tmp_relation_matrix = torch.FloatTensor([[1,1,0],[0,1,1],[1,0,1]]).to('cuda')
            #物品间信息传播
            tmp_item_propagation = torch.mm(torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8), self.item_propagate_W[i])
            tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)
            tmp_item_embedding = tmp_item_propagation[item]

            #item向user传播嵌入
            tmp_user_neighbour = torch.mm(tmp_relation_matrix, self.item_embedding) / (self.user_behaviour_degree[:,i].unsqueeze(-1) + 1e-8)
            #print(tmp_user_neighbour)
            #用来算得分的
            tmp_user_item_neighbour_p = torch.mm(tmp_relation_matrix, tmp_item_propagation) / (self.user_behaviour_degree[:,i].unsqueeze(-1) + 1e-8)
            if i == 0:
                #目标行为
                #行为影响力加权
                user_feature = user_behaviour_weight[:,i].unsqueeze(-1) * tmp_user_neighbour
                #算得分
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                #选出训练集里有的用户 这里expand是因为 BPR样本一个用户对应两个物品 所以要把用户交互过物品的整体信息复制一份
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user].expand(-1, item.shape[1], -1)
                score2 = torch.sum(tuser_tbehaviour_item_projection * tmp_item_embedding, dim=2)
            else:
                user_feature += user_behaviour_weight[:,i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user].expand(-1, item.shape[1], -1)
                score2 += torch.sum(tuser_tbehaviour_item_projection * tmp_item_embedding, dim=2)

        score2 = score2 / len(self.mgnn_weight)

        #user向item传播嵌入
        item_feature = torch.mm(train_matrix.t(), self.user_embedding)

        #user和item传播结果聚合
        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)

        #更新嵌入 和之前的拼到一起
        user_feature = torch.cat((self.user_embedding, user_feature), dim=1)
        item_feature = torch.cat((self.item_embedding, item_feature), dim=1)

        #消息dropout
        user_feature = self.message_drop(user_feature)
        item_feature = self.message_drop(item_feature)

        #算基于用户分数
        tmp_user_feature = user_feature[user].expand(-1,item.shape[1],-1)
        tmp_item_feature = item_feature[item]
        score1 = torch.sum(tmp_user_feature * tmp_item_feature,dim=2)

        scores = self.lamb * score1 + (1 - self.lamb) * score2

        L2_loss = self.regularize(tmp_user_feature, tmp_item_feature)

        #偏好正则化项
        #print(pre_reg)
        return scores, L2_loss
        #return scores, L2_loss, pre_reg[0]

    def evaluate(self, user):
        #给用户推荐东西
        #算行为影响力
        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)
        for i, key in enumerate(self.relation_dict):
            #老规矩 先算物品传播结果
            tmp_item_propagation = torch.mm(torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8), self.item_propagate_W[i])
            tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)

            #第一个是item向user传播嵌入 第二个是聚合用户相邻物品算分的
            tmp_user_neighbour = torch.mm(self.relation_dict[key], self.item_embedding) / (self.user_behaviour_degree[:,i].unsqueeze(-1) + 1e-8)
            tmp_user_item_neighbour_p = torch.mm(self.relation_dict[key], tmp_item_propagation) / (self.user_behaviour_degree[:,i].unsqueeze(-1) + 1e-8)
            if i == 0:
                user_feature = user_behaviour_weight[:,i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user]
                score2 = torch.mm(tuser_tbehaviour_item_projection, tmp_item_propagation.t())
            else:
                user_feature += user_behaviour_weight[:,i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user]
                score2 += torch.mm(tuser_tbehaviour_item_projection, tmp_item_propagation.t())

        score2 = score2 / len(self.mgnn_weight)

        item_feature = torch.mm(self.train_matrix.t(), self.user_embedding)

        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)
        
        user_feature = torch.cat((self.user_embedding,user_feature),dim=1)
        item_feature = torch.cat((self.item_embedding,item_feature),dim=1)

        tmp_user_feature = user_feature[user]
        score1 = torch.mm(tmp_user_feature, item_feature.t())

        scores = self.lamb * score1 + (1 - self.lamb) * score2

        return scores


class MBGCNUD(model_base):
    # MBGCNUD模型
    def __init__(self, flags_obj, trainset, device):
        super().__init__(flags_obj, trainset, device)
        self.relation_dict = trainset.relation_dict  # 各交互类型下的交互矩阵
        self.mgnn_weight = flags_obj.mgnn_weight  # 各行为固有影响力
        self.item_graph = trainset.item_graph  # 物品之间的关系图
        self.train_matrix = trainset.train_matrix.to(self.device)  # 训练集的交互矩阵
        self.relation = trainset.relation  # 交互类型
        self.lamb = flags_obj.lamb  # λ系数
        self.K = flags_obj.K  # 潜在偏好因子个数
        self.item_graph_degree = trainset.item_graph_degree  # 物品之间关系图各节点的度
        self.user_behaviour_degree = trainset.user_behaviour_degree.to(self.device)  # 不同交互类型下用户节点度数
        self.message_drop = nn.Dropout(p=flags_obj.message_dropout)  # 消息dropout
        self.train_node_drop = nn.Dropout(p=flags_obj.node_dropout)  # 训练集节点dropout
        self.node_drop = nn.ModuleList(
            [nn.Dropout(p=flags_obj.node_dropout) for _ in self.relation_dict])  # 所有交互类型的节点dropout
        self.__to_gpu()  # 送到gpu
        self.__param_init()  # 参数初始化

    def __to_gpu(self):
        # 送到gpu
        for key in self.relation_dict:
            self.relation_dict[key] = self.relation_dict[key].to(self.device)
        for key in self.item_graph:
            self.item_graph[key] = self.item_graph[key].to(self.device)
        for key in self.item_graph_degree:
            self.item_graph_degree[key] = self.item_graph_degree[key].to(self.device)

    def __decode_weight(self):
        # 计算基于用户的各行为权重 这个函数似乎没被调用
        weight = nn.functional.softmax(self.mgnn_weight).unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        # 按列除
        self.user_behaviour_weight = self.user_behaviour_degree.float() / (total_weight + 1e-8)

    def __param_init(self):
        # 参数初始化
        # 各行为权重初始化(初始化之后就可以优化了!)
        self.mgnn_weight = nn.Parameter(torch.FloatTensor(self.mgnn_weight))
        # 物品预测得分时的参数矩阵
        '''self.item_behaviour_W = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.embed_size * 2, self.embed_size * 2)) for _ in self.mgnn_weight])'''
        self.item_behaviour_W = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size)) for _ in self.mgnn_weight])
        for param in self.item_behaviour_W:
            nn.init.xavier_normal_(param)
        # 物品间传播的参数矩阵
        self.item_propagate_W = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size)) for _ in self.mgnn_weight])
        for param in self.item_propagate_W:
            nn.init.xavier_normal_(param)
        # 用户物品间传播的参数矩阵
        self.W = nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size))
        nn.init.xavier_normal_(self.W)
        # 偏好正则化项投影矩阵
        self.W_P = nn.Parameter(torch.FloatTensor(self.K, int(self.embed_size / self.K)))
        nn.init.xavier_normal_(self.W_P)
        # 偏好正则化项投影矩阵残差
        self.B_P = nn.Parameter(torch.FloatTensor(self.K, 1))
        nn.init.xavier_normal_(self.B_P)
        #投影矩阵
        self.W_K = nn.ParameterList([nn.Parameter(torch.FloatTensor(int(self.embed_size/self.K), self.embed_size)) for _ in range(self.K)])
        for param in self.W_K:
            nn.init.xavier_normal_(param)
        #投影矩阵残差
        self.B_K = nn.ParameterList([nn.Parameter(torch.FloatTensor(int(self.embed_size / self.K), 1)) for _ in range(self.K)])
        for param in self.B_K:
            nn.init.xavier_normal_(param)

    def forward(self, user, item):
        # 前向传播
        # 进行了一遍节点dropout
        indices = self.train_matrix._indices()  # 索引 即[用户id 物品id]
        values = self.train_matrix._values()  # 值 是否交互
        values = self.train_node_drop(values)  # 节点dropout 有的交互记录删掉了
        train_matrix = torch.sparse.FloatTensor(indices, values, size=self.train_matrix.shape)
        # 计算基于用户的各行为权重
        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)
        tmp_ReLU = nn.ReLU()
        for i, key in enumerate(self.relation_dict):
            # 又跑了一遍节点dropout 不过是针对不通交互类型的
            indices = self.relation_dict[key]._indices()
            values = self.relation_dict[key]._values()
            values = self.node_drop[i](values)
            tmp_relation_matrix = torch.sparse.FloatTensor(indices, values, size=self.relation_dict[key].shape)
            # tmp_relation_matrix = torch.FloatTensor([[1,1,0],[0,1,1],[1,0,1]]).to('cuda')
            # 物品间信息传播
            tmp_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8),
                self.item_propagate_W[i])
            #tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)
            tmp_item_embedding = tmp_item_propagation[item]

            # item向user传播嵌入
            for j in range(self.K):
                if j == 0:
                    s_i = nn.functional.normalize(
                        tmp_ReLU(torch.mm(self.W_K[j], self.item_embedding.t()) + self.B_K[j]), dim=0)
                    s_u = nn.functional.normalize(
                        tmp_ReLU(torch.mm(self.W_K[j], self.user_embedding.t()) + self.B_K[j]), dim=0)
                else:
                    s_i = torch.cat((s_i, nn.functional.normalize(
                        tmp_ReLU(torch.mm(self.W_K[j], self.item_embedding.t()) + self.B_K[j]), dim=0)), dim=0)
                    s_u = torch.cat((s_u, nn.functional.normalize(
                        tmp_ReLU(torch.mm(self.W_K[j], self.user_embedding.t()) + self.B_K[j]), dim=0)), dim=0)
            s_u = s_u.t_()

            for j in range(self.K):
                if j == 0:
                    tmp_r_sum = torch.exp(
                        self.user_embedding[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)] @ s_i[j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K), ...])
                else:
                    tmp_r_sum = tmp_r_sum + torch.exp(
                        self.user_embedding[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)] @ s_i[j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K), ...])
            for j in range(self.K):
                tmp_r = torch.exp(
                    self.user_embedding[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)] @ s_i[j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K), ...]) / tmp_r_sum
                s_u[..., j*int(self.embed_size/self.K):(j+1)*int(self.embed_size/self.K)] = s_u[..., j*int(self.embed_size/self.K):(j+1)*int(self.embed_size/self.K)]+tmp_r*tmp_relation_matrix.to_dense()@s_i[j*int(self.embed_size/self.K):(j+1)*int(self.embed_size/self.K), ...].t()
            for j in range(self.K):
                if j == 0:
                    tmp_user_neighbour = nn.functional.normalize(s_u[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)], dim=1)
                else:
                    tmp_user_neighbour = torch.cat(
                        (tmp_user_neighbour, nn.functional.normalize(s_u[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)], dim=1)),
                        dim=1)
            # print(tmp_user_neighbour)
            # 用来算得分的
            tmp_user_item_neighbour_p = torch.mm(tmp_relation_matrix, tmp_item_propagation) / (
                        self.user_behaviour_degree[:, i].unsqueeze(-1) + 1e-8)
            if i == 0:
                # 目标行为
                # 行为影响力加权
                user_feature = user_behaviour_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                # 算得分
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                # 选出训练集里有的用户 这里expand是因为 BPR样本一个用户对应两个物品 所以要把用户交互过物品的整体信息复制一份
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user].expand(-1, item.shape[1], -1)
                score2 = torch.sum(tuser_tbehaviour_item_projection * tmp_item_embedding, dim=2)
            else:
                user_feature += user_behaviour_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user].expand(-1, item.shape[1], -1)
                score2 += torch.sum(tuser_tbehaviour_item_projection * tmp_item_embedding, dim=2)

        score2 = score2 / len(self.mgnn_weight)

        # user向item传播嵌入
        item_feature = torch.mm(train_matrix.t(), self.user_embedding)

        # user和item传播结果聚合
        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)

        # 更新嵌入 和之前的拼到一起
        '''user_feature = torch.cat((self.user_embedding, user_feature), dim=1)
        item_feature = torch.cat((self.item_embedding, item_feature), dim=1)'''

        # 消息dropout
        user_feature = self.message_drop(user_feature)
        item_feature = self.message_drop(item_feature)

        # 算基于用户分数
        tmp_user_feature = user_feature[user].expand(-1, item.shape[1], -1)
        tmp_item_feature = item_feature[item]
        score1 = torch.sum(tmp_user_feature * tmp_item_feature, dim=2)

        scores = self.lamb * score1 + (1 - self.lamb) * score2

        L2_loss = self.regularize(tmp_user_feature, tmp_item_feature)

        # 偏好正则化项
        pre_reg = torch.FloatTensor([0]).to('cuda')
        for j in range(self.K):
            P = self.W_P @ user_feature[..., j * int(self.embed_size / self.K):(j + 1) * int(self.embed_size / self.K)].t() + self.B_P
            #P = self.W_P @ user_feature[..., self.embed_size + j * int(self.embed_size / self.K):self.embed_size + (j + 1) * int(self.embed_size / self.K)].t() + self.B_P
            P = nn.functional.softmax(P, dim=0)
            P = torch.log(P+1e-8)
            pre_reg = pre_reg + torch.sum(P, dim=1)[j]
        pre_reg = -(1 / self.K) * pre_reg
        # print(pre_reg)
        return scores, L2_loss, pre_reg[0]

    def evaluate(self, user):
        # 给用户推荐东西
        # 算行为影响力
        weight = self.mgnn_weight.unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        user_behaviour_weight = self.user_behaviour_degree * self.mgnn_weight.unsqueeze(0) / (total_weight + 1e-8)
        tmp_ReLU = nn.ReLU()
        for i, key in enumerate(self.relation_dict):
            # 老规矩 先算物品传播结果
            tmp_item_propagation = torch.mm(
                torch.mm(self.item_graph[key].float(), self.item_embedding) / (self.item_graph_degree[key] + 1e-8),
                self.item_propagate_W[i])
            #tmp_item_propagation = torch.cat((self.item_embedding, tmp_item_propagation), dim=1)

            # 第一个是item向user传播嵌入 第二个是聚合用户相邻物品算分的
            for j in range(self.K):
                if j == 0:
                    s_i = nn.functional.normalize(
                        tmp_ReLU(torch.mm(self.W_K[j], self.item_embedding.t()) + self.B_K[j]), dim=0)
                    s_u = nn.functional.normalize(
                        tmp_ReLU(torch.mm(self.W_K[j], self.user_embedding.t()) + self.B_K[j]), dim=0)
                else:
                    s_i = torch.cat((s_i, nn.functional.normalize(
                        tmp_ReLU(torch.mm(self.W_K[j], self.item_embedding.t()) + self.B_K[j]), dim=0)), dim=0)
                    s_u = torch.cat((s_u, nn.functional.normalize(
                        tmp_ReLU(torch.mm(self.W_K[j], self.user_embedding.t()) + self.B_K[j]), dim=0)), dim=0)
            s_u = s_u.t_()

            for j in range(self.K):
                if j == 0:
                    tmp_r_sum = torch.exp(
                        self.user_embedding[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)] @ s_i[j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K), ...])
                else:
                    tmp_r_sum = tmp_r_sum + torch.exp(
                        self.user_embedding[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)] @ s_i[j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K), ...])
            for j in range(self.K):
                tmp_r = torch.exp(
                    self.user_embedding[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)] @ s_i[j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K), ...]) / tmp_r_sum
                s_u[..., j*int(self.embed_size/self.K):(j+1)*int(self.embed_size/self.K)] = s_u[..., j*int(self.embed_size/self.K):(j+1)*int(self.embed_size/self.K)]+tmp_r*self.relation_dict[key].to_dense()@s_i[j*int(self.embed_size/self.K):(j+1)*int(self.embed_size/self.K), ...].t()
            for j in range(self.K):
                if j == 0:
                    tmp_user_neighbour = nn.functional.normalize(s_u[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)], dim=1)
                else:
                    tmp_user_neighbour = torch.cat(
                        (tmp_user_neighbour, nn.functional.normalize(s_u[..., j * int(self.embed_size/self.K):(j + 1) * int(self.embed_size/self.K)], dim=1)),
                        dim=1)
            tmp_user_item_neighbour_p = torch.mm(self.relation_dict[key], tmp_item_propagation) / (
                        self.user_behaviour_degree[:, i].unsqueeze(-1) + 1e-8)
            if i == 0:
                user_feature = user_behaviour_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user]
                score2 = torch.mm(tuser_tbehaviour_item_projection, tmp_item_propagation.t())
            else:
                user_feature += user_behaviour_weight[:, i].unsqueeze(-1) * tmp_user_neighbour
                tbehaviour_item_projection = torch.mm(tmp_user_item_neighbour_p, self.item_behaviour_W[i])
                tuser_tbehaviour_item_projection = tbehaviour_item_projection[user]
                score2 += torch.mm(tuser_tbehaviour_item_projection, tmp_item_propagation.t())

        score2 = score2 / len(self.mgnn_weight)

        item_feature = torch.mm(self.train_matrix.t(), self.user_embedding)

        user_feature = torch.mm(user_feature, self.W)
        item_feature = torch.mm(item_feature, self.W)

        '''user_feature = torch.cat((self.user_embedding, user_feature), dim=1)
        item_feature = torch.cat((self.item_embedding, item_feature), dim=1)'''

        tmp_user_feature = user_feature[user]
        score1 = torch.mm(tmp_user_feature, item_feature.t())

        scores = self.lamb * score1 + (1 - self.lamb) * score2

        return scores