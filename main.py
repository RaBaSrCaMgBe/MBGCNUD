from absl import app
from absl import flags

from train import TrainManager
from utils import VisManager, ContentManager

#构建absl命令行系统
FLAGS = flags.FLAGS

#absl里的模型参数
flags.DEFINE_enum('model','MF',['MF','MBGCN'],'Model name') #模型名称(矩阵分解,MBGCN)
flags.DEFINE_string('name','MF-experiment','Experiment name') #实验名称
flags.DEFINE_string('dataset_name','Tmall_release',"Dataset's name") #数据集名称
flags.DEFINE_float('lr',0.0001,'Learning rate') #学习率
flags.DEFINE_float('L2_norm',0.001,'L2 norm') #L2系数
flags.DEFINE_bool('gpu','True','Use GPU or not') #是否使用GPU
flags.DEFINE_integer('gpu_id',6,'GPU ID') #GPUid
flags.DEFINE_integer('num_workers', 8, 'Number of processes for training and testing') #用于训练和测试的进程数
flags.DEFINE_integer('epoch', 400, 'Max epochs for training') #训练时迭代次数
flags.DEFINE_string('path','/MBGCN-main','The path where the data is') #数据集目录
flags.DEFINE_string('output','/data3/jinbowen/multi_behavior/output','The path to store output message') #输出目录
flags.DEFINE_integer('port',33337,'Port to show visualization results for visdom') #可视化结果端口号
flags.DEFINE_integer('batch_size',2048,'Batch size') #一次训练样本量
flags.DEFINE_integer('test_batch_size',512,'Test batch size') #一次测试样本量
flags.DEFINE_integer('embedding_size',32,'Embedding Size') #嵌入大小
flags.DEFINE_integer('es_patience',10,'Early Stop Patience') #EarlyStop参数
flags.DEFINE_enum('loss_mode','mean',['mean','sum'],'Loss Mode') #损失类型(平均损失,累积损失)
flags.DEFINE_multi_string('relation', ['buy','wanted','clicked','detail_view'], 'Relations') #交互类型
#预训练模型参数
flags.DEFINE_bool('pretrain_frozen','False','Froze the pretrain parameter or not') #是否冻结预训练参数
flags.DEFINE_string('create_embeddings','False','Pretrain or not? If not create embedding here!') #是否使用预训练嵌入
flags.DEFINE_string('pretrain_path','/data3/jinbowen/multi_behavior/output/Steam/MF-Steam-lr3-L4@jinbowen',"Path where the pretrain model is.") #预训练模型目录
#dropout机制
flags.DEFINE_float('node_dropout',0.2,'Node dropout ratio') #节点dropout系数
flags.DEFINE_float('message_dropout',0.2,'Message dropout ratio') #消息dropout系数
#MBGCN和MGNN的参数
flags.DEFINE_float('lamb',0.1,'Lambda for the loss for MultiGNN with item space calculation') #λ系数 衡量基于用户兴趣得分和基于物品兴趣得分权重
flags.DEFINE_multi_float('mgnn_weight',[1,1,1,1],'Weight for MGNN') #各行为固有影响力

class test_obj:
    def __init__(self):
        self.model = 'MBGCN'
        self.name = 'MBGCNUD-test'
        self.dataset_name = 'Tmall'
        self.lr = 3e-4
        self.L2_norm = 1e-4
        self.gpu = True
        self.gpu_id = 0
        self.num_workers = 0
        self.epoch = 400
        self.path = 'dataset'
        self.output = 'output'
        self.port = 8097
        self.batch_size = 2048
        self.test_batch_size = 512
        self.embedding_size = 32
        self.es_patience = 40
        self.loss_mode = 'mean'
        self.relation = ['buy','cart','click','collect']
        self.pretrain_frozen = False
        self.create_embeddings = True
        self.pretrain_path = 'MBGCN-main/pretrain'
        self.node_dropout = 0.2
        self.message_dropout = 0.2
        self.lamb = 0.7
        self.mgnn_weight = [1,1,1,1]
        self.K = 8



def main(argv):
    #实例化FLAG
    flags_obj = FLAGS
    #根据输入参数实例化可视化对象
    vm = VisManager(flags_obj)
    #根据输入参数实例化内容管理对象
    cm = ContentManager(flags_obj)
    #根据输入参数实例化训练模块对象
    Train = TrainManager(flags_obj, vm, cm)
    #训练
    Train.train()

if __name__=='__main__':
    a = test_obj()
    vm = VisManager(a)
    cm = ContentManager(a)
    Train = TrainManager(a, vm, cm)
    #Train.train()
    Train.test()
