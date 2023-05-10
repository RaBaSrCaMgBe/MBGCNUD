from train import TrainManager
from utils import ContentManager

class test_obj:
    def __init__(self):
        self.model = 'MBGCNUD'
        self.name = 'MBGCNUD-K2-solo'
        self.dataset_name = 'Tmall'
        self.lr = 3e-4
        self.L2_norm = 1e-4
        self.gpu = True
        self.gpu_id = 0
        self.num_workers = 0
        self.epoch = 400
        self.path = 'dataset'
        self.output = 'output'
        self.batch_size = 2048
        self.test_batch_size = 512
        self.embedding_size = 32
        self.es_patience = 40
        self.loss_mode = 'mean'
        self.relation = ['buy','cart','click','collect']
        self.pretrain_frozen = False
        self.create_embeddings = False
        self.pretrain_path = 'MBGCN-main/pretrain'
        self.node_dropout = 0.3
        self.message_dropout = 0.3
        self.lamb = 0.7
        self.mgnn_weight = [1,1,1,1]
        self.K = 2

if __name__=='__main__':
    a = test_obj()
    cm = ContentManager(a)
    Train = TrainManager(a, cm)
    #Train.train()
    Train.test()
