import torch
import setproctitle
import os

from model import MF, MBGCN, MBGCNUD

class ContentManager(object):
    #用来实现进程命名 模型加载保存的模块
    def __init__(self, flag_obj):
        self.name = flag_obj.name
        self.dataset_name = flag_obj.dataset_name
        self.output_path = flag_obj.output
        self.path = os.path.join(self.output_path, self.dataset_name, self.name)
        self.set_proctitle()
        self.output_init()

    def set_proctitle(self):
        #设置进程名字
        setproctitle.setproctitle(self.name)

    def output_init(self):
        #创建输出目录
        if not os.path.exists(os.path.join(self.output_path, self.dataset_name)):
            os.mkdir(os.path.join(self.output_path, self.dataset_name))

        if not os.path.exists(os.path.join(self.output_path, self.dataset_name, self.name)):
            os.mkdir(os.path.join(self.output_path, self.dataset_name, self.name))

    def model_save(self, model):
        #保存模型
        torch.save(model.state_dict(), os.path.join(self.path, 'model.pkl'))

    def model_load(self, model):
        #加载模型
        model.load_state_dict(torch.load(os.path.join(self.path, 'model.pkl')))

class EarlyStopManager(object):
    #用来实现EarlyStop的模块
    def __init__(self, flags_obj):
        self.es_patience = flags_obj.es_patience
        self.count = 0
        self.max_metric = 0

    def step(self, metric, epoch):
        if epoch <=10:
            return False

        if metric > self.max_metric:
            self.max_metric = metric
            self.count = 0
            return False
        else:
            self.count += 1
            if self.count < self.es_patience:
                return False
            else:
                return True


class ModelSelector(object):
    def __init__(self):
        pass

    @staticmethod
    def getModel(flags_obj, dataset, model_name, device):
        if model_name=='MBGCN':
            return MBGCN(flags_obj, dataset, device)
        elif model_name=='MBGCNUD':
            return MBGCNUD(flags_obj, dataset, device)
        else:
            raise ValueError('Model name is not correct!')
