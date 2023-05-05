import torch
import numpy as np
relation = ['buy','cart','click','collect']
for aa in relation:
    index = []
    with open('dataset/Tmall/'+str(aa)+'.txt') as f:
        data = f.readlines()
        for row in data:
            user, item = row.strip().split()
            user, item = int(user), int(item)
            index.append([user, item])
    a = torch.tensor(index)
    index = []
    for i in range(10000):
        b = a[a[:,0]==i][:,1]
        for j in range(b.shape[0]):
            for k in range(j, b.shape[0]):
                index.append([b[j].item(), b[k].item()])
    b = np.unique(index, axis=0)
    b = torch.LongTensor(b)
    c = torch.sparse.FloatTensor(b.t(), torch.ones(b.shape[0],dtype=torch.int), torch.Size([3000,3000]))
    print(c.shape)
    torch.save(c.to_dense(), 'dataset/Tmall/item_'+str(aa)+'.pth')

'''relation = ['buy','cart','click','collect','test','validation','train']
for i in relation:
    index = []
    with open('dataset/Tmall/'+str(i)+'.txt') as f:
        data = f.readlines()
        for row in data:
            user, item = row.strip().split()
            if int(user) >= 10000 or int(item) >= 3000:
                continue
            else:
                index.append([user, item])
    with open('dataset/Tmall/new/'+str(i)+'.txt',mode='w+') as f:
        for user, item in index:
            f.write(user+' '+item+'\n')'''