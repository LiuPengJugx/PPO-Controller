import os
import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from controller.db.conf import Conf
import matplotlib.pyplot as plt
torch.cuda.set_device(1)
class WorkloadPredictor(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size,device,activation):
        super().__init__()
        self.input_dim=int(np.prod(input_dim))
        self.output_dim=int(np.prod(output_dim))
        self.device=device
        # pretrained=[nn.Linear(self.input_dim,hidden_size[0]),nn.Dropout(p=0.4),activation]
        model=[nn.Linear(self.input_dim,hidden_size[0]),activation]
        for i in range(len(hidden_size) - 1):
            model += [nn.Linear(hidden_size[i], hidden_size[i + 1]),activation]
        model+=[nn.Linear(hidden_size[-1], self.output_dim)]
        self.model = nn.Sequential(*model)
    def forward(self,x):
        x=torch.as_tensor(x,dtype=torch.float)
        return self.model(x.view(1,-1))

class WorkloadClassifier(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_size,device,activation):
        super().__init__()
        self.input_dim=int(np.prod(input_dim))
        self.output_dim=int(np.prod(output_dim))
        self.device=device
        # pretrained=[nn.Linear(self.input_dim,hidden_size[0]),nn.Dropout(p=0.4),activation]
        model=[nn.Linear(self.input_dim,hidden_size[0]),activation]
        for i in range(len(hidden_size) - 1):
            model += [nn.Linear(hidden_size[i], hidden_size[i + 1]),activation]
        model+=[nn.Linear(hidden_size[-1], self.output_dim),nn.Softmax(dim=1)]
        self.model = nn.Sequential(*model)
    def forward(self,x):
        x=torch.as_tensor(x,dtype=torch.float).cuda()
        return self.model(x.view(1,-1))

def read_sample(file_path):
    df = pd.read_csv(file_path, header=None)
    data=[]
    for row in range(df.shape[0]):
        a1 = np.array(json.loads(df.iloc[row][0]))
        a2 = np.array(json.loads(df.iloc[row][1]))
        data.append((a1,a2))
    return data

def plot_error_curve(loss_seq):
    plt.plot(range(1,len(loss_seq)+1), loss_seq)
    plt.xlabel('Epoch Num')
    plt.ylabel("Cross Entropy Loss")
    plt.show()

def train_classifier_process(data_paths,trained_model_path):
    data=[]
    for path in data_paths:
        data += read_sample(path)
    hidden_size = [128,128]
    # input_dim = [Conf.TABLE_ATTRIBUTE_NUM]
    input_dim = np.array(data[0][1]).shape
    output_dim = [2]
    epoch_num = 1500
    # 开始训练
    classifier = WorkloadClassifier(input_dim, output_dim, hidden_size, 'cuda', torch.nn.ReLU())
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    predictor = classifier.cuda()
    # if torch.cuda.device_count() > 1:
    #     predictor = torch.nn.DataParallel(predictor)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(predictor.parameters(), lr=1e-5)
    loss_seq=[]
    for epoch in range(epoch_num):
        total_loss = 0.0
        for idx, sample_data in enumerate(data):
            target_type_prob,input_vector = torch.tensor(sample_data[0]).cuda(), \
                                              torch.tensor(sample_data[1], dtype=torch.float).cuda()

            # forward + backward + optimize
            output_type_prob = predictor(input_vector)
            loss = criterion(output_type_prob,target_type_prob.long())
            # zero the parameter gradients
            optimizer.zero_grad()  # 清空所管理参数的梯度
            loss.backward()
            optimizer.step()  # 执行一步更新
            total_loss += loss.item()
            if pd.isnull(loss.item()):
                print("None None None None ! ! !")
            # print(loss.item())
            # if idx % 2000 == 1999:
            #     print('[%d,%d], loss: %.3f' % (epoch, idx, total_loss / 2000))
            #     total_loss = 0.0
        loss_seq.append(total_loss / len(data))
    plot_error_curve(loss_seq)
    # torch.save(classifier, 'classifier_union_.pkl')
    torch.save(classifier, trained_model_path)
    # torch.save(classifier, 'classifier.pkl')

def train_predictor_process(data_paths):
    data = []
    for path in data_paths:
        data += read_sample(path)
    hidden_size = [128, 128, 128, 128]
    input_dim = [1, Conf.TABLE_ATTRIBUTE_NUM]
    output_dim = [1, Conf.TABLE_ATTRIBUTE_NUM]
    epoch_num = 100
    # 开始训练
    predictor = WorkloadPredictor(input_dim, output_dim, hidden_size, 'cuda', torch.nn.ReLU())
    os.environ['CUDA_VISIBLE_DEVICES']="0,1,2"
    predictor=predictor.cuda()
    if torch.cuda.device_count()>1:
        predictor=torch.nn.DataParallel(predictor)
    # target_matrix = torch.tensor(np.zeros(output_dim), dtype=torch.float32)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(predictor.parameters(), lr=1e-8)
    loss_seq=[]
    for epoch in range(epoch_num):
        total_loss = 0.0
        for idx,sample_data in enumerate(data):
            input_matrix_change,target_matrix_change=torch.tensor(sample_data[0],dtype=torch.float).cuda(),torch.tensor(sample_data[1],dtype=torch.float).view(1,-1).cuda()

            # forward + backward + optimize
            output_matrix_change = predictor(input_matrix_change)
            loss = criterion(output_matrix_change, target_matrix_change)

            # zero the parameter gradients
            optimizer.zero_grad()  # 清空所管理参数的梯度
            loss.backward()
            optimizer.step()  #执行一步更新
            total_loss += loss.item()
            if pd.isnull(loss.item()):
                print("None None None None ! ! !")
            print(loss.item())
            if idx%2000==1999:
                print('[%d,%d], loss: %.3f' % (epoch, idx, total_loss / 2000))
                total_loss = 0.0
        loss_seq.append(total_loss/len(data))

    torch.save(predictor, 'predictor.pkl')


def test_process(criterion,model_path,test_dataset_path):
    # criterion = nn.MSELoss()
    predictor = torch.load(model_path)
    print(predictor)
    # 训练后，测试
    # data = read_sample("attr_change_sample_12000_test.csv")
    data = read_sample(test_dataset_path)
    accuracy=0
    TP,FP,FN,TN=0,0,0,0

    for idx, sample_data in enumerate(data):
        if 'classifier' in model_path:
            target_type_prob,input_vector = torch.tensor(sample_data[0]).cuda(), torch.tensor(
                sample_data[1]).cuda()
            output_matrix = predictor(input_vector)
            print(criterion(output_matrix, target_type_prob.long()).item())
            print(idx+1," ",torch.argmax(output_matrix,1))
            if torch.argmax(output_matrix, 1).item() == 1:
                if target_type_prob.item() == 1:TP+=1
                else: FP+=1
            else:
                if target_type_prob.item() == 1:FN += 1
                else:TN += 1
            if target_type_prob==torch.argmax(output_matrix,1):accuracy+=1
        else:
            input_matrix_change, target_matrix_change = torch.tensor(sample_data[0],dtype=torch.float).cuda(), \
                                                        torch.tensor(sample_data[1], dtype=torch.float).view(1, -1).cuda()
            output_matrix_change = predictor(input_matrix_change)
            print(criterion(output_matrix_change, target_matrix_change))
    precision = TP/(TP+FP) if TP+FP>0 else 0
    recall = TP/(TP+FN) if TP+FN>0 else 0
    print(f"准确率:{accuracy/len(data)},查准率:{precision},查全率:{recall}")

if __name__=="__main__":
    # trained_model_path='classifier_union_v1.pkl'
    # trained_model_path='classifier_union_v2.pkl'
    # trained_model_path='classifier_union_v3.pkl'
    trained_model_path='classifier_union_v4.pkl'
    # train_classifier_process(["par_vector_sample_v1_1500.csv","par_vector_sample_v1_3000.csv","par_vector_sample_v1_4000.csv"],trained_model_path)
    # train_classifier_process(["par_vector_sample_v2_1500.csv","par_vector_sample_v2_3000.csv","par_vector_sample_v2_4000.csv"],trained_model_path)
    # train_classifier_process(["par_vector_sample_v3_1300.csv","par_vector_sample_v3_1500.csv","par_vector_sample_v3_3000.csv","par_vector_sample_v3_4000.csv"],trained_model_path)
    # train_classifier_process(["par_vector_sample_v4_1500.csv","par_vector_sample_v4_3000.csv","par_vector_sample_v4_4000.csv"],trained_model_path)
    # test_process(nn.MSELoss(),'predictor.pkl',"attr_change_sample_1200
    test_process(nn.CrossEntropyLoss(),trained_model_path,"par_vector_sample_v4_4000.csv")

    # unit test
    # a=[23, 345, 34, 45,253423345,26352346,35,533,64376,7546]
    # input_vector=a+[0 for _ in range(50-len(a))]
    # predictor = torch.load(trained_model_path)
    # print(predictor(input_vector))


    """
        版本v3
        样本集：par_vector_sample_1500、3000、4000
        训练模型：classifier_union.pkl
                                        
        测试集：par_vector_sample_5000   准确率:0.7943548387096774,查准率:0.8287292817679558,查全率:0.8823529411764706
              par_vector_sample_2600    准确率:0.663003663003663,查准率:0.6918604651162791,查全率:0.7531645569620253
             
        训练集：par_vector_sample_1300   准确率:0.9632352941176471,查准率:0.9789473684210527,查全率:0.96875
              par_vector_sample_1500   准确率:0.9207317073170732,查准率:0.9508196721311475,查全率:0.7785234899328859
              par_vector_sample_3000   准确率:0.9096774193548387,查准率:0.9758064516129032,查全率:0.75625
              par_vector_sample_4000   准确率:0.9698189134808853,查准率:0.9805194805194806,查全率:0.9710610932475884
    """

