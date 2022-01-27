import numpy as np
import torch.nn as nn #仅用于保存模型参数 

DATA_PATH = "handin.utf8"
STATES = 2
OUTPUT_ALPHABET = 3
"""
B矩阵只考虑在i状态时发射k的概率，不考虑转移
"""

class HMM(nn.Module):
    def __init__(self,states,output_alphabet) -> None: 
        self.states = states #隐藏状态个数
        self.output_alphabet = output_alphabet #可能的输出种类:10

        # self.initial = np.random.rand(self.states,1) #随机初始化,(1,states)
        self.initial = np.ones((self.states,1))
        self.initial = self.initial/np.sum(self.initial) #和为1

        # self.transfer = np.random.rand(self.states,self.states) #随机初始化，每行和为1,(states,states)
        self.transfer = np.ones((self.states,self.states))
        self.transfer = self.transfer/np.sum(self.transfer,axis=1,keepdims=True)

        self.emission = np.ones((self.states,self.output_alphabet))
        # self.emission = np.ones((output_alphabet,states,states))
        self.emission = self.emission/np.sum(self.emission,axis=1,keepdims=True)
        return

    def forward(self,sequence):
        """
        计算序列的alpha矩阵, a[i,t] = 给定t时刻和之前的观察序列，处于i状态的累积概率值
        """
        time_steps = len(sequence)
        alpha = np.zeros((self.states,time_steps+1))
        alpha[:,0] = self.initial.flatten() #alpha在t0时与initial相等
        for t in range(1,time_steps+1):
            #更新后一列
            alpha[:,t] = np.sum(
                (alpha[:,t-1].reshape(self.states,1) #取出列向量
                *self.transfer #按元素相乘
                *self.emission[:,sequence[t-1]]
                ),0).T #按列求和后转置
        print_helper(sequence,self,alpha)
        return alpha
    
    def backward(self,sequence):
        """
        计算序列的beta矩阵, b[i,t] = 给定t时刻和之后的观察序列，处于i状态的累积概率值
        """
        time_steps = len(sequence)
        beta = np.zeros((self.states,time_steps+1))
        beta[:,time_steps] = np.ones(self.states) #beta在T+1步时全1
        for t in range(time_steps,0,-1): #倒序遍历
            #更新前一列
            beta[:,t-1] = np.sum(self.emission[:,sequence[t-1]] \
                * self.transfer \
                * beta[:,t],axis=1)
        print(beta)
        return beta

    def calculate_helper_matrix(self,alpha,beta,sequence):
        """
        计算序列的gamma矩阵和P矩阵
        gamma[i,t] = 给定观察序列，在t时刻处于i状态的累积概率值
        P[t,i,j] = 给定观察值，在t时刻状态为i,t+1时刻是j的概率
        """
        time_steps = len(sequence)
        total_p = (alpha * beta)[:,0:time_steps] #去掉最后一列

        gamma = total_p/np.sum(total_p,0) #(states,time_step)
        print(gamma)

        p = np.zeros((time_steps,self.states,self.states)) #(time_step,states,states)
        for t in range(time_steps): #每个时间步
            #pt (states,states)
            p_t = alpha[:,t].reshape(self.states,1) \
                * self.transfer * self.emission[:,sequence[t]] \
                * beta[:,t+1]
            p[t]=p_t/np.sum(p_t) 
        print(p)
        return gamma,p

    def update_para(self,gamma,p,sequence):
        """
        更新参数
        """
        self.initial = gamma[:,0]
        self.transfer = np.sum(p,axis=0)/np.sum(gamma,axis=1,keepdims=True)
        for k in range(self.output_alphabet):
            self.emission[:,k] = np.sum(np.sum(p[np.argwhere(sequence==k)],axis=0),1)[0]/ \
                np.sum(np.sum(p,axis=0),1)

        print("pi")
        print(self.initial)
        print("A")
        print(self.transfer)
        print("B")
        print(self.emission)
        return
    
    def get_sequence_prob(self,sequence):
        """
        计算给定观察序列出现的可能性
        """
        alpha = self.forward(sequence)
        return np.sum(alpha[:,-1])

    def train(self,epochs):
        data = load_data(DATA_PATH)
        total_sequence = data.flatten()
        for epoch in range(epochs):
            print("---------------------epoch %s---------------------"%epoch)
            for i,sequence in enumerate(data):
                # 训练
                alpha = self.forward(sequence)
                # print_helper(sequence,model,alpha)

                beta = self.backward(sequence)
                # print_helper(sequence,model,beta)

                gamma,p = self.calculate_helper_matrix(alpha,beta,sequence)
                # print(alpha)
                # print(beta)
                # print_helper(sequence,model,p)

                self.update_para(gamma,p,sequence)
                # 评估
                prob = self.get_sequence_prob(total_sequence) #计算整个序列的可能性
                print("epoch %s, batch %s, prob %s"%(epoch,i,prob))
            
def load_data(data_path):
    """
    加载数据，返回(样本数，单个样本长度)的np数组
    """
    with open(data_path,'r',encoding="UTF-8") as f:
        data = f.read().split("\n")
    result = np.zeros((len(data),len(data[0])),dtype=int) #结果是包含每个观察结果的一维数组
    for i,str in enumerate(data):
        result[i,:] = list(map(lambda str:int(str),list(str)))
    return result

def print_helper(sequence,model,matrix):
    print("---------------sequence---------------")
    print(sequence)
    print("---------------pi---------------")
    print(model.initial)
    print("---------------A---------------")
    print(model.transfer)
    print("---------------B---------------")
    print(model.emission)
    print("---------------result---------------")
    print(matrix)
    
if __name__ == "__main__":
    model = HMM(STATES,OUTPUT_ALPHABET)
    model.train(epochs = 3)