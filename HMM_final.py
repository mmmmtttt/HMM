from os import times, times_result
import numpy as np
from numpy.core.fromnumeric import argsort
import torch

class HMM():
    def __init__(self, states=4, output_alphabet=10):
        self.states = states #隐藏状态数
        self.output_alphabet = output_alphabet #可能的输出种类

        #初始化概率矩阵
        self.initial = np.ones(states) * (1.0 / states)  #初始状态
        self.transfer = np.ones((states, states)) * (1.0 / states)  #状态转移矩阵
        self.emission = np.random.random(size=(self.states,self.output_alphabet)) #发射矩阵
        for s in range(self.states):
            self.emission[s] = self.emission[s]/np.sum(self.emission[s])
        
    def forward(self, sequence, initial):
        """
        计算序列的alpha矩阵, a[i,t] = 给定t时刻和之前的观察序列，处于i状态的累积概率值
        """
        time_step = len(sequence)
        alpha = np.zeros((time_step, self.states))  
        alpha[0] = self.emission[:,sequence[0]] * self.initial * initial[0] # 初始值
        # 归一化因子
        c = np.zeros(time_step)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]

        for t in range(1,time_step):
            alpha[t] = self.emission[:,sequence[t]] * np.dot(alpha[t - 1], self.transfer) * initial[t]
            c[t] = np.sum(alpha[t])
            if c[t]==0: continue
            alpha[t] = alpha[t] / c[t]
        return alpha, c

    def backward(self, sequence, initial, c):
        """
        计算一批序列的beta矩阵, b[i,t] = 给定t时刻和之后的观察序列，处于i状态的累积概率值
        """
        time_step = len(sequence)
        beta = np.zeros((time_step, self.states))  
        beta[time_step - 1] = np.ones((self.states)) #beta最后一步时全1
        
        for t in reversed(range(time_step-1)):
            beta[t] = np.dot(beta[t + 1] * self.emission[:,sequence[t + 1]], self.transfer.T) * initial[t]
            if c[t+1]==0: continue
            beta[t] = beta[t] / c[t + 1]
        return beta

    def train(self, epoch,batch_data):
        """
        训练批数据
        处理初始状态：用batch_initial单独记录每个序列的初始概率pi
        对一批数据计算完成后，统计，作一次参数的更新
        """
        batch_size = batch_data.shape[0]
        time_step = batch_data.shape[1]
        batch_initial = np.ones((batch_size,time_step,self.states)) # 初始化状态序列,每个样本都有
        for e in range(epoch): 
            print("epoch %d"%e)
            batch_gamma = []  
            batch_gamma = np.zeros((batch_size,self.states,time_step))
            batch_p = np.zeros((self.states, self.states)) 
            batch_start_prob = np.zeros(self.states) 

            for idx,(sequence,initial) in enumerate(batch_data,batch_initial):
                time_step = len(sequence)
                alpha, c = self.forward(sequence, initial)  
                beta = self.backward(sequence, initial, c)  

                #gamma[i,t] = 给定观察序列，在t时刻处于i状态的累积概率值
                gamma = alpha * beta / np.sum(alpha * beta) 
                batch_gamma[idx] = gamma
                batch_p += self.calculate_p(c,sequence,time_step,alpha,beta)  
                batch_start_prob += batch_gamma[idx,0] 
            
            ### 对一批数据更新参数
            #用一批数据的初始状态概率的均值作为更新的新值
            batch_start_prob += 0.001*np.ones(self.states) #加一个很小的数，保证其不为0，不然计算出来有nan
            self.initial = batch_start_prob / np.sum(batch_start_prob) 

            batch_p += 0.001
            for s in range(self.states):
                if np.sum(batch_p[s])==0: continue
                self.transfer[s] = batch_p[s] / np.sum(batch_p[s])

            self.update_emission(batch_data.flatten(),batch_gamma.flatten())
            self.saveModel(CHECKPOINT_PATH)
    
    def calculate_p(self,c,sequence,time_step,alpha,beta):
        """
        计算Pt,P[t,i,j] = 给定观察值，在t时刻状态为i,t+1时刻是j的概率
        """
        pt = np.zeros((self.states, self.states)) 
        for t in range(1,time_step):
            if c[t]==0: continue
            pt += (1 / c[t]) * np.outer(alpha[t - 1],
                beta[t] * self.emission[:,sequence[t]]) * self.transfer
        if np.sum(pt)!=0:
            pt = pt/np.sum(pt) 
        return pt

    def update_emission(self, sequence, gamma):
        """
        更新发射概率矩阵
        """
        time_step = len(sequence)
        self.emission = np.zeros((self.states, self.output_alphabet))
        for t in range(time_step):
            self.emission[:,int(sequence[t])] += gamma[t]

        self.emission+= 0.1/self.output_alphabet
        for s in range(self.states):
            if np.sum(gamma[:,s])==0: continue
            self.emission[s] = self.emission[s]/np.sum(gamma[:,s])

    def get_possibility(self, sequence):
        """
        计算指定序列的生成概率值
        """
        Z = np.ones((len(sequence), self.states))
        alpha, _ = self.forward(sequence, Z)  
        # 序列的出现概率估计
        prob_X = np.sum(alpha[:,-1])
        return prob_X
    
    def predict_sequences(self,num=10,len=7):
        total = 10000000
        possibilities = np.zeros(total)
        for i in range(total):
            s = "%07d"%i #补0成7位数
            x = np.array(list(map(lambda x:int(x),list(s))))
            p = self.get_possibility(x)
            possibilities[i] = p
            print("seq:%s p:%s"%(i,p))
        top_index = possibilities.argsort()[::-1][0:10] #得到概率最大的10个序列
        np.save(POSSIBILITIES_PATH,possibilities)
        np.save(RESULT_PATH,top_index)
        print(top_index)

    def saveModel(self,path):
            checkpoint = {'initial': self.initial,
                'transfer':self.transfer,
                'emission':self.emission,}
            torch.save(checkpoint, path)
            print("finish saving model")
    
    def load_model(self):
        checkpoint = torch.load(CHECKPOINT_PATH)
        self.initial = checkpoint['initial']
        self.transfer = checkpoint['transfer']
        self.emission = checkpoint['emission']

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

DATA_PATH = "observation.utf8"
CHECKPOINT_PATH = "hmm.pkl"
POSSIBILITIES_PATH = "possibilities.npy"
RESULT_PATH = "top10.npy"
STATES = 4
OUTPUT_ALPHABET = 10

if __name__=="__main__":
    model = HMM()
    data = load_data(DATA_PATH)
    # model.train_batch(10,data)
    model.load_model()
    model.predict_sequences()
    