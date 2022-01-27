import numpy as np
import torch #仅用于保存模型参数 

DATA_PATH = "observation.utf8"
CHECKPOINT_PATH = "hmm.pkl"
STATES = 4
OUTPUT_ALPHABET = 10

class HMM():
    def __init__(self,states,output_alphabet) -> None: 
        self.states = states #隐藏状态个数
        self.output_alphabet = output_alphabet #可能的输出种类:10

        self.initial = np.random.rand(self.states,1) #随机初始化,(1,states)
        # self.initial = np.ones((self.states,1))
        self.initial = self.initial/np.sum(self.initial) #和为1

        self.transfer = np.random.rand(self.states,self.states) #随机初始化，每行和为1,(states,states)
        # self.transfer = np.ones((self.states,self.states))
        self.transfer = self.transfer/np.sum(self.transfer,axis=1)

        # self.emission = np.ones((output_alphabet,states,states)) #(output_alphabet,states,states)
        self.emission = np.random.rand(output_alphabet,states,states)
        self.emission = self.emission/np.sum(self.emission,axis=0)
        return

    def forward(self,batch_data):
        """
        计算一批序列的alpha矩阵, a[i,t] = 给定t时刻和之前的观察序列，处于i状态的累积概率值
        """
        batch_size = batch_data.shape[0]
        time_steps = batch_data.shape[1]
        batch_alpha = np.zeros((batch_size,self.states,time_steps+1))
    
        batch_alpha[:,:,0] = np.tile(self.initial,(batch_size,1)).reshape(batch_alpha[:,:,0].shape) #alpha在t0时与initial相等
        for t in range(1,time_steps+1):
            for idx,sequence in enumerate(batch_data):
                #更新后一列
                batch_alpha[idx,:,t] = np.sum(((batch_alpha[idx,:,t-1].reshape(self.states,1)) \
                    * self.transfer \
                        * self.emission[sequence[t-1]]),0).T #按列求和后转置
        return batch_alpha
    
    def backward(self,batch_data):
        """
        计算一批序列的beta矩阵, b[i,t] = 给定t时刻和之后的观察序列，处于i状态的累积概率值
        """
        batch_size = batch_data.shape[0]
        time_steps = batch_data.shape[1]

        batch_beta = np.zeros((batch_size,self.states,time_steps+1))
        batch_beta[:,:,time_steps] = np.ones(batch_beta[:,:,time_steps].shape)#beta在T+1步时全1

        for t in range(time_steps,0,-1): #倒序遍历
            for idx,sequence in enumerate(batch_data):
                #更新前一列
                batch_beta[idx,:,t-1] = (self.emission[sequence[t-1]] * self.transfer).dot(
                    batch_beta[idx,:,t].reshape(self.states,1)).flatten()
        return batch_beta

    def calculate_helper_matrix(self,alpha,beta,batch_data):
        """
        计算一批序列的gamma矩阵和P矩阵
        gamma[i,t] = 给定观察序列，在t时刻处于i状态的累积概率值
        P[t,i,j] = 给定观察值，在t时刻状态为i,t+1时刻是j的概率
        """
        batch_size = batch_data.shape[0]
        time_steps = batch_data.shape[1]
        
        batch_total_p = alpha * beta #(batch_size,states,time_step+1)
        batch_gamma = np.zeros((batch_size,self.states,time_steps))
        batch_p = np.zeros((batch_size,time_steps,self.states,self.states))

        for idx,sequence in enumerate(batch_data):
            #计算gamma
            total_p = batch_total_p[idx]
            batch_gamma[idx] = (total_p/np.tile(np.sum(total_p,0),(self.states,1)))[:,0:time_steps] #去掉最后一列，变成(states,time_step)
            
            #计算p
            p = np.zeros((time_steps,self.states,self.states)) #(time_step,states,states)
            for t in range(time_steps): #每个时间步
                #pt (states,states)
                p_t = alpha[idx,:,t].reshape(self.states,1) \
                    * self.transfer * self.emission[sequence[t]] \
                    * beta[idx,:,t+1]
                p[t]=p_t/np.sum(p_t) 
            batch_p[idx] = p

        return batch_gamma,batch_p

    def update_para(self,batch_gamma,batch_p,batch_data):
        """
        统计一批样本的数据来更新一次参数
        batch_gamma (batch_size,states,time_step)
        batch_p (batch_size,time_steps,states,states)
        """
        self.initial = np.mean(batch_gamma,axis=0)[:,1] #计算一批的初始概率的均值当作新的初始概率

        p = np.sum(batch_p,axis=0) #(states,time_step)
        self.transfer = np.sum(p,axis=0) / \
            np.sum(np.sum(batch_gamma,axis=0),axis=1,keepdims=True)

        for k in range(self.output_alphabet):
            # 一批数据中out_k[i,j]= 从i到j的转移中发射k的概率之和
            out_k =np.zeros((self.states,self.states)) #分子
            total = np.zeros((self.states,self.states)) #分母
            for idx,sequence in enumerate(batch_data):
                p = batch_p[idx]
                out_k += np.sum(p[np.argwhere(sequence==k)],axis=0).reshape(self.states,self.states)
                total += np.sum(p,axis=0)
            self.emission[k] = out_k/total

        # print_helper(self)
        return
    
    def get_sequence_prob(self,sequence):
        """
        计算给定观察序列出现的可能性
        """
        alpha = self.forward(sequence)
        return np.sum(alpha[:,:,-1])

    def train(self,epochs,batch_size):
        data = load_data(DATA_PATH)
        N = data.shape[0] #总样本数
        for epoch in range(epochs):
            print("---------------------epoch %s---------------------"%epoch)
            batch_ptr = 0
            batch_idx = 0
            while(batch_ptr!=N):
                batch_data = data[batch_ptr:min(N,batch_ptr+batch_size),:]
                batch_ptr = min(N,batch_ptr+batch_size)
                batch_idx=batch_idx+1
                # 训练
                alpha = self.forward(batch_data)
                beta = self.backward(batch_data)
                gamma,p = self.calculate_helper_matrix(alpha,beta,batch_data)
                self.update_para(gamma,p,batch_data)
                #评估
                prob = self.get_sequence_prob(batch_data) 
                print("epoch %s, batch %s, prob %s"%(epoch,batch_idx,prob))
            self.saveModel(CHECKPOINT_PATH)

    def saveModel(self,path):
            checkpoint = {'initial': self.initial,
                'transfer':self.transfer,
                'emission':self.emission,}
            torch.save(checkpoint, path)
            print("finish saving model")
            
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

def print_helper(model):
    print("---------------pi---------------")
    print(model.initial)
    print("---------------A---------------")
    print(model.transfer)
    print("---------------B---------------")
    print(model.emission)
    
if __name__ == "__main__":
    model = HMM(STATES,OUTPUT_ALPHABET)
    model.train(epochs = 10000,batch_size=100)