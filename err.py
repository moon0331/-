# comments which are list form with numbers are the very first data calculated by this code.

import math

def printArr(string, lst):
    print('-------------------')
    print(string)
    print(lst)
    print('-------------------')

def _sigmoid(x):
    return 1/(1+math.exp(-x))

class neuron():
    def __init__(self,lst): #set connection weight
        self.x1=lst[0]
        self.x2=lst[1]
        self.x3=lst[2]
        
    def fix_w(self, lst, lr):
        self.x1-=(lr*lst[0])
        self.x2-=(lr*lst[1])
        self.x3-=(lr*lst[2])

    def print_w(self):
        print([self.x1,self.x2,self.x3])
        
    def input(self,a,b,c=1,sigmoid=True):
        self.result = self.x1 * a + self.x2 * b + self.x3 * c
        return self.getResult(sigmoid)
    
    def getResult(self, sigmoid=True):
        if sigmoid is True:
            return _sigmoid(self.result)
        else:
            return self.result

def calDel_output(k, j, n, t, o, h): #t_nk, o_nk, h_nj
    k-=1
    j-=1
    n-=1
    return -(t[n]-o[n])*o[n]*(1-o[n])*h[j][n]

def calDel_hidden(i,j,n,x,h,w,t,o): #x_input,w_output,t_nk,o_nk
    #print('===========================')
    #print('i={}, j={}, n={}'.format(i,j,n))
    i-=1
    j-=1
    n-=1
    return -x[n][i]*h[j][n]*(1-h[j][n])*w[j]*(t[n]-o[n])*o[n]*(1-o[n])

if __name__=='__main__':

    learning_rate=0.5

    x_input=[[1,1,1,0],[1,0,1,1],[0,1,1,1],[0,0,1,0]]
    
    w_hidden=[[-0.089, 0.028, 0.092],[0.098, -0.07, -0.01]]
    w_output=[0.056,0.067,0.016]

    hidden_layer=[neuron(w_hidden[0]), neuron(w_hidden[1])] #set neuron with connection weight
    output_layer=neuron(w_output)                           #set neuron with connection weight
    
    t_nk=[0,1,1,0]

    print('output neuron connection weight (before learning : )', w_output)
    print('hidden layer neuron connection weight (before learning : )')
    for i in range(2):
        print('neuron {} :'.format(i+1),w_hidden[i])

    how_much_training=int(input('input the number of training : '))

    for epoch in range(how_much_training):
        print("===========================================")
        
        net_nj=[[n.input(1,1,sigmoid=False),
                     n.input(1,0,sigmoid=False),
                     n.input(0,1,sigmoid=False),
                     n.input(0,0,sigmoid=False)] for n in hidden_layer] #calculate net_nj
        printArr('net_nj ', net_nj)
        '''
        [[0.031, 0.0030000000000000027, 0.12, 0.092],
        [0.017999999999999995, 0.08800000000000001, -0.08, -0.01]]
        '''
        
        h_nj=[[n.input(1,1),
               n.input(1,0),
               n.input(0,1),
               n.input(0,0)] for n in hidden_layer] #calculate h_nj, which is result of _sigmoid(net_nj)
        h_nj.append([1,1,1,1]) #bias
        printArr('h_nj ', h_nj)
        '''
        [[0.5077493794138049, 0.5007499994375005, 0.5299640517645717, 0.5229837910524483],
        [0.5044998785039364, 0.5219858136524729, 0.48001065984441826, 0.49750002083312506],
        [1, 1, 1, 1]]
        '''
        
        net_nk=[output_layer.input(h_nj[0][i],h_nj[1][i], sigmoid=False) for i in range(4)] #calculate net_nk
        '''[0.07823545710693681, 0.07901504948321572, 0.07783870110839204, 0.07861959369475648]'''
        o_nk=[_sigmoid(x) for x in net_nk] #calculate o_nk
        '''[0.5195488940761298, 0.5197434912662044, 0.5194498559336176, 0.5196447807004962]    '''
        printArr('net_nk ', net_nk)
        printArr('o_nk ', o_nk)

        dEn_dw_output=[] 
        dEn_dw_hidden=[] 

        dEn_dw_output=[[calDel_output(1,j,n,t_nk,o_nk,h_nj) for n in [1,2,3,4]] for j in [3,2,1]] #calculate dEn/dw_kj
        printArr('dEn_dw_output ',dEn_dw_output)

        '''[[0.12968867309834645, -0.11987692058020538, -0.11995574538880055, 0.12971065520787498],
        [0.0654279198214525, -0.06257405192721138, -0.05758003649620719, 0.0645310536681961],
        [0.06584934328268523, -0.060028367913107145, -0.06357223285868809, 0.06783657020051145]]'''

        dEn_dw_hidden=[[calDel_hidden(i,j,n,x_input,h_nj,w_output,t_nk,o_nk) for n in [1,2,3,4]]
                       for j,i in [(2,3),(2,2),(2,1),(1,3),(1,2),(1,1)]] #calculate dEn/dw_ji
        printArr('dEn_dw_hidden \n',dEn_dw_hidden)
        '''[[0.0021721093287912167, -0.002004056062736408, -0.0020060473473144055, 0.0021725991593002972],
        [0.0021721093287912167, 0.0, -0.0020060473473144055, -0.0],
        [0.0021721093287912167, -0.002004056062736408, 0.0, -0.0],
        [0.0018152052853813313, -0.0016782731120055412, -0.0016733491461779068, 0.001812112038522936],
        [0.0018152052853813313, 0.0, -0.0016733491461779068, -0.0],
        [0.0018152052853813313, -0.0016782731120055412, 0.0, -0.0]]'''

        dE_dw_output=[sum(result) for result in dEn_dw_output] #calculate dE/dw_kj
        dE_dw_hidden=[sum(result) for result in dEn_dw_hidden] #and dE/dw_ji
        printArr('dE/dW for output neuron is : ',dE_dw_output)
        printArr('dE/dW for hidden neuron is : ',dE_dw_hidden)

        '''
        w_hidden=[[-0.089, 0.028, 0.092],[0.098, -0.07, -0.01]]
        w_output=[0.056,0.067,0.016]
        '''

        '''for i in range(len(w_output)):
            print('{}-0.5*{}'.format(w_output[i],dE_dw_output[i]))
        for i in range(2):
            for j in range(len(w_hidden[0])):
                print('{}-0.5*{}'.format(w_hidden[i][j],dE_dw_hidden[3*i+j]))'''

        output_layer.fix_w(dE_dw_output,learning_rate)
        hidden_layer[0].fix_w(dE_dw_hidden[3:],learning_rate)
        hidden_layer[1].fix_w(dE_dw_hidden[:3],learning_rate)

        print('output layer neuron after trained {} times :\t'.format(epoch+1), end='')
        output_layer.print_w() #gradient descent for each neuron
        for i,n in enumerate(hidden_layer): #gradient descent for each neuron
            print('hidden layer neuron {} after trained [{}] times:\t'.format(i,epoch+1), end='')
            n.print_w()


