```python
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
print('z');print('a')
```

    z
    a
    


```python
# Random seed to make results deterministic and reproducible
torch.manual_seed(0)
```




    <torch._C.Generator at 0x289815f6110>




```python
# scaling function for input data
def minmax_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)  # 작은 값을 더해 무한대되는걸 방지
```


```python
# make dataset to input
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length): # i의 시작점은 맨뒤 데이터의 seq길이 앞
        #print('seg:',i)
        _x = time_series[i:i + seq_length, :] # : 인풋의 모든 dimenstion
        _y = time_series[i + seq_length, [-1]]  # [-1]: Next close price
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)
```


```python
# hyper parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 50
```


```python
# load data
xy = np.loadtxt("./stock_data.csv", delimiter=",") #732개의 데이터 셋 (732,5)
xy = xy[::-1]  # reverse order
print(len(xy))
# split train-test set
train_size = int(len(xy) * 0.7) #  732*0.7 = 512개
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # 512-7 = 505  505~732

print(train_size)
print(len(test_set))

```

    732
    512
    227
    


```python
# scaling data
train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set)



# make train-test dataset to input
trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)




# convert to tensor
trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)

testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)



print('trainX_tensor: ',trainX_tensor.shape)
print('trainY_tensor: ',trainY_tensor.shape)
print('testX_tensor: ',testX_tensor.shape)
print('testY_tensor: ',testY_tensor.shape)
```

    trainX_tensor:  torch.Size([505, 7, 5])
    trainY_tensor:  torch.Size([505, 1])
    testX_tensor:  torch.Size([220, 7, 5])
    testY_tensor:  torch.Size([220, 1])
    


```python
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _status = self.rnn(x)
        print(type(x))
        print(len(_status))
        print('output_shape:',x.shape)
        print('_hidden[0]:',_status[0].shape)
        print('_hidden[1]:',_status[1].shape)

        x = self.fc(x[:, -1]) # -1: close price
        
        return x

```


```python
net = Net(data_dim, hidden_dim, output_dim, 1)
```


```python
# loss 
criterion = torch.nn.MSELoss()
# optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
```


```python
weig = list(net.parameters())
weig
```




    [Parameter containing:
     tensor([[-1.8745e-01,  3.1570e-01, -3.0348e-01, -2.8175e-01,  1.9423e-01],
             [ 3.3050e-02,  1.8243e-02, -1.7511e-01, -1.3265e-01, -9.2461e-02],
             [-3.0805e-01,  1.6434e-02,  5.5928e-02, -2.6771e-04,  1.0212e-01],
             [ 3.0005e-01,  8.4080e-02, -1.1576e-01, -1.3014e-01, -2.0232e-01],
             [-2.1922e-01, -5.0928e-02, -5.5930e-02,  1.4068e-01, -1.3517e-01],
             [ 2.5210e-01, -2.2189e-01,  8.9878e-04,  2.8426e-01,  3.1445e-01],
             [-1.8318e-01,  5.6329e-02,  3.7354e-02, -1.4826e-01, -1.0926e-01],
             [ 8.5652e-02, -2.1989e-01,  5.2176e-02,  1.3684e-01, -1.2462e-01],
             [ 2.6267e-01, -2.0811e-02,  1.4348e-01,  3.1316e-01, -9.6661e-02],
             [ 1.7184e-01, -9.0450e-02, -4.6192e-02, -5.3591e-02, -1.9646e-03]],
            requires_grad=True),
     Parameter containing:
     tensor([[-0.1195,  0.0741,  0.0119,  0.2005, -0.0640,  0.0317, -0.1176, -0.2648,
               0.1280,  0.0405],
             [-0.1266, -0.1069,  0.0827, -0.0572,  0.2066,  0.0179,  0.1195,  0.1378,
              -0.0776,  0.1368],
             [ 0.2334,  0.0146,  0.0619,  0.0115,  0.2189, -0.1335, -0.1681,  0.1378,
              -0.2752,  0.0051],
             [-0.1454,  0.2087, -0.2944,  0.1917,  0.3129,  0.0640, -0.0211,  0.3113,
              -0.1351, -0.0253],
             [-0.1385, -0.0437,  0.0687,  0.0422, -0.0597, -0.3162,  0.0162, -0.0094,
               0.0476,  0.2423],
             [ 0.3074, -0.1876, -0.0197, -0.1267, -0.2914, -0.2301,  0.2968,  0.1844,
               0.1699,  0.1757],
             [-0.2512,  0.0332,  0.2943, -0.1765,  0.2813,  0.0831,  0.2225, -0.1355,
               0.1458, -0.2803],
             [-0.0201,  0.1055,  0.0948,  0.2646,  0.3107,  0.2877,  0.2123,  0.2223,
              -0.0408, -0.2454],
             [-0.1174, -0.0355, -0.1709,  0.1616,  0.1110,  0.0510,  0.0764,  0.2778,
               0.1152, -0.2773],
             [-0.2295,  0.1394,  0.0431,  0.1542, -0.3158, -0.2918,  0.1939,  0.2023,
              -0.2861,  0.1200]], requires_grad=True),
     Parameter containing:
     tensor([-0.2469,  0.2395,  0.0992,  0.3127, -0.3118, -0.3044,  0.2138,  0.2808,
              0.1937,  0.2102], requires_grad=True),
     Parameter containing:
     tensor([-0.2663,  0.2289, -0.2986,  0.0706, -0.2071, -0.1256,  0.0560, -0.1243,
             -0.0500, -0.2997], requires_grad=True),
     Parameter containing:
     tensor([[-0.2982,  0.1225, -0.1339, -0.1924, -0.1248, -0.2585, -0.2528,  0.1892,
               0.1607,  0.0421]], requires_grad=True),
     Parameter containing:
     tensor([-0.1727], requires_grad=True)]




```python
for i in weig:
    print(i.shape)
```

    torch.Size([10, 5])
    torch.Size([10, 10])
    torch.Size([10])
    torch.Size([10])
    torch.Size([1, 10])
    torch.Size([1])
    

### Training


```python
for i in range(iterations):
    
    optimizer.zero_grad()
    outputs = net(trainX_tensor)
    loss = criterion(outputs, trainY_tensor)
    net.weight_ih.data
    loss.backward()
    optimizer.step()
    
    print("epoch:{}, Loss:{}".format(i, loss.item()))
```


```python
plt.plot(testY)
plt.plot(net(testX_tensor).data.numpy())
plt.legend(['original', 'prediction'])
plt.show()
```


```python

```


```python

```


```python

```
