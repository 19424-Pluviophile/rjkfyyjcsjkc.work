import numpy as np

a2 = np.array([4,5,6])
print(type(a2))
print(a2.shape)
print(a2[0])

b3 = np.array([[4,5,6],[1,2,3]])
print(b3.shape)
print(b3[0,0])
print(b3[1,1])

a4 = np.zeros((3,3),dtype=int)
b4 = np.ones((4,5),dtype=int)
c4 = np.identity(4)
d4 = np.random.randn(3,2)

a5 = np.arange(1,13).reshape(3,4)
print(a5)
print(a5[2,3])
print(a5[0,0])

b6 = a5[0:2, 1:3]
print(b6)
print(b6[0, 0])

c7 = a5[1:3, :]
print(c7)
print(c7[0][-1])

a8 = np.array([[1,2],[3,4],[5,6]])
print(a8[[0,1,2],[0,1,0]])

a9 = np.arange(1,13).reshape(4,3)
b9 = np.array([0,2,0,1])
print(a9[[np.arange(4),b9]])

a9[[np.arange(4),b9]] += 10
print(a9[[np.arange(4),b9]])

x11 = np.array([1,2])
print(x11.dtype)

x12 =np.array([1.0,2.0])
print(x12.dtype)

x13 = np.array([[1, 2], [3, 4]], dtype=np.float64)
y13 = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x13 + y13)
np.add(x13,y13)

print(x13-y13)
np.subtract(x13,y13)

x13 * y13
print(x13 * y13)
np.multiply(x13,y13)
print(np.multiply(x13,y13))
np.dot(x13,y13)
print(np.dot(x13,y13))

print(np.divide(x13,y13))

print(np.sqrt(x13))

print(x13.dot(13))
print(np.dot(x13,y13))

print(np.sum(x13)) # 10
print(np.sum(x13,axis=0)) # [4. 6.] 两列之和
print(np.sum(x13,axis=1)) # [3. 7.] 两行之和

print(np.mean(x13))
print(np.mean(x13,axis=0))
print(np.mean(x13,axis=1))

x13.T
print(x13.T)

print(np.exp(x13))

print(np.argmax(x13))
print(np.argmax(x13,axis=0))
print(np.argmax(x13,axis=1))

import matplotlib.pyplot as plt
x = np.arange(0,100,0.1)
y = x * x
plt.figure(figsize=(6,6))  # 创建画布，并指定画布大小
plt.plot(x,y)   # 在画布上画图
plt.show()  # 展示画图结果

x = np.arange(0,3*np.pi,0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.figure(figsize=(10,6))
plt.plot(x,y1,color='Red')
plt.plot(x,y2,color='Blue')
plt.legend(['Sin','Cos'])  # 给两条线做标记
plt.show()