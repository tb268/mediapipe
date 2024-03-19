import numpy as np

landmarks1=np.array([1,2,3])
landmarks2=np.array([1,11,3])
landmarks3=np.array([4,5,6])
landmarks4=np.array([4,33,6])

x=[landmarks1,landmarks2]
y=[landmarks3,landmarks4]
for index,(landmark1,landmark2) in enumerate(zip(x,y)):
    print(index,landmark1,landmark2)
    pass