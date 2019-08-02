#reference: http://blog.csdn.net/u014365862/article/details/53869802
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
import numpy as np

def moving_average(l,N):
    sum = 0
    result = list(0 for x in l)
    
    for i in range(0,N):
        sum = sum + l[i]
        result[i] = sum/(i+1)
        
    for i in range(N, len(l)):
    sum = sum - l[i-N] + l[i]
    result[i] = sum/N
    
    return result



url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/��·������.csv'
ass_data = requests.get(url).content

#ass_data = '��·������.csv'

df = pd.read_csv(io.StringIO(ass_data.decode('utf-8'))) #python2 ʹ��StringIO.StringIO

data = np.array(df['��·������_����ֵ(����)'])

normalize_data = (data - np.mean(data))/np.std(data)

#ma_data = moving_average(data.to_list(), 3) #����ƽ��

plt.figure()
plt.plot(data, color = 'g')
plt.plot(ma_data, color = 'r')
plt.show()