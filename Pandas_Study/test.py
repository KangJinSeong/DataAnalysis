import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('test.csv', header= None)
df = df.T
df.columns = ['data']
df['count'] = list(range(0, 4000))
print(df)
print(df.info())
print(df.max())
size = (df.data / df.data.max()) * 300
df.plot(kind = 'scatter',x = 'count', y = 'data',marker = '+', cmap = 'viridis', figsize= (10,5),
        c = size, s = 50, alpha = 0.3)

plt.savefig('측정데이터.png', transparent = True)
plt.show()
