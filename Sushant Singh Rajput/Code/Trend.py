import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'..\Sentiment\YouTube\Timed Sentiment\TimesNowYTRheaChak_TimedSentiment.csv')
data = data[['sentiment']]
x = range(len(data))
y = data['sentiment']
plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.show()