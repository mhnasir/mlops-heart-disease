import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/raw/heart.csv")
df.head()
df.info()

sns.countplot(x='target', data=df)
plt.show()

df['age'].hist(bins=20)
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()
