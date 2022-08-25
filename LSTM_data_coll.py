import pandas as pd

df = pd.read_csv('gt.txt', sep=',', header=None)

new_df = df.loc[df[1] == 3]

size = new_df.shape[0]
print(size)

delx, dely, delw, delh = [], [], [], []

for i in range(size-1):
    delxs = new_df.iat[i+1, 2] - new_df.iat[i, 2]
    delys = new_df.iat[i+1, 3] - new_df.iat[i, 3]
    delws = new_df.iat[i+1, 4] - new_df.iat[i, 4]
    delhs = new_df.iat[i+1, 5] - new_df.iat[i, 5]

    delx.append(delxs)
    dely.append(delys)
    delw.append(delws)
    delh.append(delhs)

del_df = pd.DataFrame((delx, dely, delw, delh))

#del_df.to_csv(r'pandas.txt', header=False, index=False, sep=',', mode='a')



