import numpy as np
import networkx as nx
import re
import matplotlib.pyplot as plt
import pandas as pd

fixed_df = pd.read_csv('C:/Users/epish/Desktop/учеба/ДИПЛОМ/test.csv', sep = ';')
trying = [list(fixed_df.loc[i])[1:] for i in range(len(fixed_df))]
for i in range(len(trying)):
  for j in range(len(trying[i])):
    trying[i][j] = float(re.sub(r',', '.', trying[i][j]))
test = np.array(trying)
G=nx.from_numpy_matrix(test)
l = ['2rbina2rista','basta','brb','eldzhey','face','feduk','gnoiny','guf','husky','kasta','korzh','krovostok','kunteynir','lizer','lsp','morgenshtern','noizemc','skriptonite','timati','xleb']
mapping = {i:l[i] for i in range(len(l))}
H = nx.relabel_nodes(G, mapping)
nx.draw(H,with_labels=True, node_shape = 'p', font_color = 'red')
plt.show()

