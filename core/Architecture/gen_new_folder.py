import os
import glob


l = []
for folder in glob.glob('./N*'):
    folder = folder.split('N')[-1]
    l.append(int(folder))

new=max(l)+1
os.system(f'mkdir N{new}')
os.system(f'cp N0/network.py ./N{new}/')
