import numpy
import os
import csv
from config import config, log_config

for i in range(0, 100):
    i = float(i)  
    print(i/2)

'''
Name = 0
Number = 10 

csvFile = open('loss/init.csv', 'a') 
writer = csv.writer(csvFile)
for i in range (0,10):
    writer.writerow([Name, Number, 'gou'])
    Name = Name + 1
    Number = Number + 1
csvFile.close()
'''