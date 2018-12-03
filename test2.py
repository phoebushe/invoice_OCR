import os
import numpy as np

# path = 'data/results/'
# filenames = os.listdir(path)
path = 'data/results/'
filenames = os.listdir(path)
for filename in filenames:
    dict = []
    with open(path + filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            temp = line.split(',')
            temp = list(map(int, temp))
            dict.append(temp)
        array = np.array(dict)
        results = array[array[:, 1].argsort()]
        with open(path + filename, 'w') as f:
            for result in results:
                sort_result = ','.join([str(result[0]), str(result[1]), str(result[2]), str(result[3])]) + '\r\n'
                f.write(sort_result)




