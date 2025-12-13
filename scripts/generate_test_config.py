import os
# This is located in the wrong directory, was done inside test/input

dic = {}
for file in os.listdir():
    if file.endswith('canonical'):
        with open(file, 'r') as f:
            line = f.readline()
            n, m = line.split(' ') 
            n, m = int(n), int(m)
        size = n * m
        name = file.replace('.canonical', '')
        groundtruth = "../groundtruth/" + file.replace('.canonical', '.mps.txt')
        with open(groundtruth, 'r') as f:
            gt_val = float(f.readline())


        dic[name] = {
                    'm' : m,
                    'n' : n,
                    'size' : size,
                    'file' : f"/home/ubuntu/simplex_method_gpu/test/input/{file}",
                    'expected_optimum': gt_val,
                    'description': 'whatever'
                }

print(dic)
