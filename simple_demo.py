import numpy as np

for i in range(10):
    b = 88
    a = np.random.randint(1, 10)
    # print(a, end=' ')
    # if a % 3 == 0:
    #     print('a-',a)

# print(b)

q = 'sfdsfs,sdfdsfdsf, , ,ggggg'
qq = q.replace(',', '1')

w = [1,2,3,4,5]
# print(w[0:5])

def run_epoch1(FLAGS):
    print(FLAGS.ckpt_path)