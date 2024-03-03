import torch

# 随机生成一个四维张量，形状为 (2, 3, 4, 5)
x = torch.randn(2, 2, 2, 2)

# 打印四维张量的实际值
print("随机生成的四维张量 x：")
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            for l in range(x.shape[3]):
                print("x[{}][{}][{}][{}] = {}".format(i, j, k, l, x[i][j][k][l]))
