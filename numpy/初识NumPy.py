import numpy as np

# 创建一维ndarray 对象

data = [1, 34, 44, 6]
arr = np.array(data)
print(arr)
print(type(arr))
print('---------')

# 向量到python内置的list类型的转换

arr = np.arange(8)
L = arr.tolist()
print(type(L))
print(L)
print('---------')

# 构建多维数组
data = [[1., 2, 3, 4], [5, 6, 7, 8]]
arr = np.array(data)
print(arr)
print(arr.ndim)  # 数组维数
print(arr.shape)  # 形状 (x行x列)
print(arr.dtype)  # 数据类型
print(type(arr))

print('---------')

arr1 = np.array([1, 2, 3, 4], dtype=np.float64)
arr2 = np.array([1, 2, 3, 4], dtype=np.int32)
arr3 = arr2.astype(np.float64)  # 调用astype时会创建一个新的数组,对原始数据的一个深拷贝
print(arr1)
print(arr2)
print(arr3)

print('---------')
arr_0 = np.zeros(8)  # 将数组的值初始化为0
arr_1 = np.ones((3, 8))  # 将数组的值初始化为1
arr_e = np.empty((2, 3, 3))  # 未经初始化的值
print(arr_0)
print(arr_1)
print(arr_e)
print('---------')

arr2 = np.arange(0, 11, 2, dtype=float)  # numpy的arange
print(arr2)

print('---------')

# 网格生成数据, 指定起始点和终止点, 以及网格数量
arr = np.linspace(0, 80, 7)
print(arr)
print('---------')

# 生成二维数组
a = np.arange(24).reshape((6, 4))
print(a)
# 将其展平为一个24的以为数组
print(a.flatten())
# 维度转换 将 6x4 二维数组 转换为 3x8 二维数组
a = np.arange(24).reshape((6, 4))
a.resize((3, 8))
print(a)

a = np.arange(24).reshape((6, 4))
# 矩阵的转置
print(a)
print(a.transpose())

print('---------')
a = np.arange(6).reshape((3, 2))
b = a * 2
print(a)
print(b)
# 水平组合
print(np.hstack((a, b)))
# 垂直组合
print(np.vstack((a, b)))
print('---------')

# 数组的标量计算
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
print(arr + 1)
print(arr ** 2)
print(1 / arr)
print('---------')

# 数组之间的运算

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
print(arr + arr)
print(arr * arr)
print('---------')
