import numpy as np

arr = np.arange(10)
print(arr)
print(arr[5])
print(arr[5:8])

# 修改值
arr[5:8] = 999
print(arr)

arr = np.arange(10)
temp = arr[5:8]
temp[1] = 999  # 由于没有对数据进行新的拷贝, 对视图的修改会直接反应到原始数组上
print(arr)

# 显式复制
arr = np.arange(10)
temp = arr[5:8].copy()
temp[1] = 888
print(arr)

# 高维数组的索引
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[2])
print(arr2d[0, 1])
print(arr2d[0][1])

#
print('---------------')
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d)
print('---------------')
print(arr3d[0])
print('---------------')
print(arr3d[1][0])
print('---------------')
print(arr3d[0][0][1])

arr3d[0] = 111
print(arr3d)
print('---------------')

# 切片
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d[:2, 2:])
