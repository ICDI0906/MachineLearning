# @Time : 2018/9/29 下午9:03 
# @Author : Kaishun Zhang 
# @File : numpy_mit.py 
# @Function:学习斯坦福大学的numpy的numpy教程 网址：http://cs231n.github.io/python-numpy-tutorial/

# 对列表进行快速排序
# params 待排序的数组
# return 排好序的数组
def quickSort(arr):
    if len(arr)<=1:
        return arr
    middlle_value = arr[len(arr)//2]
    left = [val for val in arr if val < middlle_value]
    middle = [val for val in arr if val == middlle_value]
    right = [val for val in arr if val > middlle_value]
    return quickSort(left) + middle + quickSort(right)

# 对list进行pop 操作
# params 列表
def listPop(arr):
    while arr:
        x = arr.pop(2); # 可以pop 最后一个元素，也可以将指定索引的元素pop 出来
        print(x)

# numpy 常见的操作
def numpyOperate():
    import numpy as np
    a = np.zeros((2,2))
    # print(a)
    a = np.full((2,2),7)
    # print(a)
    a = np.eye(3)     # here just one parameter
    # print(a)
    a = np.random.random((2,2)) # notice here is np.random.random() not np.random()
    # print(a)
    # print(a[a > 0.5])  # select element with constraint

    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])

    v = np.array([9, 10])
    w = np.array([11, 12])

    print(v.dot(w))
    print('v.shape: ',v.shape,'x.shape: ',x.shape)
    print(x.dot(v))  # here is running operate in order like x * v

    print(np.sum(x, axis = 0))  # the sum of y-axis
    print(np.sum(x, axis = 1))  # the sum of x-axis

    # add the same value for each row
    add_value = [2,2]
    y = np.empty_like(x)
    for i in range(2):
        y[i,] = x[i,] + add_value
    print(x)
    # above is not good enough
    vv = np.tile(add_value, (x.shape[0],1)) # copy multiple times and if value equals 0 then []
    print(x + vv)

    # we donot need to do like that
    # just add them
    print(x + add_value)

    print(np.reshape(x,(x.shape[0] * x.shape[1])))  # reshape


# scipy.misc 图像操作
def imageOperate():
    from scipy.misc import imread,imsave,imresize
    img = imread('cat.jpg')
    print(img.dtype,img.shape)
    img_tinted = img * [1, 0.95, 0.9]
    img_tinted = imresize(img_tinted, (300,300))
    imsave('hh.jpg',img_tinted)  # dot convert


# distance between points
def disBetweenPoints():
    from scipy.spatial.distance import pdist,squareform
    import numpy as np
    x = np.array([[0, 1], [1, 0], [2, 0]])
    print(squareform(pdist(x)))

# use matlibplot
def useMatlibplot():
    from scipy.misc import imread,imsave,imresize
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)
    plt.subplot(2,2,1)  # 子图显示
    plt.plot(x, y)
    plt.subplot(2,2,2)
    plt.plot(x,y)

    img = imread('cat.jpg') # 读取图片然后进行展示
    plt.subplot(2,2,3)
    plt.imshow(img)

    plt.show()


if __name__ == '__main__':
    # listPop([1,2,3,4])
    # numpyOperate()
    # imageOperate()
    # disBetweenPoints()
    useMatlibplot()