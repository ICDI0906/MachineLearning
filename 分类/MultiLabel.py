from  sklearn.preprocessing import MultiLabelBinarizer

y = [[2,3,4],[2],[0,1,3],[0,1,2,3,4],[0,1,2]]

MultiLabelBinarizer().fit_transform(y)  #转换为矩阵的形式
