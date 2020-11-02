import numpy as np
import scipy.special
import scipy.misc
import matplotlib.pyplot as plt
import glob
import imageio


class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes1,heddenNodes2, outputNodes, learningRate):
        # 一、设置每层节点个数和学习率
        self.inodes = inputNodes
        self.hnodes1 = hiddenNodes1
        self.hnodes2 = heddenNodes2
        self.onodes = outputNodes
        self.lr = learningRate

        # 二、初始化存储权重的矩阵
        # 1.随机初始化权重矩阵，np.random.rand（x,y）随机生成元素在-1到1的x*y的矩阵
        # self.ihWeightMatrix = np.random.rand(self.hnodes,self.inodes)-0.5 #减去0.5使范围缩小到-0.5到0.5
        # self.hoWeightMatrix = np.random.rand(self.onodes,self.hnodes)-0.5
        # numpy.random.normal(中心点的值，节点数目的-0.5次方 即与下一层节点相关的标准方差，数组形状)

        # 2.也可以以正态分布方式初始化矩阵
        self.ihWeightMatrix = np.random.normal\
            (0.0, pow(self.hnodes1, -0.5), (self.hnodes1, self.inodes))   #pow(a,b)表示a^b
                                                                          #要改成两行需要加\才可以
        self.iiWeightMatrix = np.random.normal\
            (0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes1))

        self.hoWeightMatrix = np.random.normal\
            (0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes2))

        # 三、设置激活函数sigmoid
        # lambda是匿名的 输入是传入到参数列表x的值，输出是根据表达式计算得到的值
        # scipy.special.expit(x)就是sigmoid函数，它等于1/(1+exp(-x))
        self.activateFunc = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputsList, targetsList):
        # 一、先根据实际输入生成输出
        inputs = np.array(inputsList, ndmin=2).T

        hiddenInputs1 = np.dot(self.ihWeightMatrix, inputs)
        hiddenOutputs1 = self.activateFunc(hiddenInputs1)

        hiddenInputs2 = np.dot(self.iiWeightMatrix, hiddenOutputs1)
        hiddenOutputs2 = self.activateFunc(hiddenInputs2)

        finalInputs = np.dot(self.hoWeightMatrix, hiddenOutputs2)
        finalOutputs = self.activateFunc(finalInputs)

        # 二、再将实际输出与理想值比较
        targets = np.array(targetsList, ndmin=2).T
        outputErrors = targets - finalOutputs
        # 反向传播误差值：上一层的errors=WeightsMatrix的转置×这层的errors
        hiddenErrors2 = np.dot(self.hoWeightMatrix.T, outputErrors)
        hiddenErrors1 = np.dot(self.iiWeightMatrix.T, hiddenErrors2)

        # 三、根据梯度下降公式求出用于更新j层到k层之间权重的矩阵式：（•是矩阵相乘；transpose作用是使矩阵转置）
        # △W（j,k）=α×E（k）×sigmoid(O(k))×(1-sigmoid(O(k)))•（O(j)的转置）
        # 更新self.hoWeightMatrix
        self.hoWeightMatrix += self.lr * np.dot((outputErrors * finalOutputs * (1 - finalOutputs)),
                                                np.transpose(hiddenOutputs2))

        # 更新self.iiWeightMatrix
        self.iiWeightMatrix += self.lr * np.dot((hiddenErrors2 * hiddenOutputs2 * (1 - hiddenOutputs2)),
                                                np.transpose(hiddenOutputs1))

        # 更新self.ihWeightMatrix
        self.ihWeightMatrix += self.lr * np.dot((hiddenErrors1 * hiddenOutputs1 * (1 - hiddenOutputs1)),
                                                np.transpose(inputs))
        pass

    def query(self, inputsList):
        # 一、将输入的向量转置便于计算X=W(matrix)*I
        # ndmin=定义数组的最小维度 或 数组嵌套层数;“.T”表示将矩阵转置
        inputs = np.array(inputsList, ndmin=2).T

        # 二、计算隐藏层1的输入X=W(matrix)*I
        # dot可用于求数乘积、向量内积、矩阵乘法
        hiddenInputs1 = np.dot(self.ihWeightMatrix, inputs)

        # 三、隐藏层1输入通过sigmoid函数映射成输出
        hiddenOutputs1 = self.activateFunc(hiddenInputs1)

        # 四、计算隐藏层2的输入X=W(matrix)*I
        # dot可用于求数乘积、向量内积、矩阵乘法
        hiddenInputs2 = np.dot(self.iiWeightMatrix, hiddenOutputs1)

        # 五、隐藏层2输入通过sigmoid函数映射成输出
        hiddenOutputs2 = self.activateFunc(hiddenInputs2)


        # 六、计算输出层的输入
        finalInputs = np.dot(self.hoWeightMatrix, hiddenOutputs2)

        # 七、输出层输入通过sigmoid函数映射成输出
        finalOutputs = self.activateFunc(finalInputs)
        return finalOutputs

#误差分析:均方误差
def MSE(predict,fact,n):
    return np.sum((predict-fact)**2)/n

#初始化一个神经网络
input_nodes = 784
hidden_nodes1 = 30
hidden_nodes2 = 60
output_nodes = 10
learning_rate = 0.05
n = neuralNetwork(input_nodes, hidden_nodes1,hidden_nodes2,output_nodes, learning_rate)

#将训练集存入一个列表
#'r'指以只读的方式打开，mnistTrain是一个文件句柄
#读入文件中的所有行保存在trainList的列表中，列表每一项对应文件的一行字符串，可通过trainList[i]调取
mnistTrain=open("mnist_train.csv",'r')
trainList=mnistTrain.readlines()
mnistTrain.close()             #关闭文件


#开始实际训练
epochs=5  #整个训练集遍历epochs次
for i in range(epochs):
    for record in trainList:
        # 根据‘，’进行拆分
        allValues=record.split(',')
        # csv中要输入的像素点值在0~255太大，将其缩小映射到0.01~1.0方便训练（可取1但不能取0，是因为输入0值可能更新权重会失败）
        reducedInputs = (np.asfarray(allValues[1:])) / 255 * 0.99 + 0.01
        #初始化目标矩阵，用0.01和0.99而不用0和1，因为sigmoid实际输出值不可能是0或1
        targets=np.zeros(output_nodes)+0.01
        targets[int(allValues[0])]=0.99
        n.train(reducedInputs,targets)
        pass
    pass

#进行测试
mnistTest=open("mnist_test.csv",'r')
testList=mnistTest.readlines()
mnistTest.close()

score=[]    #用计分来记录准确度
sum=0.0
for record in testList:
    allValues=record.split(',')
    #期望结果
    expectedResult=int(allValues[0])
    print(expectedResult,"正确的结果")
    reducedInputs = (np.asfarray(allValues[1:])) / 255 * 0.99 + 0.01
    outputs=n.query(reducedInputs)
    # 实际输出结果
    factResult=np.argmax(outputs)   #.argmax输出outputs中的最大值
    print(factResult,"网络输出的结果\n")

    #累加计算误差平方和
    sum+=((factResult-expectedResult)**2)

    if(factResult==expectedResult):
        score.append(1)
    else:
        score.append(0)
        pass
    pass
scoreArr=np.asarray(score)
#均方差
mse=sum/len(testList)
print("accuracy:",scoreArr.sum()/scoreArr.size*100,'%')
print("均方误差MSE:",mse)



#可视化例子：
tmp=0
while tmp!=-1:
    tmp=int(input("请输入测试集中记录的编号：（输入-1退出）"))
    if not(tmp>=-1 and tmp<=10000 and isinstance(tmp,int)):
        print("格式错误，请重新输入!")
        continue
    allValues=testList[tmp].split(',')
    print(allValues[0])
    #asfarray将文本字符串转成实数并创建数组，.reshape确保数字列表每28个元素折返一次，得到28*28矩阵
    imgArr=np.asfarray(allValues[1:]).reshape((28,28))
    #imshow打印成图像，cmap是颜色映射，interpolation是抗锯齿的度
    plt.imshow(imgArr,cmap='Blues',interpolation='None')
    plt.show()
    print(n.query((np.asfarray(allValues[1:])/ 255 * 0.99) + 0.01))



print("\n下面针对自己手写的数字进行识别")
# our own image test data set
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('my_own_?.png'):
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])

    # load image data from png files into an array
    print("loading ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)

    # reshape from 28x28 to list of 784 values, invert values
    img_data = 255.0 - img_array.reshape(784)

    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01

    # append label and image data  to test data set
    record = np.append(label, img_data)
    our_own_dataset.append(record)

    pass


# 用自己手写的数字


while True:
    i = int(input("\n输入我自己写的数字进行识别:（输入-1退出）"))
    if i==-1:
        break
    item = i

    # 把这个数字打印出来
    plt.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')
    plt.show()

    #第一个元素是正确结果
    correct_label = our_own_dataset[item][0]
    # 其他的元素是输入
    inputs = our_own_dataset[item][1:]

    # 查询网络
    outputs = n.query(inputs)
    print (outputs)

    # 最大的元素的下标为预测值
    label = np.argmax(outputs)
    print("network says ", label)
    # 比较输出结果
    if (label == correct_label):
        print ("success")
    else:
        print ("fail")
        pass