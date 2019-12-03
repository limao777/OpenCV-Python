# 第八章：机器学习

本章节你将学习K-最邻近、支持向量机和K-Means聚类等OpenCV机器学习的相关内容。

更多内容请关注我的[GitHub库:TonyStark1997](https://github.com/TonyStark1997)，如果喜欢，star并follow我！

该章节从第二部分开始由limao777进行翻译

***

## 一、K-最邻近

***

### 目标：

本章节你需要学习以下内容:

    *在本章中，我们将理解k-最近邻（kNN）算法的概念。
    *我们将使用我们在kNN上的知识来构建基本的OCR应用程序。
    *我们将尝试使用OpenCV附带的数字和字母数据。

### 1、了解k-最近邻

#### （1）理论

kNN是最简单的用于监督学习的分类算法之一。想法也很简单，就是找出测试数据在特征空间中的最近邻居。我们将用下图来介绍它。

![image1](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/8.Machine%20Learning/Image/image1.jpg)

上图中的对象可以分成两组，蓝色方块和红色三角。每一组也可以称为一个 类。我们可以把所有的这些对象看成是一个城镇中房子，而所有的房子分别属于蓝色和红色家族，而这个城镇就是所谓的特征空间。（你可以把一个特征空间看成是所有点的投影所在的空间。例如在一个 2D 的坐标空间中，每个数据都两个特征 x 坐标和 y 坐标，你可以在 2D 坐标空间中表示这些数据。如果每个数据都有 3 个特征呢，我们就需要一个 3D 空间。N 个特征就需要 N 维空间，这个 N 维空间就是特征空间。在上图中，我们可以认为是具有两个特征色2D 空间）。

现在城镇中来了一个新人，他的新房子用绿色圆盘表示。我们要根据他房子的位置把他归为蓝色家族或红色家族。我们把这过程成为 分类。我们应该怎么做呢？因为我们正在学习看 kNN，那我们就使用一下这个算法吧。

一个方法就是查看他最近的邻居属于那个家族，从图像中我们知道最近的是红色三角家族。所以他被分到红色家族。这种方法被称为简单 近邻，因为分类仅仅决定与它最近的邻居。

但是这里还有一个问题。红色三角可能是最近的，但如果他周围还有很多蓝色方块怎么办呢？此时蓝色方块对局部的影响应该大于红色三角。所以仅仅检测最近的一个邻居是不足的。所以我们检测 k 个最近邻居。谁在这 k 个邻居中占据多数，那新的成员就属于谁那一类。如果 k 等于 3，也就是在上面图像中检测 3 个最近的邻居。他有两个红的和一个蓝的邻居，所以他还是属于红色家族。但是如果 k 等于 7 呢？他有 5 个蓝色和 2 个红色邻居，现在他就会被分到蓝色家族了。k 的取值对结果影响非常大。更有趣的是，如果 k 等于 4呢？两个红两个蓝。这是一个死结。所以 k 的取值最好为奇数。这中根据 k 个最近邻居进行分类的方法被称为 kNN。

在 kNN 中我们考虑了 k 个最近邻居，但是我们给了这些邻居相等的权重，这样做公平吗？以 k 等于 4 为例，我们说她是一个死结。但是两个红色三角比两个蓝色方块距离新成员更近一些。所以他更应该被分为红色家族。那用数学应该如何表示呢？我们要根据每个房子与新房子的距离对每个房子赋予不同的权重。距离近的具有更高的权重，距离远的权重更低。然后我们根据两个家族的权重和来判断新房子的归属，谁的权重大就属于谁。这被称为 修改过的kNN。

那么你在这里看到的重要事情是什么？

* 我们需要整个城镇中每个房子的信息。因为我们要测量新来者到所有现存房子的距离，并在其中找到最近的。如果那里有很多房子，就要占用很大的内存和更多的计算时间。

* 训练和处理几乎不需要时间。

现在让我们在OpenCV中看到它。

#### （2）OpenCV中的K-最邻近

我们这里来举一个简单的例子，和上面一样有两个类。下一节我们会有一个更好的例子。

这里我们将红色家族标记为 Class-0，蓝色家族标记为 Class-1。还要再创建 25 个训练数据，把它们非别标记为 Class-0 或者 Class-1。Numpy中随机数产生器可以帮助我们完成这个任务。

然后借助 Matplotlib 将这些点绘制出来。红色家族显示为红色三角蓝色家族显示为蓝色方块。

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')

# Take Blue families and plot them
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

plt.show()
```

你可能会得到一个与上面类似的图形，但不会完全一样，因为你使用了随机数产生器，每次你运行代码都会得到不同的结果。

下面就是 kNN 算法分类器的初始化，我们要传入一个训练数据集，以及与训练数据对应的分类来训练 kNN 分类器（构建搜索树）。

最后要使用 OpenCV 中的 kNN 分类器，我们给它一个测试数据，让它来进行分类。在使用 kNN 之前，我们应该对测试数据有所了解。我们的数据应该是大小为数据数目乘以特征数目的浮点性数组。然后我们就可以通过计算找到测试数据最近的邻居了。我们可以设置返回的最近邻居的数目。返回值包括：

1. 由 kNN 算法计算得到的测试数据的类别标志（0 或 1）。如果你想使用最近邻算法，只需要将 k 设置为 1，k 就是最近邻的数目。
 
2. k 个最近邻居的类别标志。

3. 每个最近邻居到测试数据的距离。

让我们看看它是如何工作的。测试数据被标记为绿色。

```python
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)

print( "result: {}\n".format(results) )
print( "neighbours: {}\n".format(neighbours) )
print( "distance: {}\n".format(dist) )

plt.show()
```

我得到的结果如下：

```python
result: [[ 1.]]
neighbours: [[ 1. 1. 1.]]
distance: [[ 53. 58. 61.]]
```

这说明我们的测试数据有 3 个邻居，他们都是蓝色，所以它被分为蓝色家族。结果很明显，如下图所示：

![image2](https://raw.githubusercontent.com/TonyStark1997/OpenCV-Python/master/8.Machine%20Learning/Image/image2.jpg)

如果我们有大量的数据要进行测试，可以直接传入一个数组。对应的结果同样也是数组。

```python
# 10 new comers
newcomers = np.random.randint(0,100,(10,2)).astype(np.float32)
ret, results,neighbours,dist = knn.findNearest(newcomer, 3)
# The results also will contain 10 labels.
```

### 2、使用kNN对手写数据进行OCR

#### （1）使用 kNN 对手写数字 OCR

我们的目的是创建一个可以对手写数字进行识别的程序。为了达到这个目的我们需要训练数据和测试数据。OpenCV附带一个images digits.png（在文件夹opencv/samples/data/中），其中有 5000 个手写数字（每个数字重复 500遍）。每个数字是一个 20x20 的小图。所以第一步就是将这个图像分割成 5000个不同的数字。我们在将拆分后的每一个数字的图像重排成一行含有 400 个像素点的新图像。这个就是我们的特征集，所有像素的灰度值。这是我们能创建的最简单的特征集。我们使用每个数字的前 250 个样本做训练数据，剩余的250 个做测试数据。让我们先准备一下：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('digits.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print( accuracy )
```

现在最基本的 OCR 程序已经准备好了，这个示例中我们得到的准确率为91%。改善准确度的一个办法是提供更多的训练数据，尤其是判断错误的那些数字。为了避免每次运行程序都要准备和训练分类器，我们最好把它保留，这样在下次运行是时，只需要从文件中读取这些数据开始进行分类就可以了。  
Numpy 函数 np.savetxt，np.load 等可以帮助我们搞定这些。

```python
# save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)

# Now load the data
with np.load('knn_data.npz') as data:
    print( data.files )
    train = data['train']
    train_labels = data['train_labels']
```

在我的系统中，占用的空间大概为 4.4M。由于我们现在使用灰度值（unint8）作为特征，在保存之前最好先把这些数据装换成 np.uint8 格式，这样就只需要占用 1.1M 的空间。在加载数据时再转会到 float32。

#### （2）英文字母的 OCR

接下来我们来做英文字母的 OCR。和上面做法一样，但是数据和特征集有一些不同。现在 OpenCV 给出的不是图片了，而是一个数据文件（/samples/cpp/letter-recognition.data）。如果打开它的话，你会发现它有 20000 行，第一样看上去就像是垃圾。实际上每一行的第一列是我们的一个字母标记。接下来的 16 个数字是它的不同特征。这些特征来源于UCI Machine LearningRepository。你可以在此页找到更多相关信息。  

有 20000 个样本可以使用，我们取前 10000 个作为训练样本，剩下的10000 个作为测试样本。我们应在先把字母表转换成 asc 码，因为我们不正直接处理字母。

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the data, converters convert the letter to a number
data= np.loadtxt('letter-recognition.data', dtype= 'float32', delimiter = ',',
converters= {0: lambda ch: ord(ch)-ord('A')})

# split the data to two, 10000 each for train and test
train, test = np.vsplit(data,2)

# split trainData and testData to features and responses
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])

# Initiate the kNN, classify, measure accuracy.
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)

correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
print( accuracy )
```

准确率达到了 93.22%。同样你可以通过增加训练样本的数量来提高准确率。

## 二、支持向量机

***

### 目标：

本章节你需要学习以下内容:

    *我们将看到对SVM的直观理解
    *我们将重新访问手写数据OCR，但是，使用SVM而不是kNN。

### 1、了解SVM

### 目标

这节我们将直观地了解SVM

### 理论

#### 线性分离数据

假设下图中有两种类型数据：红和蓝。在KNN里面，我们曾经测量它到所有训练样本的距离，然后以最小的距离进行提取作为测试数据。它将花费大量时间和内存测量所有训练样本的距离。但是下图中的图片我们需要这么多资源吗？

![svm_basics1](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/svm_basics1.png)

想象另一种方案，我们找到一条线，f(x) = ax1 + bx2 + c，这条线把两种数据分成了两个区域。当有一个新的测试数据X时，把它代入f(x)，如果f(X)>0,那么它属于蓝色类型，否则则是红色类型。这种我们可以称之为 **决策边界** 。它非常简单并且节约内存。这种能被一条直线（或更高维度的超平面）分成两部分的数据称为 **线性分割** 。

在上图中，我们能发现划线分割数据可以划很多条，那么我们该取哪条呢？非常直观地，我们可以说该线应尽可能远离所有点，因为新的数据可能会制造噪点，这个数据不应该影响分类精度，所以选一个离它们都最远的线可以尽可能消除噪点。因此SVM会找一个离训练样本最远的直线（或超平面），这条线见下图的粗线。

![svm_basics2](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/svm_basics2.png)

要找到决策边界，你需要训练数据，但是你不需要训练所有数据，只需要那些接近对方的数据就足够了。在图中，他们是蓝色圈圈和红色方形。我们称之为 **支持向量** ，而这套线我们称为 **支持平面** 。他们足以让我们找到决策边界。我们不需要担心传递所有数据，它将帮助我们自动减少数据。

它怎么做的呢，首先找到最能代表数据的两个超平面。比如蓝色数据用wTx + b0 > 1代表，而红色用wTx + b0 < -1代表，w表示 **权重向量** （w=[w1, w2, ..., wn]），x是特征向量（x=[x1,x2,...,xn），b0是 **偏置量** ，这些超平面用wTx + b0 = 0表示。从支持向量到决策边界的最小值用distance support vectors=1||w||表示，裕度是此距离的两倍，因此我们需要最大化此裕度。 即我们需要最小化新函数L（w, b0），约束如下：

![formula1](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/formula1.png)

ti是每个分类的标签，ti∈[-1, 1]。

#### 非线性分离数据
想象有些数据不能被一条直线分割，比如某一维数据，'x'在-3 & +3，'O'在-1 & +1。明显地他们不能线性地分割。但是我们有办法解决这类问题，如果我们能把这些数据放进f(x) = x^2中，我们得到'X'为9而'O'为1，这样就能线性地分割了。

要不然的话我们也可以把它从一维转成二维数据。我们能用f(x) = (x,x^2)来放数据。'X'变成了（-3, 9）以及（3, 9）,'O'则是（-1, 1）以及（1, 1），这样依然是线性可分割的。简而言之，低维空间中的非线性可分离数据更有可能在高维空间中变为线性可分离。

通常，我们可以将d维空间中的点映射到某个D维空间（D > d）来看是否能线性分割，这样就产生了一个想法可以通过在低维输入（特征）空间中执行计算来帮助在高维（内核）空间中计算点积。 我们可以用下面的例子来说明。

假设在二维空间中有两点，p = (p1, p2)以及q = (q1, q2)，令ϕ是一个映射函数，该函数将二维点映射到三维空间，如下所示：

![formula2](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/formula2.png)

定义一个两点点积的核心方法K(p, q)，如下

![formula3](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/formula3.png)

即使用二维空间中的平方点积可以实现三维空间中的点积，这在更高维度也适用，因此我们可以从较低维度本身计算较高维度的特征，因此一旦将他们输入，我们就能得到高位空间数据。

除了所有这些概念之外，如果分类错误呢，仅找到具有最大余量的决策边界是不够的，所以我们还应考虑到错误分类的问题。有时可能会找到裕度较小的决策边界，但减少了误分类。 无论如何我们应修改我们的模型，以便它应该找到具有最大裕度但分类错误较少的决策边界。 最小化标准修改为：

min||w||2+C(distanceofmisclassifiedsamplestotheircorrectregions)

下图展示了此概念。对于每个样本训练数据，我们定义一个新的参数ξi。它是从其相应的训练样本到其正确决策区域的距离。 对于未分类错误的数据，它们落在相应的支撑平面上，因此它们的距离为零。

![svm_basics3](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/svm_basics3.png)

新的优化问题是

minw,b0L(w,b0)=||w||2+C∑iξi subject to yi(wTxi+b0)≥1−ξi and ξi≥0 ∀i

如何选择参数C？ 显然，这个问题的答案取决于训练数据的分布方式。 尽管没有常规答案，但以下规则会很有用：
* C的值越大，解决方案的分类错误越少，但裕度也越小。 考虑到在这种情况下，进行错误分类错误是昂贵的。 由于优化的目的是使参数最小化，因此几乎没有错误分类错误。
* C的值越小，解决方案的裕度就越大，分类误差也越大。 在这种情况下，最小化对总和项的考虑不多，因此它更多地关注于找到具有大余量的超平面。

#### 更多资源
NPTEL notes on Statistical Pattern Recognition, Chapters 25-29.（http://www.nptel.ac.in/courses/106108057/26）

### 2、使用SVM对手写数据进行OCR

#### 目标
使用SVM而不是kNN来重新做手写识别的OCR

#### 手写数字的OCR

在kNN中，我们直接使用像素强度作为特征向量。 这次我们将使用定向梯度直方图（HOG）作为特征向量。

在找到HOG之前，我们使用其二阶矩阵对图像进行校正。首先定义一个函数deskew（），该函数获取一个数字图像并将其校正。 下面是deskew（）函数：

```python
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
```

下图显示了应用于数字零图像的上偏移校正功能。 左图像是原始图像，右图像是校正后图像。

![deskew.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/deskew.jpg)

接下来，我们须找到每个单元格的HOG描述符。为此我们找到了每个单元在X和Y方向上的Sobel导数。然后在每个像素处找到它们的大小和梯度方向。该梯度被量化为16个整数值。 将此图像划分为四个子方块。对于每个子方块，在直方图中计算权重方向(16bins)大小。每个字方块提供了一个包含16个值的向量。对这些向量(四个子方块的)一起给我们提供了一个包含64个值得特征向量。这是我们用于训练数据的特征向量。

```python
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
```

最后，与前面的情况一样，我们首先将大数据集拆分为单个单元格。 对于每个数字，保留250个单元用于训练数据，其余250个数据保留用于测试。 完整的代码如下：

```python
#!/usr/bin/env python
import cv2 as cv
import numpy as np
SZ=20
bin_n = 16 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
img = cv.imread('digits.png',0)
if img is None:
    raise Exception("we need the digits.png image from samples/data here !")
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]
deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')
deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1]
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)
```

这种特殊的技术给了我近94％的准确性。 您可以为SVM的各种参数尝试不同的值，以检查是否可以实现更高的精度。 或者，您可以阅读有关此领域的技术论文并尝试实施它们。

#### 更多资源
Histograms of Oriented Gradients Video（Histograms of Oriented Gradients Video）

#### 练习
OpenCV样本包含digits.py，它对上述方法进行了一些改进以得到改进的结果。它还包含注释，去查看并了解它吧。

## 三、K-Means聚类

***

### 目标：

本章节你需要学习以下内容:

    *在本章中，我们将了解K-Means聚类的概念，它是如何工作的等等。
    *学习在OpenCV中使用cv.kmeans（）函数进行数据聚类

### 1、了解K-Means聚类

#### 目标
我们将在这章中学习理解K-Means聚类，以及它如何工作等

#### 理论
我们将用一个常用的例子来解释

#### T恤尺寸问题
假设有个公司，将打算投放一批新的T恤到市场上去。显然他们需要制作各种尺寸来适应各种人的尺寸。所以这家公司把人们的身高体重统计起来画出了如下的图表：

![tshirt.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/tshirt.jpg)

这个公司不可能把所有尺寸都生产一遍，相应地，他们将尺寸分为小号、中号以及大号使这三种型号能适应所有人。这种分类方式可以用k-means方法来解决，算法将给我们提供3个最好的尺寸来适应所有人。如果没有的话，公司可以将尺寸分更多的组，可能有五个，可能更多，如下图：

![tshirt_grouped.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/tshirt_grouped.jpg)

#### 它怎么工作
这个算法是一个迭代过程。我们将在图像的帮助下逐步解释它。

考虑如下数据（你可以将其视为T恤问题）。我们需要将此数据分为两类。

![testdata.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/testdata.jpg)

** Step1： ** 算法随机选择两个质心C1和C2（有时，直接将任何两个数据作为质心）。

** Step2： ** 它计算每个点到两个质心的距离。如果测试数据更接近C1，则该数据标记为“0”。如果它更靠近C2，则标记为“1”（如果存在更多质心，则标记为“2”，“3”等）。

在我们的示例中，我们将为所有标记为红色的“0”和标记为蓝色的所有“1”上色。 因此，经过以上操作，我们得到以下图像。

![initial_labelling.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/initial_labelling.jpg)

** Step3： ** 接下来，我们分别计算所有蓝点和红点的平均值，这将成为我们的新质心。即C1和C2转移到新计算的质心。（显示的图像不是真实值，也不是真实比例，仅用于演示）。

接着再执行Step2，并设置“0”和“1”，我们得到如下结果：

![update_centroid.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/update_centroid.jpg)

现在迭代步骤2和步骤3，直到两个质心都收敛到固定点。*（或者可以根据我们提供的标准来停止，例如最大迭代次数，或者达到特定的精度等。）*这些点使得测试数据与其对应质心之间的距离之和最小。或者简单地说，C1↔Red_Points和C2↔Blue_Points之间的距离之和最小。

![formula4.png](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/formula4.png)

最终结果大概看起来如下：

![final_clusters.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/final_clusters.jpg)

因此，这仅仅是对K-Means聚类的直观理解。 有关更多详细信息和数学解释，请阅读其他标准的机器学习教科书或查看其他资源中的链接，它只是K-Means群集的顶层。真实的算法有很多修改，例如如何选择初始质心，如何加快迭代过程等。

#### 更多资源
Machine Learning Course, Video lectures by Prof. Andrew Ng (Some of the images are taken from this)（https://www.coursera.org/course/ml）

### 2、在OpenCV中的K-Means聚类

#### 目标
学会在OpenCV中使用cv.kmeans()函数做数据聚类

#### 理解参数
#### 输入参数
1. **samples** ：np.float32类型，每个特征单独一列(column)

2. **nclusters(K)** ：最后需要的集群数

3.  **criteria** ：这是迭代终止条件。 满足此条件后，算法迭代将停止。实际上它是3个参数组成的数组。 它们是`（type，max_iter，epsilon）`：

type参数有三种标志可以传：
* cv.TERM_CRITERIA_EPS - 如果达到指定的精度epsilon，则停止算法迭代。
* cv.TERM_CRITERIA_MAX_ITER - 在指定的迭代次数max_iter之后停止算法。
* cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER - 满足以上任何条件时，停止迭代。

max_iter - 整型，最大迭代次数。

epsilon - 要求的精度

4. **attempts** ：用于指定使用不同的初始标签执行算法的次数的标志。该算法返回产生最佳紧密度的标签。该紧凑性作为输出返回。

5. **flags** ：此标志用于指定如何获取初始中心。 通常使用两个标志：cv.KMEANS_PP_CENTERS和cv.KMEANS_RANDOM_CENTERS。

#### 输出参数
1. **compactness** ：是每个点到其对应中心的平方距离的总和。

2. **labels** ：这是标签数组（与上一篇文章中的“code”相同），其中每个元素标记为“ 0”，“ 1” .....

3. **centers** ：这是群集中心的数组。

#### 1.仅有一项特征的数据
假设我们数据仅有一项特征，即一维。比如在T恤问题中我们只有人们的身高，那么我们怎么来决定我们制作的尺寸呢。

我们先把数据在matplotlib中画出来

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
x = np.random.randint(25,100,25)
y = np.random.randint(175,255,25)
z = np.hstack((x,y))
z = z.reshape((50,1))
z = np.float32(z)
plt.hist(z,256,[0,256]),plt.show()
```

z是50个内容的数据，数值从0-255.我们把z重塑为列向量，这样将在有多个特征的时候更加有用，接下来用np.float32类型画出了如下图：

![oc_1d_testdata.png](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/oc_1d_testdata.png)

现在我们应用KMeans函数。 在此之前，我们需要指定标准。 我的标准是，每当运行10次算法迭代或达到epsilon = 1.0的精度时，就停止算法并返回答案。

```python
# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv.KMEANS_RANDOM_CENTERS
# Apply KMeans
compactness,labels,centers = cv.kmeans(z,2,None,criteria,10,flags)
```

这为我们提供了紧凑性，标签和质心。在这种情况下，我得到的中心分别为60和207。标签的大小将与测试数据的大小相同，其中每个数据的质心都将标记为“0”，“1”，“2”等。现在，我们根据标签将数据分为不同的群集。

```python
A = z[labels==0]
B = z[labels==1]
```

现在我们以红色绘制A，以蓝色绘制B，以黄色绘制其质心

```python
# Now plot 'A' in red, 'B' in blue, 'centers' in yellow
plt.hist(A,256,[0,256],color = 'r')
plt.hist(B,256,[0,256],color = 'b')
plt.hist(centers,32,[0,256],color = 'y')
plt.show()
```

下图则是绘制后的图：

![oc_1d_clustered.png](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/oc_1d_clustered.png)

#### 2.有多项特征的数据
在前一个例子中，我们仅仅用身高来解决T恤问题，现在我们用身高和体重，即两个特征。

记住，在前一个例子中，我们把数据做成了单列向量，每个特征位于每一列，则每一行则代表输入的测试样本。

比如在这个例子中，我们把数据设置成50*2格式，表示有50人的身高体重。第一列表示50人的身高，第二列表示他们的体重。第一行包含两个元素代表的是第一个人的身高体重，类似的其他行表示其他人的身高体重，如下表：

![oc_feature_representation.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/oc_feature_representation.jpg)

我们直接看代码：
```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
Z = np.vstack((X,Y))
# convert to np.float32
Z = np.float32(Z)
# define criteria and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv.kmeans(Z,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]
# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
```

输出图像如下：

![oc_2d_clustered.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/oc_2d_clustered.jpg)

#### 3.颜色量化
颜色量化是减少图像中颜色数量的过程。这样做的原因之一是减少内存。有时某些设备可能会受到限制只能生成有限数量的颜色。同样在那些情况下，也要执行颜色量化。在这里，我们使用k均值聚类进行颜色量化。

这里没有新内容要解释。有3个特征，R，G，B。 因此，我们需要将图像重塑为Mx3大小的数组（M是图像中的像素数）。在聚类之后，我们将质心值（也是R，G，B）应用于所有像素，这样生成的图像将具有指定的颜色数。再一次的我们需要将其还原为原始图像的形状。下面是代码：
```python
import numpy as np
import cv2 as cv
img = cv.imread('home.jpg')
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv.imshow('res2',res2)
cv.waitKey(0)
cv.destroyAllWindows()
```

K=8的结果如下：

![oc_color_quantization.jpg](https://github.com/limao777/OpenCV-Python/blob/master/8.Machine%20Learning/Image/oc_color_quantization.jpg)