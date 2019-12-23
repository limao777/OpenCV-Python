# OpenCV-Python
The Opencv-Python tutorial Chinese translation
***

本章由limao777进行翻译，水平有限不足之处请见谅~

# Object Detection（物体检测）
本章你将开启物体检测比如面部检测等技术

## 使用haar-cascades技术的面部检测
使用haar-cascades技术的面部检测

### 目标
* 你将看到使用基于Haar特征的级联分类器的基本的面部检测
* 你将把这项技术扩展到眼睛检测上

### 基础
使用基于Haar特征的级联分类器的物体检测是一个非常有效的物体检测技术，它被Paul Viola和Michael Jones的论文《Rapid Object Detection using a Boosted Cascade of Simple Features》在2001年提出。这是一种基于机器学习的方法，其中从许多正负图像中训练级联函数。 然后用于检测其他图像中的对象。

在这里我们将运行面部检测，这个算法需要大量正关联图片（图像中有面部）和负相关图像（图像中没有面部）来分类。接下来我们需要从中提取特征。对于此，我们将使用下图所示的Haar功能。这些就像我们的卷积核，每个特征都是通过从黑色矩形下的像素总和中减去白色矩形下的像素总和而获得的单个值。

![haar_features.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/haar_features.jpg)

现在，这个卷积核将被使用到所有可能的尺寸和位置来计算大量特征（可以想象有多少计算量？即使24*24的窗体就会有多达160000特征）。对于每个特征计算，我们需要找到在白色和黑色方块之下的所有像素。为了解决这个问题，他们引入了整体形象：即不管你的图像有多大，它都会将给定像素的计算减少到仅涉及四个像素的操作。非常棒是不是？这样使运算非常快。

但是在我们计算的所有这些功能中，大多数都不相关。比如下图：最上面一行展示了两个正向特征。选择的第一个特征似乎着眼于眼睛区域通常比鼻子和脸颊区域更暗的性质。选择的第二个功能依赖于眼睛比鼻梁更黑的属性。但是，将相同的窗口应用于脸颊或其他任何地方都是无关紧要的。 那么，我们如何从160000+个功能中选择最佳功能？ 它是由Adaboost实现的。

![haar.png](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/haar.png)

对于此，我们把每个特征都拿去训练，对于每个特征，它将寻找最优的阈值来区分正向关联和负向关联。显然地，这里将有错误或者误分类。我们选择最低错误的特征，这意味着他们的特征是最精确的有脸和无脸的分类图像。（处理过程没有这么简单。每张图片最开始都给予相同的权重。每次发呢类过后，误分类图像的权重将会增加。接着同样的过程会执行。新的错误比例将被计算，这也包括新的权重。这样的处理过程将被持续知道要求的精度或者错误比例达到要求亦或是规定的特征数量被找到）。

最终分类器是这些弱分类器的加权和。最终分类器是这些弱分类器的加权和。 之所以称为弱分类，是因为仅凭它不能对图像进行分类，而是与其他分类一起形成强分类器。该论文说，甚至200个特征都可以提供95％的准确度检测。他们的最终设置具有大约6000个特征。（想象一下，从160000多个特征减少到6000个特征。这是很大的收获）。

现在您拍摄一张照片。取每个24x24窗口。向其应用6000个特征来检查是否有脸。哇..这不是效率低下又费时吗？是的。作者对此有一个很好的解决方案。

对于一个图像，更多的是没有面部的区域。所以有个简单的好办法来检测一个窗口是否有图像，如果不是，则一次性丢弃它并且不再参与计算。然后把目光聚集到有脸的区域。这样的话我们花费更多的时间就是在检测是否有面部区域上面了。

对于此他们引入了 ** 级联分类器 ** 的概念。不是将所有6000个特征部件应用到一个窗口中，而是将这些特征部件分组到不同阶段的分类器中，并一一应用。（通常前几个阶段将包含很少的特征）。如果窗口在第一阶段失败，则将其丢弃。我们不考虑它的其余功能。如果通过，则应用功能的第二阶段并继续该过程。经过所有阶段的窗口是一个面部区域。这个计划怎么样？

作者的检测器具有6000多个特征，具有38个阶段，在前五个阶段具有1、10、25、25和50个特征。（上图中的两个功能实际上是从Adaboost获得的最佳两个功能）。根据作者的说法，每个子窗口平均评估了6000多个特征中的10个特征。

这是Viola-Jones人脸检测工作原理的简单直观说明。阅读其论文获取更多详细信息，或查看其他资源部分中的参考资料。

### OpenCV中的Haar-cascade检测
OpenCV配备了训练器以及检测器。如果你想训练你自己的分类比如一个汽车，飞机等等，可以使用OpenCV来创造一个。所有的详细的都在这里：Cascade Classifier Training.（https://docs.opencv.org/3.4.1/dc/d88/tutorial_traincascade.html）

这里我们将实行检测。OpenCV已经包含 了很多训练过的脸部、眼睛、笑脸等等。这些XML文件放在opencv/data/haarcascades/ 里面，让我们来用OpenCV检测脸部和眼睛吧。

首先我们需要载入XML格式的分类文件。接下来用灰度模式载入我们的图片（或视频）。

```python
import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
img = cv.imread('sachin.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

现在，我们在图像中找到了面孔。 如果找到人脸，则将检测到的人脸的位置返回为Rect（x，y，w，h）。 一旦获得这些位置，就可以为面部创建ROI，并在此ROI上进行眼睛检测（因为眼睛始终在脸上！！！）。

```python
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
```

结果如下图所示：

![face.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/face.jpg)

### Additional Resources
1. Video Lecture on Face Detection and Tracking

2. An interesting interview regarding Face Detection by Adam Harvey