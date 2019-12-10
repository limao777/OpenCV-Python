# OpenCV-Python
The Opencv-Python tutorial Chinese translation
***

本章由limao777进行翻译，水平有限不足之处请见谅~


# Computational Photography(摄影算法技术)
本章你将学习到不同的摄影算法技术如图像降噪等

## 图像降噪
来看看使用Non-Local Means Denoising这种好技术来去除图像噪点

### 目标
* 你将学习到Non-Local Means Denoising降噪算法
* 你将看到不同的函数方法如 cv.fastNlMeansDenoising(), cv.fastNlMeansDenoisingColored() 等

### 理论
再之前的章节，我们知道了很多图像平滑处理技术如高斯过滤（Gaussian Blurring），中值过滤（Median Blurring）等，对于少量的噪点他们处理得还行。在这些技术里面，我们在像素周围进行了一个小邻域化，并进行了一些操作（例如高斯加权平均值，中位数等）来替换中心元素。 简而言之，一个像素点的噪声去除取决于它周围。

这里有一种噪声的性质。噪声通常被认为是均值为0的随机变量。假设有个噪声点，p = p0 + n，p0是像素真实值，n是它的噪声。你可以从不同图像中获取大量的相同点（如N）并计算其平均值。理想情况下，你应该得到p=p0因为噪声平均值为0.

你能通过简答的步骤验证它。将相机静态地放在一个地方数秒钟，这将给你很多帧画面，或同一张景色大量的照片，接着写个程序找到所有帧图像的均值（现在这对你来说应该很简单了）。再用最终的结果和第一张图像帧对比，你将看到减少的噪声。不幸的是，这种简单的方法对摄像机和场景的运动并不稳健。 通常，只有一张嘈杂的图像可用。

想法很简单的，我们需要一组相似的图像来平均噪声。假设图像中一个小的地方（比如5*5）。很可能修补位置位于图像中的其他位置，有时候在它附近的一个地方。把这些相似的补丁放一起然后找它们的平均值怎么样呢？对于那些特定的地方，这很好的，如下面的图面：

![nlm_patch.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/nlm_patch.jpg)

图像中蓝色的补丁看起来都差不多，所有绿色的也差不多。因此我们拾取一个点，再在周围拾取一小块地方，再从整个图像中找到其相似的地方，把我们拾取的点替换为它们所有的平均值。这个方法就是Non-Local Means Denoising。与我们之前的滤波技术相比，它花费了更多的时间，单效果非常好。再additional resources链接里面可以找到更多详细信息和在线演示。

对于彩色图像，图像被传唤为CIELAB色彩空间，然后分裂对L和AB进行降噪。

### OpenCV中的图像降噪
OpenCV提供四个变量

1.cv.fastNlMeansDenoising() - 用于单一的灰度图像处理

2.cv.fastNlMeansDenoisingColored() - 处理彩色图像

3.cv.fastNlMeansDenoisingMulti() - 处理短时间内捕获的一系列图像（灰度的）

4.cv.fastNlMeansDenoisingColoredMulti() - 同上，但是处理彩色的

相同的参数有：

* h : 决定滤波器强度的参数。 较高的h值可以更好地消除噪点，但同时也可以消除图像细节。（10比较OK）
* hForColorComponents : 与h相同，但仅适用于彩色图像。 （通常与h相同）
* templateWindowSize : 应为奇数。（推荐7）
* searchWindowSize : 应该为奇数。（推荐21）

可以访问additional resources的第一个链接获取详细的参数项

我们将在此处演示2和3。剩下的留给你自己。

**1. cv.fastNlMeansDenoisingColored()**

前文说到，它是去除彩色照片的噪声的。（噪声假设为高斯噪声）。看下面例子：

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('die.png')
dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()
```

以下是结果的放大版本。 我的输入图像的高斯噪声为σ= 25。 结果如下：

![nlm_result1.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/nlm_result1.jpg)

**2. cv.fastNlMeansDenoisingMulti()**
现在我们在一个短视频下做同样的事。第一个参数是有噪声短片的前几帧。第二个参数是imgToDenoiseIndex指明我们那一帧去除噪声，为此我们在输入列表中传递真的索引。第三个是temporalWindowSize指明我们要取临近的多少帧画面来计算降噪。它应该是奇数。在那个例子中，所有帧都被使用作为其参数来参与计算。举个例子，你输入了5帧视频，让imgToDenoiseIndex = 2 以及 temporalWindowSize = 3。那么-1帧、-2帧以及-3帧则用于对-2帧降噪，如下：
```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
cap = cv.VideoCapture('vtest.avi')
# create a list of first 5 frames
img = [cap.read()[1] for i in xrange(5)]
# convert all to grayscale
gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in img]
# convert all to float64
gray = [np.float64(i) for i in gray]
# create a noise of variance 25
noise = np.random.randn(*gray[1].shape)*10
# Add this noise to images
noisy = [i+noise for i in gray]
# Convert back to uint8
noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]
# Denoise 3rd frame considering all the 5 frames
dst = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
plt.subplot(131),plt.imshow(gray[2],'gray')
plt.subplot(132),plt.imshow(noisy[2],'gray')
plt.subplot(133),plt.imshow(dst,'gray')
plt.show()
```

下图展示了放大的结果：

![nlm_multi.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/nlm_multi.jpg)

该运算需要大量时间。在上图中，第一张是原始的帧图像，第二张是有噪音的一张，第三张是降噪后的图像。

### Additional Resources

http://www.ipol.im/pub/art/2011/bcm_nlm/ (它包含详细信息，在线演示等。强烈建议访问。我们的测试图像是从此链接生成的)
Online course at coursera (第一张图片来自这里https://www.coursera.org/course/images)

## 图像修补
您是否有一张老旧的退化照片，上面有很多黑点和划痕？那么让我们尝试使用一种称为图像修复的技术来还原它们。

### 目标
* 你将学习到使用inpainting的函数方法来去除老照片的小噪点，划痕等
* 你将看到OpenCV的inpainting函数方法

### 基础
我们大多数人家里面都有一些老的质量较差的有一些黑点以及有一些划痕等的照片。你有想过将他们复原吗？我们不能简单地在绘图工具中将他们擦除，因为它将简单地用白色结构代替黑色，这是没有用的。这些案例中，一种名叫“图像修补”的技术将被应用。基本的方法十分简单：将这些坏的污迹用图像周围的像素点替换以至于使这些地方看起来像它周围的图像。假设有图像如下（来自 Wikipedia）：

![inpaint_basics.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/inpaint_basics.jpg)

有数种算法被设计来达到此目的，OpenCV提供了其中两个。它们都用同一个方法函数，叫cv.inpaint()

第一种算法基于论文《一种基于快速匹配方法的的图像修复技术》（An Image Inpainting Technique Based on the Fast Marching Method），作者Alexandru Telea于2004年发布。它基于快速匹配技术，假设有个区域待修复，算法从该区域的边界开始然后再逐渐像里面填充边界的所有内容。在要修复的邻域上的像素周围它需要一个小的邻域，用附近所有已知像素的归一化加权总和替换该像素。权重的选择是重要的事情。 那些位于该点附近，边界法线附近的像素和那些位于边界轮廓线上的像素将获得更大的权重。一旦像素点被修复，算法将移动到另一个最近的像素点使用快速匹配方法。该方法（FMM）可以确保先修复已知像素附近的那些像素，这样就可以像手动启发式操作一样工作，我们可以使用cv.INPAINT_TELEA标志来使能这个算法。

第二种算法基于论文《导航-划痕，流体动力学以及图像和视频修复》（Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting），作者 Bertalmio, Marcelo, Andrea L. Bertozzi, Guillermo Sapiro在2001年发布。该算法基于流体动力学并利用偏微分方程，其基本原理使启发式的。它首先从一直的区域边界慢慢走向未知的边界（因为边界是连续的），它延续了等强度线（线连接具有相同强度的点，就像轮廓连接具有相同高程的点一样）同时在修复区域的边界匹配梯度矢量。对于此，一些流体动力学的方法被使用了。当它们被使用时，它将将填充颜色以减少该区域的最小差异。该方法可以用cv.INPAINT_NS来使能。

### 代码
我们需要创建一个和图像一样大的遮罩，非0的像素代表要修复的未知，其他的都很简单。我们的图片上有一些黑色的划痕（手动加的）。我们能用画图程序创建一个相应的划痕。

```python
import numpy as np
import cv2 as cv
img = cv.imread('messi_2.jpg')
mask = cv.imread('mask2.png',0)
dst = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)
cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()
```

看下面的结果。第一张图像展示了有问题的图像，第二张是遮罩，第三张是第一种算法的结果，最后一张是第二种算法的结果。

![inpaint_result.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/inpaint_result.jpg)

### Additional Resources
1.Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro. "Navier-stokes, fluid dynamics, and image and video inpainting." In Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on, vol. 1, pp. I-355. IEEE, 2001.
2.Telea, Alexandru. "An image inpainting technique based on the fast marching method." Journal of graphics tools 9.1 (2004): 23-34.
（主要是两篇算法论文的出处）

### Exercises
1.OpenCV随附了一个有关修复的交互式示例，samples/python/inpaint.py，可以尝试以下。
2.几个月前，我观看了有关Content-Aware Fill的视频，Content-Aware Fill是Adobe Photoshop中使用的一种先进的修复技术。 在进一步的搜索中，我发现GIMP中已经存在相同的技术，但名称不同，叫“ Resynthesizer”（您需要安装单独的插件）。 我相信您会喜欢这项技术的。

## 高动态范围（HDR）
### 目标
* 了解如何根据曝光顺序生成和显示HDR图像。
* 使用曝光融合来合并曝光序列。

### 理论
高动态范围图像（HDRI或HDR）与标准的数字成像或摄影技术相比，它是一种用于成像和摄影的技术，可以再现更大的动态亮度范围。虽然人眼可以适应各种光照条件，但是大多数成像设备每个通道使用8位，因此我们仅限于256级。 当我们拍摄现实世界的照片时，明亮的区域可能会曝光过度，而黑暗的区域可能会曝光不足，因此我们无法一次曝光就捕获所有细节。 HDR成像适用于每个通道使用8位以上（通常为32位浮点值）的图像，从而允许更大的动态范围。

有很多种方法获取HDR图像，最通常的方法就是使用照相机用不同的曝光度拍摄图片，然后将这些曝光的图片合并。了解相机的响应功能非常有用，并且有一些算法可以对其进行估算。在这些HDR摘片合并后，它将被转换为8位就像通常的照片那样，这总处理叫做色调映射，当场景或相机的对象在两次拍摄之间移动时，还会增加其他复杂性，因为应记录并调整具有不同曝光度的图像。

这个教程中我们展示两种算法（Debvec, Robertson）来生成和展示不同曝光序列的HDR图像，再演示另外一种方法叫曝光融合（Mertens）。它不需要曝光时间数据就生成低动态范围图像。此外我们估计相机响应函数（CRF）对于许多计算机视觉算法都具有重要价值。 HDR流水线的每个步骤都可以使用不同的算法和参数来实现，因此请查看参考手册以了解所有内容。

### 曝光序列HDR
这个教程中我们将看到如下场景，我们有4张曝光的图片，使用15, 2.5, 1/4 and 1/30 secondsv曝光参数（你也能再Wikipedia取去下载这张图）

![exposures.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/exposures.jpg)


#### 1.把曝光照片导入到list中
第一步简单地把所有照片放进一个list，额外地，对于常规的HDR算法我们将需要曝光时间。注意这些数据类型，图像应该1个通道或3个通道的8位（np.uint8），曝光时间需要float32类型，单位为秒。

```python
import cv2 as cv
import numpy as np
# Loading exposure images into a list
img_fn = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
img_list = [cv.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)
```

#### 2.把曝光序列合并进HDR图像
在此阶段，我们将曝光序列合并为一张HDR图像，显示了OpenCV中的两种方法可用。 第一种方法是Debvec，第二种方法是Robertson。 请注意，HDR图像的类型为float32，而不是uint8，因为它包含所有曝光图像的完整动态范围。

```python
# Merge exposures to HDR image
merge_debvec = cv.createMergeDebevec()
hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())
```

#### 3.HDR图像调色
我们把32位的浮点数HDR数据放进[0...1]中，事实上，在一些例子中这些数据可能比1大或者比0小，所以注意我们稍后必须将数据切断以免溢出。
```python
# Tonemap HDR image
tonemap1 = cv.createTonemapDurand(gamma=2.2)
res_debvec = tonemap1.process(hdr_debvec.copy())
tonemap2 = cv.createTonemapDurand(gamma=1.3)
res_robertson = tonemap2.process(hdr_robertson.copy())
```

#### 4.使用Mertens fusion来合并曝光序列
在这里，我们展示了一种替代算法，用于合并曝光图像，而我们不需要曝光时间。 我们也不需要使用任何色调映射算法，因为Mertens算法已经为我们提供了[0..1]范围内的结果。
```python
# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)
```

#### 5.转换为8位数据并保存
为了保存图像，我们必须把数据转换为8位图像在[0..255]范围内
```python
# Convert datatype to 8-bit and save
res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
cv.imwrite("ldr_debvec.jpg", res_debvec_8bit)
cv.imwrite("ldr_robertson.jpg", res_robertson_8bit)
cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)
```

### 结果
我们可以看到不同的结果，但可以认为每种算法都有其他额外的参数，您应该将它们附加以达到期望的结果。 最佳实践是尝试不同的方法，然后看看哪种方法最适合您的场景。

#### Debvec:
![ldr_debvec.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/ldr_debvec.jpg)

#### Robertson:
![ldr_robertson.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/ldr_robertson.jpg)

#### Mertenes Fusion:
![fusion_mertens.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/fusion_mertens.jpg)

### 预估相机响应功能
相机响应功能（CRF）使我们可以将场景辐射度与测得的强度值联系起来。CRF在某些计算机视觉算法（包括HDR算法）中非常重要。 在这里，我们估计逆相机响应函数并将其用于HDR合并。

```python
# Estimate camera response function (CRF)
cal_debvec = cv.createCalibrateDebevec()
crf_debvec = cal_debvec.process(img_list, times=exposure_times)
hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy(), response=crf_debvec.copy())
cal_robertson = cv.createCalibrateRobertson()
crf_robertson = cal_robertson.process(img_list, times=exposure_times)
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy(), response=crf_robertson.copy())
```
相机响应功能由每个颜色通道的256长度向量表示。 对于此序列，我们得到以下估算值：
![crf.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/crf.jpg)

### Additional Resources
1.Paul E Debevec and Jitendra Malik. Recovering high dynamic range radiance maps from photographs. In ACM SIGGRAPH 2008 classes, page 31. ACM, 2008.
2.Mark A Robertson, Sean Borman, and Robert L Stevenson. Dynamic range improvement through multiple exposures. In Image Processing, 1999. ICIP 99. Proceedings. 1999 International Conference on, volume 3, pages 159–163. IEEE, 1999.
3.Tom Mertens, Jan Kautz, and Frank Van Reeth. Exposure fusion. In Computer Graphics and Applications, 2007. PG'07. 15th Pacific Conference on, pages 382–390. IEEE, 2007.
4.Images from Wikipedia-HDR
（主要是一些论文出处和维基百科HDR篇）

### Exercises
1.尝试所有色调图算法：Drago，Durand，Mantiuk和Reinhard。
2.尝试更改HDR校准和色调图方法中的参数。