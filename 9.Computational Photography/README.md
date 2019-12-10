# OpenCV-Python
The Opencv-Python tutorial Chinese translation
***

������limao777���з��룬ˮƽ���޲���֮�������~


# Computational Photography(��Ӱ�㷨����)
�����㽫ѧϰ����ͬ����Ӱ�㷨������ͼ�����

## ͼ����
������ʹ��Non-Local Means Denoising���ֺü�����ȥ��ͼ�����

### Ŀ��
* �㽫ѧϰ��Non-Local Means Denoising�����㷨
* �㽫������ͬ�ĺ��������� cv.fastNlMeansDenoising(), cv.fastNlMeansDenoisingColored() ��

### ����
��֮ǰ���½ڣ�����֪���˺ܶ�ͼ��ƽ�����������˹���ˣ�Gaussian Blurring������ֵ���ˣ�Median Blurring���ȣ�����������������Ǵ���û��С�����Щ�������棬������������Χ������һ��С���򻯣���������һЩ�����������˹��Ȩƽ��ֵ����λ���ȣ����滻����Ԫ�ء� �����֮��һ�����ص������ȥ��ȡ��������Χ��

������һ�����������ʡ�����ͨ������Ϊ�Ǿ�ֵΪ0����������������и������㣬p = p0 + n��p0��������ʵֵ��n����������������ԴӲ�ͬͼ���л�ȡ��������ͬ�㣨��N����������ƽ��ֵ����������£���Ӧ�õõ�p=p0��Ϊ����ƽ��ֵΪ0.

����ͨ�����Ĳ�����֤�����������̬�ط���һ���ط������ӣ��⽫����ܶ�֡���棬��ͬһ�ž�ɫ��������Ƭ������д�������ҵ�����֡ͼ��ľ�ֵ�������������˵Ӧ�úܼ��ˣ����������յĽ���͵�һ��ͼ��֡�Աȣ��㽫�������ٵ����������ҵ��ǣ����ּ򵥵ķ�����������ͳ������˶������Ƚ��� ͨ����ֻ��һ�����ӵ�ͼ����á�

�뷨�ܼ򵥵ģ�������Ҫһ�����Ƶ�ͼ����ƽ������������ͼ����һ��С�ĵط�������5*5�����ܿ����޲�λ��λ��ͼ���е�����λ�ã���ʱ������������һ���ط�������Щ���ƵĲ�����һ��Ȼ�������ǵ�ƽ��ֵ��ô���أ�������Щ�ض��ĵط�����ܺõģ��������ͼ�棺

![nlm_patch.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/nlm_patch.jpg)

ͼ������ɫ�Ĳ�������������࣬������ɫ��Ҳ��ࡣ�������ʰȡһ���㣬������ΧʰȡһС��ط����ٴ�����ͼ�����ҵ������Ƶĵط���������ʰȡ�ĵ��滻Ϊ�������е�ƽ��ֵ�������������Non-Local Means Denoising��������֮ǰ���˲�������ȣ��������˸����ʱ�䣬��Ч���ǳ��á���additional resources������������ҵ�������ϸ��Ϣ��������ʾ��

���ڲ�ɫͼ��ͼ�񱻴���ΪCIELABɫ�ʿռ䣬Ȼ����Ѷ�L��AB���н��롣

### OpenCV�е�ͼ����
OpenCV�ṩ�ĸ�����

1.cv.fastNlMeansDenoising() - ���ڵ�һ�ĻҶ�ͼ����

2.cv.fastNlMeansDenoisingColored() - �����ɫͼ��

3.cv.fastNlMeansDenoisingMulti() - �����ʱ���ڲ����һϵ��ͼ�񣨻Ҷȵģ�

4.cv.fastNlMeansDenoisingColoredMulti() - ͬ�ϣ����Ǵ����ɫ��

��ͬ�Ĳ����У�

* h : �����˲���ǿ�ȵĲ����� �ϸߵ�hֵ���Ը��õ�������㣬��ͬʱҲ��������ͼ��ϸ�ڡ���10�Ƚ�OK��
* hForColorComponents : ��h��ͬ�����������ڲ�ɫͼ�� ��ͨ����h��ͬ��
* templateWindowSize : ӦΪ���������Ƽ�7��
* searchWindowSize : Ӧ��Ϊ���������Ƽ�21��

���Է���additional resources�ĵ�һ�����ӻ�ȡ��ϸ�Ĳ�����

���ǽ��ڴ˴���ʾ2��3��ʣ�µ��������Լ���

**1. cv.fastNlMeansDenoisingColored()**

ǰ��˵��������ȥ����ɫ��Ƭ�������ġ�����������Ϊ��˹�����������������ӣ�

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

�����ǽ���ķŴ�汾�� �ҵ�����ͼ��ĸ�˹����Ϊ��= 25�� ������£�

![nlm_result1.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/nlm_result1.jpg)

**2. cv.fastNlMeansDenoisingMulti()**
����������һ������Ƶ����ͬ�����¡���һ����������������Ƭ��ǰ��֡���ڶ���������imgToDenoiseIndexָ��������һ֡ȥ��������Ϊ�������������б��д��������������������temporalWindowSizeָ������Ҫȡ�ٽ��Ķ���֡���������㽵�롣��Ӧ�������������Ǹ������У�����֡����ʹ����Ϊ�������������㡣�ٸ����ӣ���������5֡��Ƶ����imgToDenoiseIndex = 2 �Լ� temporalWindowSize = 3����ô-1֡��-2֡�Լ�-3֡�����ڶ�-2֡���룬���£�
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

��ͼչʾ�˷Ŵ�Ľ����

![nlm_multi.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/nlm_multi.jpg)

��������Ҫ����ʱ�䡣����ͼ�У���һ����ԭʼ��֡ͼ�񣬵ڶ�������������һ�ţ��������ǽ�����ͼ��

### Additional Resources

http://www.ipol.im/pub/art/2011/bcm_nlm/ (��������ϸ��Ϣ��������ʾ�ȡ�ǿ�ҽ�����ʡ����ǵĲ���ͼ���ǴӴ��������ɵ�)
Online course at coursera (��һ��ͼƬ��������https://www.coursera.org/course/images)

## ͼ���޲�
���Ƿ���һ���Ͼɵ��˻���Ƭ�������кܶ�ڵ�ͻ��ۣ���ô�����ǳ���ʹ��һ�ֳ�Ϊͼ���޸��ļ�������ԭ���ǡ�

### Ŀ��
* �㽫ѧϰ��ʹ��inpainting�ĺ���������ȥ������Ƭ��С��㣬���۵�
* �㽫����OpenCV��inpainting��������

### ����
���Ǵ�����˼����涼��һЩ�ϵ������ϲ����һЩ�ڵ��Լ���һЩ���۵ȵ���Ƭ��������������Ǹ�ԭ�����ǲ��ܼ򵥵��ڻ�ͼ�����н����ǲ�������Ϊ�����򵥵��ð�ɫ�ṹ�����ɫ������û���õġ���Щ�����У�һ�����С�ͼ���޲����ļ�������Ӧ�á������ķ���ʮ�ּ򵥣�����Щ�����ۼ���ͼ����Χ�����ص��滻������ʹ��Щ�ط�������������Χ��ͼ�񡣼�����ͼ�����£����� Wikipedia����

![inpaint_basics.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/inpaint_basics.jpg)

�������㷨��������ﵽ��Ŀ�ģ�OpenCV�ṩ���������������Ƕ���ͬһ��������������cv.inpaint()

��һ���㷨�������ġ�һ�ֻ��ڿ���ƥ�䷽���ĵ�ͼ���޸���������An Image Inpainting Technique Based on the Fast Marching Method��������Alexandru Telea��2004�귢���������ڿ���ƥ�似���������и�������޸����㷨�Ӹ�����ı߽翪ʼȻ���������������߽���������ݡ���Ҫ�޸��������ϵ�������Χ����Ҫһ��С�������ø���������֪���صĹ�һ����Ȩ�ܺ��滻�����ء�Ȩ�ص�ѡ������Ҫ�����顣 ��Щλ�ڸõ㸽�����߽編�߸��������غ���Щλ�ڱ߽��������ϵ����ؽ���ø����Ȩ�ء�һ�����ص㱻�޸����㷨���ƶ�����һ����������ص�ʹ�ÿ���ƥ�䷽�����÷�����FMM������ȷ�����޸���֪���ظ�������Щ���أ������Ϳ������ֶ�����ʽ����һ�����������ǿ���ʹ��cv.INPAINT_TELEA��־��ʹ������㷨��

�ڶ����㷨�������ġ�����-���ۣ����嶯��ѧ�Լ�ͼ�����Ƶ�޸�����Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting�������� Bertalmio, Marcelo, Andrea L. Bertozzi, Guillermo Sapiro��2001�귢�������㷨�������嶯��ѧ������ƫ΢�ַ��̣������ԭ��ʹ����ʽ�ġ������ȴ�һֱ������߽���������δ֪�ı߽磨��Ϊ�߽��������ģ����������˵�ǿ���ߣ������Ӿ�����ͬǿ�ȵĵ㣬�����������Ӿ�����ͬ�̵߳ĵ�һ����ͬʱ���޸�����ı߽�ƥ���ݶ�ʸ�������ڴˣ�һЩ���嶯��ѧ�ķ�����ʹ���ˡ������Ǳ�ʹ��ʱ�������������ɫ�Լ��ٸ��������С���졣�÷���������cv.INPAINT_NS��ʹ�ܡ�

### ����
������Ҫ����һ����ͼ��һ��������֣���0�����ش���Ҫ�޸���δ֪�������Ķ��ܼ򵥡����ǵ�ͼƬ����һЩ��ɫ�Ļ��ۣ��ֶ��ӵģ����������û�ͼ���򴴽�һ����Ӧ�Ļ��ۡ�

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

������Ľ������һ��ͼ��չʾ���������ͼ�񣬵ڶ��������֣��������ǵ�һ���㷨�Ľ�������һ���ǵڶ����㷨�Ľ����

![inpaint_result.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/inpaint_result.jpg)

### Additional Resources
1.Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro. "Navier-stokes, fluid dynamics, and image and video inpainting." In Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on, vol. 1, pp. I-355. IEEE, 2001.
2.Telea, Alexandru. "An image inpainting technique based on the fast marching method." Journal of graphics tools 9.1 (2004): 23-34.
����Ҫ����ƪ�㷨���ĵĳ�����

### Exercises
1.OpenCV�渽��һ���й��޸��Ľ���ʽʾ����samples/python/inpaint.py�����Գ������¡�
2.������ǰ���ҹۿ����й�Content-Aware Fill����Ƶ��Content-Aware Fill��Adobe Photoshop��ʹ�õ�һ���Ƚ����޸������� �ڽ�һ���������У��ҷ���GIMP���Ѿ�������ͬ�ļ����������Ʋ�ͬ���С� Resynthesizer��������Ҫ��װ�����Ĳ������ ����������ϲ��������ġ�

## �߶�̬��Χ��HDR��
### Ŀ��
* �˽���θ����ع�˳�����ɺ���ʾHDRͼ��
* ʹ���ع��ں����ϲ��ع����С�

### ����
�߶�̬��Χͼ��HDRI��HDR�����׼�����ֳ������Ӱ������ȣ�����һ�����ڳ������Ӱ�ļ������������ָ���Ķ�̬���ȷ�Χ����Ȼ���ۿ�����Ӧ���ֹ������������Ǵ���������豸ÿ��ͨ��ʹ��8λ��������ǽ�����256���� ������������ʵ�������Ƭʱ��������������ܻ��ع���ȣ����ڰ���������ܻ��عⲻ�㣬��������޷�һ���ع�Ͳ�������ϸ�ڡ� HDR����������ÿ��ͨ��ʹ��8λ���ϣ�ͨ��Ϊ32λ����ֵ����ͼ�񣬴Ӷ��������Ķ�̬��Χ��

�кܶ��ַ�����ȡHDRͼ����ͨ���ķ�������ʹ��������ò�ͬ���ع������ͼƬ��Ȼ����Щ�ع��ͼƬ�ϲ����˽��������Ӧ���ܷǳ����ã�������һЩ�㷨���Զ�����й��㡣����ЩHDRժƬ�ϲ���������ת��Ϊ8λ����ͨ������Ƭ���������ܴ������ɫ��ӳ�䣬������������Ķ�������������֮���ƶ�ʱ�������������������ԣ���ΪӦ��¼���������в�ͬ�ع�ȵ�ͼ��

����̳�������չʾ�����㷨��Debvec, Robertson�������ɺ�չʾ��ͬ�ع����е�HDRͼ������ʾ����һ�ַ������ع��ںϣ�Mertens����������Ҫ�ع�ʱ�����ݾ����ɵͶ�̬��Χͼ�񡣴������ǹ��������Ӧ������CRF��������������Ӿ��㷨��������Ҫ��ֵ�� HDR��ˮ�ߵ�ÿ�����趼����ʹ�ò�ͬ���㷨�Ͳ�����ʵ�֣������鿴�ο��ֲ����˽��������ݡ�

### �ع�����HDR
����̳������ǽ��������³�����������4���ع��ͼƬ��ʹ��15, 2.5, 1/4 and 1/30 secondsv�ع��������Ҳ����Wikipediaȡȥ��������ͼ��

![exposures.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/exposures.jpg)


#### 1.���ع���Ƭ���뵽list��
��һ���򵥵ذ�������Ƭ�Ž�һ��list������أ����ڳ����HDR�㷨���ǽ���Ҫ�ع�ʱ�䡣ע����Щ�������ͣ�ͼ��Ӧ��1��ͨ����3��ͨ����8λ��np.uint8�����ع�ʱ����Ҫfloat32���ͣ���λΪ�롣

```python
import cv2 as cv
import numpy as np
# Loading exposure images into a list
img_fn = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
img_list = [cv.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)
```

#### 2.���ع����кϲ���HDRͼ��
�ڴ˽׶Σ����ǽ��ع����кϲ�Ϊһ��HDRͼ����ʾ��OpenCV�е����ַ������á� ��һ�ַ�����Debvec���ڶ��ַ�����Robertson�� ��ע�⣬HDRͼ�������Ϊfloat32��������uint8����Ϊ�����������ع�ͼ���������̬��Χ��

```python
# Merge exposures to HDR image
merge_debvec = cv.createMergeDebevec()
hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())
```

#### 3.HDRͼ���ɫ
���ǰ�32λ�ĸ�����HDR���ݷŽ�[0...1]�У���ʵ�ϣ���һЩ��������Щ���ݿ��ܱ�1����߱�0С������ע�������Ժ���뽫�����ж����������
```python
# Tonemap HDR image
tonemap1 = cv.createTonemapDurand(gamma=2.2)
res_debvec = tonemap1.process(hdr_debvec.copy())
tonemap2 = cv.createTonemapDurand(gamma=1.3)
res_robertson = tonemap2.process(hdr_robertson.copy())
```

#### 4.ʹ��Mertens fusion���ϲ��ع�����
���������չʾ��һ������㷨�����ںϲ��ع�ͼ�񣬶����ǲ���Ҫ�ع�ʱ�䡣 ����Ҳ����Ҫʹ���κ�ɫ��ӳ���㷨����ΪMertens�㷨�Ѿ�Ϊ�����ṩ��[0..1]��Χ�ڵĽ����
```python
# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)
```

#### 5.ת��Ϊ8λ���ݲ�����
Ϊ�˱���ͼ�����Ǳ��������ת��Ϊ8λͼ����[0..255]��Χ��
```python
# Convert datatype to 8-bit and save
res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
cv.imwrite("ldr_debvec.jpg", res_debvec_8bit)
cv.imwrite("ldr_robertson.jpg", res_robertson_8bit)
cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)
```

### ���
���ǿ��Կ�����ͬ�Ľ������������Ϊÿ���㷨������������Ĳ�������Ӧ�ý����Ǹ����Դﵽ�����Ľ���� ���ʵ���ǳ��Բ�ͬ�ķ�����Ȼ�󿴿����ַ������ʺ����ĳ�����

#### Debvec:
![ldr_debvec.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/ldr_debvec.jpg)

#### Robertson:
![ldr_robertson.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/ldr_robertson.jpg)

#### Mertenes Fusion:
![fusion_mertens.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/fusion_mertens.jpg)

### Ԥ�������Ӧ����
�����Ӧ���ܣ�CRF��ʹ���ǿ��Խ�������������õ�ǿ��ֵ��ϵ������CRF��ĳЩ������Ӿ��㷨������HDR�㷨���зǳ���Ҫ�� ��������ǹ����������Ӧ��������������HDR�ϲ���

```python
# Estimate camera response function (CRF)
cal_debvec = cv.createCalibrateDebevec()
crf_debvec = cal_debvec.process(img_list, times=exposure_times)
hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy(), response=crf_debvec.copy())
cal_robertson = cv.createCalibrateRobertson()
crf_robertson = cal_robertson.process(img_list, times=exposure_times)
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy(), response=crf_robertson.copy())
```
�����Ӧ������ÿ����ɫͨ����256����������ʾ�� ���ڴ����У����ǵõ����¹���ֵ��
![crf.jpg](https://raw.githubusercontent.com/limao777/OpenCV-Python/master/8.Machine%20Learning/Image/crf.jpg)

### Additional Resources
1.Paul E Debevec and Jitendra Malik. Recovering high dynamic range radiance maps from photographs. In ACM SIGGRAPH 2008 classes, page 31. ACM, 2008.
2.Mark A Robertson, Sean Borman, and Robert L Stevenson. Dynamic range improvement through multiple exposures. In Image Processing, 1999. ICIP 99. Proceedings. 1999 International Conference on, volume 3, pages 159�C163. IEEE, 1999.
3.Tom Mertens, Jan Kautz, and Frank Van Reeth. Exposure fusion. In Computer Graphics and Applications, 2007. PG'07. 15th Pacific Conference on, pages 382�C390. IEEE, 2007.
4.Images from Wikipedia-HDR
����Ҫ��һЩ���ĳ�����ά���ٿ�HDRƪ��

### Exercises
1.��������ɫ��ͼ�㷨��Drago��Durand��Mantiuk��Reinhard��
2.���Ը���HDRУ׼��ɫ��ͼ�����еĲ�����