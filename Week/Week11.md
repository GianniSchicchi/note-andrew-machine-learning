## Week 11 Application Example: Photo OCR

### 大綱

`Problem description and pipeline`

* Problem: 如何辨識照片上的文字?
* Pipeline:
  * 1. 先偵測出照片上的文字區塊。
  * 2. 再將文字區塊上的文字進行拆解成丹一文字。
  * 3. 在對單一文字逐一進行辨識。

  
`Ceiling analysis` 

當完成一照片上的文字辨識的功能，我們可以根據各pipeline的準確度數據，來推估目前哪一個pipeline是最具有改善的空間。

### 重點

* 了解pipeline跟Ceiling analysis。

### Concept Graph

