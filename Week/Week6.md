## Week 6 Advice for Applying Machine Learning
### 大綱
本週的重點是 - `Debugging a learning algorithm`，當發現在用我們建立的hypothesis去測試新的數據時有很大的誤差產生，我們要如何去診斷我們的learning algorithm是哪裡發生問題，該如何改進。

`Evaluating a hypothesis`，我們必須有一個客觀的數據可以評估hypothesis function的好壞，會先將數據分成兩堆**training set(70%) 和 test test(30%)**，利用traing set來找出hypothesie function最佳化theta，再利用test test來計算**regressuib和classification的hypothesis error**。

`Model Selection`，如何讓model generalize，會先將數據分成三堆**training set(60%) Cross vaildation set(20%) test test(20%)**，利用training set來找出hypothesie function最佳化theta，再利用Cross vaildation set，找出polynomial的最佳degreee，最後利用test test來計算hypothesis error。
 
再來的重點是，利用下列兩個圖形來理解`Bias跟Variance`關係:

**1. X軸: d(polynome degree) Y軸:cost。**

* high bias(underfitting)，d小，traing和vaildation cost都大且相近。
* high variance(overfitting)，d大，traing cost下降，但vaildation cost上升。

**2. X軸: lamba(regularization) Y軸:cost。**

* high variance(overfitting)，lamba小，traing cost小，但vaildation cost大。
* high bias(underfitting)，lamba大，traing和vaildation cost都上升且相近。

`Model & lamba coombo Selection`，把不同degree的model跟不同大小的lamba做所有的排列組合，把個組合都利用training set來找出每組hypothesie function最佳化theta，**再利用Cross vaildation set來計算error，找出最佳的hypothesie function(lamba = 0來進行測試)**，最後才利用test test來驗證hypothesis error。

`learning curves`，如何從learning curves去判斷high bias or high variacne。

**X軸: error Y軸:traing set size**

high bias跟high variance的各種狀況。

`典型的診斷手法`

* 增加trainig example: fix high variance
* 降低feature數量: fix high variance
* 增加feature數量: fix high bias
* 增加polynomial degree: fix high bias
* 降低lamba: fix high bias

`錯誤分析`

* 快速設計，快速實作，快速驗證
* 畫出learning curve，找出我們需要那種診斷手法
* 錯誤分析，分析檢差錯誤example的原因

`Skewed classes`，是指這個class在整個data set是佔很少數。利用`percision,recall,F score`，來評估hypothesis performance。


### 重點
* **如何透過資料分群，來驗證hypothesis, model, lamba**
* **如何從各種不同關係圖觀察出是high bias或variance的問題**
* **如何根據問題，下診斷手法**
* **錯誤分析的流程**
* **對於Skewed classes的評估**

### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week6/1.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week6/2.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week6/3.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week6/4.png)

