# 機器學習 - Anderw Ng
希望可以用自己理解的內容在10分鐘內解釋這11週的課程中，我到底學到些什麼東西。


## Week 1 Introduction ＆ Linear Regression with One Variable ＆ Linear Algebra Review
### 大綱

在Introduction中，先定義什麼叫`Machine Learning`(older, modern定義)，接下來解釋ML中的主要分成兩種不同的學習方法 - `supervied learning & unsupervised learning`。 

接下來對supervied learning，提出了`Linear Regression`概念，先針對我們所要處理的問題定義出一個`hypothesis function`，再來透過`cost function`來評估我們hypothesis function的誤差，最後再透過`gradient descent`來收斂hypothesis function中的theta參數，找出最佳的hypothesis function。

### 重點

* **supervied learning problem和unsupervised learning的差異**
* **supervied learning problem中regression跟classification的差異**
* **supervised regression problems**
* **linear regression中的hypothesis function和cost function的定義跟關係**
* **gradient descent algorithm的含義**
* **linear regression algorithm中運用gradient descent**


### 技能樹

![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/1.png)

## Week 2 Linear Regression with Multiple Variables
### 大綱
本週的重點還是**Linear Regression**只是從原本**One Variable**擴展到**Multiple Variables**，並利用`房價計算`的實際案例來實作`Multiple Variables Linear Regression`，讓我們可以從房子的多個特徵值和對應的歷史價格來進行訓練，進而推測不同特徵值房子的價格。

### 重點
* **multivariate linear regression的hypothesis function**
* **multivariate linear regression的gradient descent**
* **feature scaling如何進行以及優點**
* **如何決定learning rates**
* **multivariate linear regression如何運用polynomial regression**
* **normal equation和gradient descent的比較**

### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/2.png)

## Week 3 Logistic Regression
### 大綱
講完了Linear Regression後，本週重點在於處理`classification problem`，會利用`Logistic regression`來處理此問題，例如判別電子郵件是否是垃圾郵件。首先，先講到classification的基本概念(`Binary classification`)，以及相關的cost function，最後如何處理`multi-class classification`。

另外，我們會提到`regularization`，透過regularization讓我們的machine learning models避免`overfitting problem`。

### 重點
* **說明為什麼Linear Regression不適合用於classification problem**
* **了解Sigmoid function**
* **了解logistic regression model的hypothesis function的意義**
* **了解decision boundary跟hypothesis function和logistic regression的關係**
* **了解logistic regression的cost function**
* **了解gradient decent logistic regressionc和linear regression的差異**
* **說明multi-class classification one-vs-all的意義**
* **如何利用regularization來處理overfitting problem**

### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/3.png)

## Week 4 Neural Networks: Representation
### 大綱
本週會開始講到目前最流行的`Neural Networks`，如何讓電腦去透過模擬大腦學習的行為。首先，我們必須先瞭解linear regression的限制，利用**Neural Networks**處理這限制，介紹**神經元**跟邏輯函數的關係，**hidden layer**，**input layer**，**output layer**架構。何謂**activation**，最後利用間單的範例來介紹如何利用神經網路來模擬AND,OR,XNOR等function。

### 重點
* **說明當linear model的feature數量很大時會遇到怎樣的挑戰**
* **了解神經元的input和ouptput跟mathematical model的關係**
* **了解具有一層hidden layer的neural network的結構**
* **如何計算activation的值**
* **了解forward propagate**

### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/4%2C5.png)

## Week 5  Neural Networks: Learning
這一週將要學到的是`backpropagation algorithm`，這個algorithm是我認為這系列課程中做複雜的，要如何從類神經的網路的ouput的誤差去反推校正每一個偏差值。需要懂得forward propagation和back-propagation在Neural Networks的算法，接下來如何用`Gradient Checking`去驗證backpropagation是否正確。
### 大綱

### 重點
* **Neural Networks是如何處理classification problem**
* **了解Neural Networks的cost function，並跟logistic regression的cost function 比較差異**
* **了解Neural Networks的cost function中的regulation**
* **forward propagation和back-propagation，是如何計算cost function**
* **back-propagation的背後含義(數學推導太複雜了，放棄，只有懂的原理即可)**
* **Gradient Checking的原理**
* **如何決定一個適當的neural network architecture**

### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/4%2C5.png)

## Week 6 Advice for Applying Machine Learning
### 大綱
前幾週學了好幾個ML的演算法，這週的目標要學得去評估我們的所設計的演算法到底是好還不好，利用`training set`，`validation set`，`test test`將數據分組進行測試，將`Learning Curves`繪製後，來判斷我們是有可能遇到`Bias`，`Variance`的問題 ，該怎麼改進。那如果數據中有`skewed data`該要怎樣處理呢，如何在`precision`跟`recall`中取得平衡。
### 重點
* **數據跟如何分類進行驗證**
* **為何modele selection應該是cross-validation決定，Performance不應該是由training set來決定**
* **了解high bias跟higt variance**
* **如何去從Learning Curves去推估可能bias或variance問題**
* **有哪些方法可以處理bias或variance問題**
* **skewed classes問題是什麼**
* **precision，recall，F1的定義跟計算方式**


### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/6.png)

## Week 7 Support Vector Machines
### 大綱
`Support Vector Machines SVM`是另外一種supervised machine learning algorithm，主要是將logistic regression的cost function的曲線拆成兩個直線段來處理。如何從數學關係來了解SVM其實是一種`larger margin classifier`，接下來如果SVM要處理non-linear classifier問題，就是要利用`kernel`，

### 重點
* **SVM跟logistic regression的cost funciton差異**
* **SVM中的參數c跟regularization(lambda)的相似性**
* **kernels跟similarity functions的關係**
* **SVM跟logistic regression的比較**


### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/7.png)

## Week 8 Unsupervised Learning
### 大綱
這週終於來到`Unsupervised Learning`，如何利用` k-Means algorithm `將資料分類`clustering`。再來提到了`Principal Components Analysis PCA`將負責的資料數據進行壓縮，加速演算法的學習以及視覺化的觀察。

### 重點
* **如何利用K-means來進行資料分類**
* **主成分分析法(Principal Componet Analysis, PCA)將資料降維**
* **如何將壓縮過的資料在反推(reconstruction)回原始資料(有失真的狀況)**

### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/8.png)

## Week 9 Anomaly Detection ＆ Recommender Systems
### 大綱
本週主要是講應用，也比較有趣多了，第一個部分是`Anomaly Detection 異常檢測`，如何利用`Gaussian distribution`來進行異常檢測。接下來的部分，更是有趣，那就是`Recommender Systems 推薦系統`，會利用`collaborative filtering algorithm` and `low-rank matrix factorization`。

### 重點
* **了解Gaussian distribution的mean跟standard deviations**
* **如何選擇anomaly detection system的threshold**
* **multivariate Gaussian distribution，多個因子Anomaly Detection處理**
* **如何使用collaborative filtering algorithm**
* **collaborative filtering algorithm和low rank matrix factorization的關係**


### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/9.png)

## Week 10 Large Scale Machine Learning
### 大綱
本週提出幾種不同方式來進行大量的數據學習，一種是改進演算法，`Stochastic and min-Batch Gradient Descent`，另外一種這是應用平行計算的原理，讓多台電腦同時進行學習`Map Reduce and  Data Parallelism`。

### 重點
* **了解stochastic gradient descent algorithm**
* **了解min-Batch gradient descent algorithm**
* **說明stochastic和min-Batch不同**
* **什麼是online learning**
* **什麼是Map Reduce and  Data Parallelism**

### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/10.png)

## Week 11 application Example: Photo OCR
### 大綱
本週也是講個實際的應用，`電腦視覺OCR`，到底電腦如何辨識文字，`sliding window`是怎樣用來進行OCR辨識。另外透過這個應用，我們也會了解如何把ML的過程進行`pipeline`，透過pipeline的效能分析，可以讓我們知道要如何分配資源在最主要的模組下，進行提升最大的學習效能。

### 重點
* **了解Photo OCR的pipeline**
* **sliding window的概念**
* **什麼請況下，可以利用人工合成數據**
* **如何透過pipeline的分析，了解整個ML過程的bottleneck**


### 技能樹
![Syntax highlighting example](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/%E6%8A%80%E8%83%BD%E6%A8%B9/11.png)
