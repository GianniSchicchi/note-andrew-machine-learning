## Week 7 Support Vector Machines

### 大綱
`SVM`是一種supervised machine learning algorithm，簡化了logistic regression，**SVM的hypotheis function的output只有0,1兩種，而logistic regression的hypotheis function的output是介於0~1之間，表示機率。**

`Large Margin Classifier`，SVM就是一種Large Margin Classifier，也就是說**SVM的decision boundary必須離postive和negative examples越遠越好。** [SVM解釋](https://www.zhihu.com/question/21094489)

`Kernals`，**可以讓SVM來分進行複雜的非線性分類**。常見的`Gaussian Kernel`，定義一個`similarity function`來評估測試資料跟`landmark`的相似程度。當測試資料跟landmark夠相似similarity function回傳1，如果非常不相似則回傳0。如何挑出landmark，最間單的方法就是從測試資料隨機挑出m個來當landmark。當給予任何一個點x，可以計算出跟所有landmarks之間的similarity大小，因此可以得到一組similarity的向量，這時**SVM with kernel的cost function可以改成用這個向量f來表示。**

`Choosing SVM Parameters`，SVM with kernel的cost function中兩個parameter可以選擇。

* C = (1/λ) => large => high variance / lower bias
            
* 高斯核函数中的σ^2 => large => f smoothly => high bias / lower viarance

`Using SVM`，在實作上如何使用SVM

* 1. 選擇參數C。
* 2. 選擇kernel函數，如果feature數量遠大於training set數量，會選擇使用liner kernel，相反，則會使用Gaussian kernel。 

`Multi-class classification`，許多現成的SVM library都已經具有Multi-class的功能，當然也可以用 one-vs-all的技巧來進行多分類。

`Logistic Regression vs. SVMs`

* n(feature)大，m(training set size)小 => Logistic Regression or SVM without kernel
* n小，m適中 => SVM with kernel
* n小，m大 => add more feature, then use Logistic Regression or SVM without kernel

### 重點

* 了解SVM的hypotheis function的output跟logistic funciton的差異。
* 了解SVM跟Large Margin Classifier的關係。
* Kernel在SVM扮演的角色，cost function如何寫，有哪些參數會影響。
* 根據feature，training set size，來決定要使用Logistic Regression或SVM(with/without kernel)。

### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week7/1.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week7/2.png)

