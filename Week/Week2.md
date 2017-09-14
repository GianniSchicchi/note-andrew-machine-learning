## Week 2 Linear Regression with Multiple Variables
### 大綱

從原本的Linear Regression問題從**One Variable**擴展到**Multiple Variables**，讓我們可**同時對多個特徵值**一起進行評估推算。跟上週一樣，仍然會提到hypothesis function，cost function，gradient descent，但不同的是我們必須把這個東西改用`vector`來表示。

當有多個特徵值要進行處理時，我們就必須注意到`Feature Normalization`，將不同範圍大小的特徵值限制在**-1至1，或-0.5至0.5**，特徵值經過Feature Normalization後可以加速gradient descent的收斂。一般會利用這兩種方式- `feature scaling`和`mean normalization`來進行Feature Normalization。

`Gradient Decent tips`，如何知道**Gradient Decent的是否正確在收斂**，利用圖型來判斷 (Y軸: cost funciton的值，X軸: iterations的次數) ，另外，**如何選擇一個適當大小的的學習速率也是個重點**，學習速率過小，收斂速度慢，學習速率過大，可能無法收斂。根據經驗法則，每次學習速率都是除以3來進行測試。

如果linear hypothesis function無法滿足大多數的資料時，我們可以`合併`某些特徵值成為新的特徵，或者將某些特徵值進行平方，立方，開方根等，將linear hypothesis function改成`polynomial regression`，最終目的就是讓hypothesis function跟貼切data，讓我們預測的結果可以更加準確，記得當使用**polynomial regression**時，必須更注意**Feature Normalization**。

`Normal equation`，是另外一種不需要iteration找出最佳的theta值的方法，請比較**Normal equation**和**gradient descent**兩種不同方式找最佳值的優缺點。當遇到`normal equation Noninvertibiity`的狀況，通常試著減少**不需要的特徵值，或則是相似的特徵值**。

### 重點
* **hypothesis function，cost function，gradient descent的向量表示**
* **如何進行Feature Normalization**
* **如何判斷Gradient Decent的是否正確在收斂**
* **polynomial regression的運用**
* **normal equation和gradient descent的比較**

### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/1.png)

![2](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/2.png) 

![3](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/3.png) 

![4](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/4.png) 

![5](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/5.png) 
