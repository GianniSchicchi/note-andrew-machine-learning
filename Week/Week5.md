## Week 5 Neural Network: Learning
### 大綱
我覺得這週課程是這系列最難的一週，公式複雜，要理解的觀念也複雜。

首先，我們還是要先提到**Cost Function的計算**，在Neural Network中的Cost Function其實跟logistic regression的cost function很像，因為他們是用相同hypothesis function - **sigmoid function**，差別是Neural network有layer觀念，所以在公式上會變得更複雜一點，**需要多一層累加layer的動作**。

`Backpropgation Algorithm`，主要就是用來找出cost function的最小值。主要的觀念是利用**將產生最後的總誤差，根據theta的比例，分散各layer的node上，再利用這些誤差去調整theta，不斷重複這樣的動作，直到總誤差開始收斂為止**。細節的實作，真的是需要自己下去玩過一遍才會有感覺。

`Unrolling Patameters`，這個是實作上要特別注意的地方，其實就是**把不同陣列合併成一個陣列**，當然也要懂得還原回去，我想這是**因為octave的fminic()的函數限制，只能傳一個大陣列進去**。

`Gradient Checking`，**是用來驗證Backpropgation Algorithm的實作是否正確，兩邊算出來的值之間的誤差需小於0.0001**，因為Gradient Checking的計算量是非常龐大，不可能在Backpropgation中的每一次的收斂都進行這樣的驗證，因此**只要做一次，確認誤差是OK就可以了**。

`Random Initialization`，將weight(theta)全補初始化為零，當在執行Backpropgation就沒意義了，因為**找不出誤差的分配比例**，所以要隨機初始化這些weight，雖然說是隨機，但也是有一定方法跟原則來進行隨機。

最後，要把這東西串接再一起。

#### 1. 怎樣決定network架構
* **input units的數量，是根據feature決定**
* **output units的數量，是根據需求的分類量決定**
* **hidden units的數量，一般是越多越好，但越多計算量就越龐大**
* **hidden layer的數量，default是一層，若大於一層，最好是每一層的unit數量可以保持一致**

#### 2. 如何訓練network
* **隨機初始化weight**
* **進行forward propagation取得h(x)**
* **計算cost function**
* **實作Backpropgation，取得cost function的偏導數**
* **利用Gradient Checking檢查Backpropgation是否正確，若OK，記得關閉此步驟**
* **最後將cost function跟cost function的偏導數帶入gradient descent或者其他最佳化函數fminic()，找出最佳的weight**

### 重點

* **了解cost function的定義**
* **了解Backpropgation的意義跟計算方式**
* **Unrolling，了解matrix的合併跟還原**
* **如何進行Gradient Checking**
* **如何將weight進行random Initialization**
  
### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week5/1.png)

![2](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week5/2.png)