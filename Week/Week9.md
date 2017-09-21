## Week 9 Anomaly Detection

### 大綱

> Anomaly Detection
> 
> 當給一堆資料時，如何檢測出異常的資料?

`Gaussian Distribution`，是一個鐘型的分佈圖，主要是根據資料的**平均值跟標準差來進行計算。**

`Algorithm`:

* 1. 選定某個feature(這個feature是可以反映出異常的sample)。
* 2. 計算平均值跟標準差。
* 3. 計算p(x) Gaussian Distribution function。
* 4. 若p(x)小於某個threshold，則判定為異常。

`Algorithm evaluation`:

先將正常的資料按照60/20/20比例分配到training/CV/test，然後將異常的資料按照50/50比例分配到CV/test。

* 1. 利用training set來訓練p(x)。
* 2. 將訓練好的P(x)，利用CV來進行異常判斷。
* 3. 計算Precision/recall/F1 score來進行演算法評估。

`Anomaly Detection VS. Supervised Learning`:

當postive和negative data數量比例差異很大，會使用Anomaly Detection。若差異不大，則會使用Supervised Learning。

`Choosing What Features to use`:

先確認此feature是否是Gaussian Distribution， 可以透過圖形繪製觀察是否是鐘型的分佈。若非是鐘型分佈，我們可以透過transofrm的方式，看是否會變成鐘型分佈。

`Multivariate Gaussian Distribution`:

當某個data的所有feature分別獨立來看時，Gaussian Distribution都不會判定是異常，但如果綜合來看這些feature，我們會判定異常，此時就是要利用Multivariate Gaussian Distribution。

> Recommender Systems
> 
> 如何推薦使用者感興趣的電影?

`Content Based Recommendation`

當知道電影的feature vector(ex. action的比例 0~1)，也知道使用者對某些電影的分數(y vector)，則可以透過linear regression，推測使用者對電影的偏好(theta vector)，進而推薦相似的電影給使用者(y值相近的電影)。

`Collaborative Filtering Algorithm`

但是很難實際定義出一部電影的feature vector，一開始先讓使用者可以直接告訴其對電影的偏好，再來反推出電影的feature vector。

* 1. 當已知使用者的電影的偏好，可推測出電影的feature vector。
* 2. 當已知電影的feature vector，可推測出使用者的電影的偏好。

透過上述步驟，不斷的重複執行收斂出最佳的theta跟feature vector，進而推薦相似的電影給使用者(y值相近的電影)。

`Mean Normalization`

如果有一個新使用者，都還沒看過任何電影，也當然沒給過任何電影分數，這時有該如何推薦電影給使用者呢？

就是使用其他使用者已給的電影的分數做平均值(u vector)，然後將這個平均值假設是新使用者會給的評價，其實就是推薦目前熱門的電影給任何尚未平價的使用者。


### 重點

* Anomaly Detection的實作。
* Recommender Systems的實作。

### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/1.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/2.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/3.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/4.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/5.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/6.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/7.png)


