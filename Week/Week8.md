## Week 8 Unsupervised Learning

### 大綱
`Unsupervised learning`

跟之前學的supervised learning的最大不同就是**unlabed training set，也就是測試資料不會有正確的解答，讓我們可以事先用來訓練機器。**

`K-Means Algorithm`

是一種會將資料分類的方法，其主要步驟如下:

* 1.先隨機初始化K個點**(cluster centroids)**。
* 2. **Cluster assigment**，將所有的trainig set的點找出與其最近的cluster centroids，並賦予對應標號。
* 3. **Move centroid**，利用相同編號的點來更新其cluster centroids的位置。
* 4. 不斷重複2,3直到找到我們所需要的clusters。

`Optimization Objective`

要了解K-Means Algorithm的最佳化目標到底是什麼，簡單來說就是當我們替training set中每個點都加上編號後，**我們最佳化目標就是在同一群的點離其cluster centroid的平均距離應該要是最小的**。

`Random initalization`

根據不同初始化的點，K-Means Algorithm有時只會得到locol optimal，**因此建議當分類的群組如果小於10時，我們選擇多組的隨機初始化K點，進行K-Means Algorithm，找出最佳的分類。**

`Choosing the number of clusters`

可以透過**elbow method(cost j 跟 cluster number k 的關係圖)來觀察**，另外也是可以根據我們的目的來選擇。

****

`Dimensionality Reduction`

這是本週的另外一個重點，為什麼要做Dimensionality Reduction

* 1. 進行資料壓縮，減少電腦記憶體的使用量，加入演算法的運算。
* 2. 視覺化，可以讓我們將資料的分布圖繪製出來。

`Principal Component Analysis PCA`

主要就是用來進行Dimensionality Reduction的一種演算法。**PCA的目標就是找到一個projection line(向量vector)，可以讓所有feature到projection line的垂直距離為最小。** 其步驟如下:

* 1. Preprocess(feature scaling or mean normalization)。
* 2. 計算convariance matrix。
* 3. 計算convariance matrix的eigenvectors。
* 4. 選擇Matrix U的前K個componet，並計算z(feature進行壓縮後的資料)。

當然我們也可以透z,u來反推壓縮前的資料，但只能取回近似值。

`Choosing the number of principle components`

到底要把資料壓縮到怎樣才是適當的(m -> k)。**一般來說，壓縮後的資料跟原始資料的差異在0.01之內，也就取說壓縮後的資料跟原始資料要有0.99以上的相似。**我們可以透過eigenvectors中的marix S來進行推算。

`Advice of applying PCA`

PCA是用來加速運算，並不是用來處理overfitting的問題。建議先不要用壓縮後資料來直接測試所設計好的learning algorithm，除非是發現資料會讓我們learning algorithm執行過慢，這時才需要利用PCA來進行資料壓縮。
 
### 重點

* K-Means Algorithm的實作。
* Principal Component Analysis PCA的實作。

### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week8/1.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week8/2.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week8/3.png)


