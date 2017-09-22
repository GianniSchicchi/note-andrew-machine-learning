## Week 10 Large Scale Machine Learning

### 大綱

`Learning with Large Datasets`

當資料量很大時，每次進行一次gradient descent都必須進行所有的資料的累加，因此計算量變得非常龐大，收斂速度也會變慢很多。

`Stochastic Gradient Descent`

是將原本gradient descent做些改良，讓gradient descent在處理大量的資料時，還可以保持一定的收列速度。**不用累加所有sample，而是針對每一個的sample，都進行theta的微調**。

* 1. 每次都隨機從dataset中選取一組資料。
* 2. 進行單次的gradient descent的動作。

`Min-Batch Gradient Descent`

相對於傳統Gradient Descent需累加所有sample，或者是Stochastic Gradient Descent是針對單一sample，**Min-Batch Gradient Descent是採用折衷的方式，一次是針對一組(2~100)sample進行Gradient Descent。**

`Stochastic Gradient Descent Converage`

如何判斷Stochastic Gradient Descent是否收斂的最好方式就是畫圖，例如每進行1000次Gradient Descent，就計算其平均cost function，觀察是否有逐漸下降的趨勢。

`Online Learning`

主要是針對某些網站會不斷有user登入，例如訂機票網站，會不斷有不同的user透過其網站訂機票。此時，online learning就是會每當user完成一次訂票過程，就立刻進行一次的學習。

`Map Reducde and Data Paralleism`

就是將計算量平行分散到不同機器進行計算，最後再將各機器的計算結果同一進行處理。

### 重點

* 比較Gradient Descent，Stochastic Gradient Descent，Min-Batch Gradient Descent的差異。
* 了解Online Learning跟Map Reducde and Data Paralleism。

### Concept Graph

