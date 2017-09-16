## Week 4 Neural Network: Representation
### 大綱
`Neural Network`可以用來處理當feature很多且複雜的hypothesis function。例如在圖像辨識上，50 X 50 pixel，即 n = 2500，如果想用二次方的假設方程式來滿足這些資料，則feature就會變成O(n^2 / 2) = 2500^2 / 2 = 312500。

`Model Representation`，先了解整個model中最基本的元件`logistic unit`的組成。 logistic unit可以拆成四大部分 - `dendrites`(input)，`axons`(output)，`sigmoid activation function`，`weight`(theta)。 了解完基本元件，接下來要看架構是由層次來分 - `input layer`，`hidden layer`，`ouput layer`。 最後是了解每個層次是如何對映，主要是透過`weight matrix`來進行mapping不同層間的關聯。

再來就是要把架構用`向量化`表示，然後試著用Neural Network來表達OR, AND, XNOR, NOR的運算。

`Multiclass Classification`，Neural Network要如何進行Multiclass Classification。其實還是抱著主要原則 one-vs-all，將output node設定成所需要分類的class數，每個node都是vector。

### 重點

* **了解non-linear hypothesis遇到龐大數量的特徵值的計算限制**
* **了解Neural Network的Model Representation**
* **如何用向量話表示**
* **OR, AND, XNOR, NOR，如何用Neural Network來處理**
* **Multiclass Classification in Neural Network**

### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week4/1.png)