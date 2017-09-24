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

### 作業

#### Linear regression with one variable

* 1. 參數的初始化


```octave
data = load('ex1data1.txt');
y = data(:, 2);
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;

```

* 2. 計算cost function

```octave
function J = computeCost(X, y, theta)
m = length(y); % number of training examples
J = 0;

% hypothesis function
hypothesis = X*theta;

% cost function
J = 1/(2*m)*sum((hypothesis-y).^2);

```

* 3. 計算gradient descent

```octave
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

	theta_prev = theta;
    featues = size(X, 2);
    
    for feature = 1:featues
        % 這其實就是J針對theta進行偏微分
        deriv = (X * theta_prev - y)' * X(:, feature) / m;
        % 更新theta，進行收斂
        theta(feature) = theta_prev(feature) - (alpha * deriv);
    end
    
    % 每次更新theta就計算對應的cost值，記錄下來，之後進行plot
    % 用來驗證gradientDescent是否有正確在收斂。
    J_history(iter) = computeCost(X, y, theta);

end

```

#### Linear regression with multiple variable

* 1. 參數的初始化

```octave
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

```


* 1. Feature Normalization

```octave
function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% 計算所有feature的平均值
mu = mean(X)
% 計算標準差standard deviation
sigma = std(X)

% 針對每個feature進行normalization
index = 1:size(X, 2);

for i = index,
	% 公式：xi:= (xi − μi) / si
	XminusMu  = X(:, i) - mu(i);
	X_norm(:, i) = XminusMu / sigma(i);
end

```

* 2. 計算cost function

```octave
function J = computeCostMulti(X, y, theta)
m = length(y);
J = 0;
J = computeCost(X, y, theta);

```

* 3. 計算gradient descent

```octave
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
	   % 向量化版本：J針對theta進行偏微分
	   deriv = 1/(2*m) * (X*theta - y)' * (X*theta - y);
	   theta = theta - alpha * deriv;
	   
	   J_history(iter) = computeCost(X, y, theta);
end

```

* 4. 計算normal equation

```octave
function [theta] = normalEqn(X, y)
theta = zeros(size(X, 2), 1);
theta = pinv(X'*X)*X'*y;

```


### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/1.png)

![2](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/2.png) 

![3](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/3.png) 

![4](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/4.png) 

![5](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week2/5.png) 
