## Week 3 Logistic Regression
### 大綱
前兩週主要都在處理regression problem，這週開始利用`logistic regression`的方式來處理classifcation problem。

`Binary classification`，表示我的output y 只有是1跟0兩種可能，因此會利用`sigmoid function`或又稱`logistic function`來將我們的**hypothesis function的output侷限在0~1之間**。例如，當output = 0.7時，表示y = 1的機率是0.7，換句話說y = 0的機率是0.3。

`Decision Boundary`，既然會進行分類，那就會有邊界的產生**(將y=1,y=0用邊界分開)**，那必須知道logistic function的decision boundary是要如何計算。

`Cost Function`，接下來要學會logisitic regression的cost function是如何定義，主要是先拆成兩塊(y=1,y=0)分開討論cost，最後再將這兩塊的cost加總後就是cost function。至於`Gradient Decent`很巧地剛好跟linear regression一樣(可以透過數學證明)。

`Advanced Optimization`，一些比Gradient Decent更好的優化演算法(更有效率找到最佳theta值)，要學會如何使用這樣現成函式庫的演算法**(Octave's fminic())**，主要是將**cost function跟cost function的篇導數帶入函式**。

`MultiClass Classification`，既然學會了二分類，當然要學著進階的多個分類，主要原則就是**進行多次的二分類(one-vs-all)**，找出機率最高的哪組。

`Regulation`，這週的最後主題，Regulation最主要的目的就是**降低overfitting問題**。必須先瞭解怎樣是`overfitting - high variance`，`underfiiting - high bias`。接下來重點就是將先前學到的linear regression和logist regression中的cost function，gradient descent，normal equlation進行regularize。

### 重點

* **了解sigmoid function背後含義**
* **了解hypothesis function的output的意義**
* **怎樣計算Decision Boundary**
* **了解Logistic Regression的cost function是如何定義**
* **如何使用Advanced Optimization function**
* **怎樣利用二分法的原則進行多分類**
* **了解regulation的含義，以及將個公式進行regularize**

### 作業

#### Logistic Regression

* 資料初始化

```octave
data = load('ex2data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

[m, n] = size(X); %[100, 2]
X = [ones(m, 1) X];  %[100, 3]
initial_theta = zeros(n + 1, 1);  %[3, 1]
```

* 1. sigmoid function

```octave
function g = sigmoid(z)

g = zeros(size(z));
g = ones(size(z)) ./ (1.0 + exp(-z));

```

* 2. cost function and gradient

```octave
function [J, grad] = costFunction(theta, X, y)

m = length(y);
J = 0;
grad = zeros(size(theta));

% 計算hypothesis function
h = sigmoid(X*theta);

% 計算cost function
J = (1/m)*sum(-y.*log(h)-(1-y).*log(1-h));

% 計算gradient
grad = (1/m)*(X'*(h-y));

```

* 3. Learning parameters using fminunc

```octave
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

```

* 4. Evaluating logistic regression

```octave
function p = predict(theta, X)

m = size(X, 1);
p = zeros(m, 1);

% set p to a vector of 0's and 1's
p = round(sigmoid(X*theta));

```

#### Regularized Logistic Regression

* 1. feature mapping

```octave

function out = mapFeature(X1, X2)

% 產生更複雜hypothes function，讓我們有機會更佳fit feature
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

```

* 2. cost function and gradient

```octave
% lambda 是regularization parameter，用來控制overfitting or underfitting
function [J, grad] = costFunctionReg(theta, X, y, lambda)

[J, grad] = costFunction(theta, X, y);
regularization_J = (lambda/(2*m)) * sum(theta(2:end).^2);
J = J + regularization_J;

regularization_grade = (lambda/m) * theta(2:end);
grad(2:end) = grad(2:end) + regularization_grade;

```

* 3. Learning parameters using fminunc

```octave
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 100;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

```


### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/1.png)

![2](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/2.png) 

![3](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/3.png) 

![4](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/4.png) 

![5](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/5.png) 

![6](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/6.png)

![7](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/7.png) 

![8](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/8.png) 

![9](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/9.png) 

![10](https://github.com/htaiwan/note-andrew-machine-learning/blob/4db4a018b65f57a950315cf948240c194c93d893/Concept%20Graph/Week3/10.png) 