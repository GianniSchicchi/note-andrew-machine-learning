## Week 6 Advice for Applying Machine Learning
### 大綱
本週的重點是 - `Debugging a learning algorithm`，當發現在用我們建立的hypothesis去測試新的數據時有很大的誤差產生，我們要如何去診斷我們的learning algorithm是哪裡發生問題，該如何改進。

`Evaluating a hypothesis`，我們必須有一個客觀的數據可以評估hypothesis function的好壞，會先將數據分成兩堆**training set(70%) 和 test test(30%)**，利用traing set來找出hypothesie function最佳化theta，再利用test test來計算**regressuib和classification的hypothesis error**。

`Model Selection`，如何讓model generalize，會先將數據分成三堆**training set(60%) Cross vaildation set(20%) test test(20%)**，利用training set來找出hypothesie function最佳化theta，再利用Cross vaildation set，找出polynomial的最佳degreee，最後利用test test來計算hypothesis error。
 
再來的重點是，利用下列兩個圖形來理解`Bias跟Variance`關係:

**1. X軸: d(polynome degree) Y軸:cost。**

* high bias(underfitting)，d小，traing和vaildation cost都大且相近。
* high variance(overfitting)，d大，traing cost下降，但vaildation cost上升。

**2. X軸: lamba(regularization) Y軸:cost。**

* high variance(overfitting)，lamba小，traing cost小，但vaildation cost大。
* high bias(underfitting)，lamba大，traing和vaildation cost都上升且相近。

`Model & lamba coombo Selection`，把不同degree的model跟不同大小的lamba做所有的排列組合，把個組合都利用training set來找出每組hypothesie function最佳化theta，**再利用Cross vaildation set來計算error，找出最佳的hypothesie function(lamba = 0來進行測試)**，最後才利用test test來驗證hypothesis error。

`learning curves`，如何從learning curves去判斷high bias or high variacne。

**X軸: error Y軸:traing set size**

high bias跟high variance的各種狀況。

`典型的診斷手法`

* 增加trainig example: fix high variance
* 降低feature數量: fix high variance
* 增加feature數量: fix high bias
* 增加polynomial degree: fix high bias
* 降低lamba: fix high bias

`錯誤分析`

* 快速設計，快速實作，快速驗證
* 畫出learning curve，找出我們需要那種診斷手法
* 錯誤分析，分析檢差錯誤example的原因

`Skewed classes`，是指這個class在整個data set是佔很少數。利用`percision,recall,F score`，來評估hypothesis performance。


### 重點
* **如何透過資料分群，來驗證hypothesis, model, lamba**
* **如何從各種不同關係圖觀察出是high bias或variance的問題**
* **如何根據問題，下診斷手法**
* **錯誤分析的流程**
* **對於Skewed classes的評估**

### 作業

#### Regularized Linear Regression and Bias-Variance

* Initialization

```octave
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');
```

* Regularized Linear Regression Cost & Gradient

```octave
theta = [1 ; 1];
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

======================

% Compute cost and gradient for regularized linear regression with multiple variables
function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h = X*theta;

J = (1/(2*m))*sum((h-y).^2);
Reg_J = (lambda/(2*m))*sum(theta(2:end).^2);
J = J + Reg_J;

grad = (1/m)*(X'*(h-y));
Reg_grad = [0;(lambda/m).*theta(2:end)];
grad = grad + Reg_grad;
```


* Train Linear Regression

```octave
%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

======================

% Trains linear regression given a dataset (X, y) and a regularization parameter lambda
function [theta] = trainLinearReg(X, y, lambda)

% Initialize Theta
initial_theta = zeros(size(X, 2), 1); 

% Create "short hand" for the cost function to be minimized
costFunction = @(t) linearRegCostFunction(X, y, t, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('MaxIter', 200, 'GradObj', 'on');

% Minimize using fmincg
theta = fmincg(costFunction, initial_theta, options);

```

* Learning Curve for Linear Regression

```octave
lambda = 0;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

======================
                  
% Generates the train and cross validation set errors needed to plot a learning curve
function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
    
% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
	X_train = X(1:i,:);
	Y_train = y(1:i);
	
	% 從training set取得最佳化theta
	theta = trainLinearReg(X_train, Y_train, lambda);
	
	% set lamba = 0, get first item that is training error & validation error
	error_train(i) = linearRegCostFunction(X_train, Y_train, theta, 0)(1);
	error_val(i) = linearRegCostFunction(Xval, yval, theta, 0)(1);
end
```

* Feature Mapping for Polynomial Regression

```octave
p = 8;
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

======================

% Maps X (1D vector) into the p-th power
function [X_poly] = polyFeatures(X, p)

X_poly = zeros(numel(X), p);

for i = 1:numel(X)  %取得每列
	for j = 1:p
		X_poly(i,j) = X(i).^j;
	end
end

======================

% Normalizes the features in X 
function [X_norm, mu, sigma] = featureNormalize(X)

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

```

* Learning Curve for Polynomial Regression

```octave

lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));


figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

```

* Validation for Selecting Lambda

```octave
[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);
    
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
    
======================

% Generate the train and validation errors needed to plot a validation curve that we can use to select lambda
function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
    
% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	% 利用不同lamba, 算出不同的theta
    theta = trainLinearReg(X, y, lambda);
    error_train(i) = linearRegCostFunction(X, y, theta, 0)(1);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0)(1);
end


```

### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week6/1.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week6/2.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week6/3.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week6/4.png)

