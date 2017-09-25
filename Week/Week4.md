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

### 作業

#### Logistic Regression

* 資料初始化

```octave
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)
                          

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;

```

* Verctorizing the Logistic Regression

```octave
function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);

% Verctorizing cost function 
J = (1/m)*sum(-y.*log(h) - (1-y).*log(1-h));

% Verctorizing the gradient
grad = (1/m)*(X'*(h-y));

% Verctorizing the regualized item
regualized_J = (lambda/(2*m))*sum(theta(2:end).^2);
regualized_G = (lambda/m)*theta(2:end);

% Verctorizing regualized Logistic Regression
J = J + regualized_J;
grad(2:end) = grad(2:end) + regualized_G;

```

* One-vs-all Classification


```octave
function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);  % row 5000
n = size(X, 2);  % column 400

all_theta = zeros(num_labels, n + 1);  % 10 * 401 這就是1~10不同的classifer
X = [ones(m, 1) X]; % 5000 * 401

% Set Initial theta
initial_theta = zeros(n + 1, 1);  % 401 * 1

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Run fmincg to obtain the optimal theta
for c = 1:num_labels
	all_theta(c, :) = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                zeros(n + 1, 1), options);
end

```

* One-vs-all Predition


```octave
function p = predictOneVsAll(all_theta, X)

m = size(X, 1); % 5000
num_labels = size(all_theta, 1); % 10 * 401

p = zeros(size(X, 1), 1); % 5000 * 1
X = [ones(m, 1) X]; % 5000 * 401

% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

[value, p] = max((X*all_theta'), [], 2) % 5000 * 2

```

#### Neural Network

* 資料初始化

```octave
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% Load the weights into variables Theta1 and Theta2
load('ex3weights.mat')

```

* Feedforward Propagation and Prediction

```octave
function p = predict(Theta1, Theta2, X)
m = size(X, 1); % 5000
num_labels = size(Theta2, 1); % 10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 5000 * 1

input_layer = [ones(1,m); X']; % 401 * 5000
hidden_layer = [ones(1,m); sigmoid(Theta1 * input_layer)]; % 26 * 5000
output_layer = sigmoid(Theta2 * hidden_layer);  % 10 * 5000

[value, p] = max(output_layer', [], 2);  % 5000 * 2
```

### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week4/1.png)