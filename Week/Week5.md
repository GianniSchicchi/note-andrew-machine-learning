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

### 作業

#### Neural Network

* Model representation

```octave

input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
                          
load('ex4weights.mat');
nn_params = [Theta1(:) ; Theta2(:)];

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

```

* Feedforward and cost function

```octave
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
m = size(X, 1); % 5000 * 400
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% mapping 0~9 to index [0,1,0...,0]
% y = 5000 * 1 ---->> Y = 10 * 50000
number_of_classes = length(unique(y));
Y = zeros(number_of_classes, m);
for i = 1:m
    Y(y(i), i) = 1;
endfor

p = zeros(size(X, 1), 1); % 5000 * 1

input_layer = [ones(1,m); X']; % 401 * 5000
hidden_layer = [ones(1,m); sigmoid(Theta1 * input_layer)]; % 26 * 5000
output_layer = sigmoid(Theta2 * hidden_layer);  % 10 * 5000
                 
% Do forward propagation
input_layer = [ones(1,m); X']; % 401 * 5000
hidden_layer = [ones(1,m); sigmoid(Theta1 * input_layer)]; % 26 * 5000
output_layer = sigmoid(Theta2 * hidden_layer);  % 10 * 5000

% A3 here is our h0
h0 = output_layer;

% Compute cost function
J = (1/m)*sum(sum(-Y.*log(h0) - (1-Y).*log(1-h0)));
             
% Regularized cost function
regualized_J = (lambda/(2*m))*sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2));

J = J + regualized_J;

```


#### Backpropagation

* Backpropagation

```octave
% 將產生最後的總誤差，根據theta的比例，分散各layer的node上
delta_output_layer = output_layer - Y;
delta_hidden_layer = (Theta2'*delta_output_layer)(2:end, :) .* hidden_layer;

% 再利用這些誤差去調整theta，不斷重複這樣的動作，直到總誤差開始收斂為止
Theta1_grad = (delta_2 * input_layer')/m;
Theta2_grad = (delta_3 * hidden_layer')/m;

% regularize
Theta1_reg_grad = (lambda/m)*Theta1;
Theta2_reg_grad = (lambda/m)*Theta2;

Theta1_grad = Theta1_grad + Theta1_reg_grad;
Theta2_grad = Theta2_grad + Theta2_reg_grad;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

```

* Sigmoid Gradient

```octave
function g = sigmoidGradient(z)

g = zeros(size(z));
g = sigmoid(z).*(1-sigmoid(z));

```


* Initializing Pameters

```octave
function W = randInitializeWeights(L_in, L_out)

W = zeros(L_out, 1 + L_in);
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
```

* Check Gradients

```octave
% 利用Gradient Checking檢查Backpropgation是否正確
function checkNNGradients(lambda)

??

```

* Learning parameters using fmincg

```octave
% 隨機初始化weight
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% also try the MaxIter to a larger
options = optimset('MaxIter', 50);

% also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
                                   
% 將cost function帶入最佳化函數fmincg()，找出最佳的weight
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

```

* Implement Predict

```octave
function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

```

#### Visualizing the hidden layer

  
### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week5/1.png)

![2](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week5/2.png)