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


### 作業

#### Anomaly Detection

* Load Example Dataset

```octave
%  The following command loads the dataset. You should now have the variables X, Xval, yval in your environment
load('ex8data1.mat');

```

* Estimate the dataset statistics

```octave
%  Estimate mu and sigma2
[mu sigma2] = estimateGaussian(X);
%  Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2);

===========================================

% This function estimates the parameters of a Gaussian distribution using the data in X
function [mu sigma2] = estimateGaussian(X)

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

mu = mean(X);
sigma2 = var(X,1);

===========================================

% Computes the probability density function of the multivariate gaussian distribution.
function p = multivariateGaussian(X, mu, Sigma2)

k = length(mu);

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
    Sigma2 = diag(Sigma2);
end

X = bsxfun(@minus, X, mu(:)');
p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

end
```

* Find Outliers

```octave
pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon F1] = selectThreshold(yval, pval);

===========================================

% Find the best threshold (epsilon) to use for selecting outliers
function [bestEpsilon bestF1] = selectThreshold(yval, pval)

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;

for epsilon = min(pval):stepsize:max(pval)
	% 若為1，則表示模型判斷異常， 若為0，則表示模型判斷為正常
	predictions = (pval < epsilon);
	
	tp = sum((predictions == 1) & (yval == 1)); % 判斷正確
	fp = sum((predictions == 1) & (yval == 0)); % 判斷錯誤
	tn = sum((predictions == 0) & (yval == 0)); % 判斷正確
	fn = sum((predictions == 0) & (yval == 1)); % 判斷錯誤
	
	recall = tp / (tp + fn); % 所有yval == 1時，判斷正確的比例
	precidion = tp / (tp + fp); % 所有predictions == 1時，判斷正確的比例

	F1 = 2 * precidion * recall / (precidion + recall);
	
	if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
    
end

```

#### Collaborative Filtering

* Loading movie ratings dataset

```octave
%  Load data
load ('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

```

* Collaborative Filtering

```octave
%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 0);
               
===========================================

% Collaborative filtering cost function
function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);
                
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% 計算cost function
predit = X*Theta';
Cost = (predit-Y).^2;
J = (1/2)*sum(sum(Cost.*R));

% 計算Gradient
X_grad = ((predit-Y).*R)*Theta;
Theta_grad = ((predit-Y).*R)'*X;

% 計算Regularization cost function
Reg_J = (lambda/2)*sum(sum(Theta.^2))) + ((lambda/2)*sum(sum(X.^2));

% 計算Regularization Gradient
Reg_X_grad = lambda*X;
Reg_Theta_grad = lambda*Theta;

% 最終Cost和Gradient
J = J + Reg_J;
X_grad = X_grad + Reg_X_grad;
Theta_grad = Theta_grad + Reg_Theta_grad;
grad = [X_grad(:); Theta_grad(:)];

```

* Learning Movie Ratings

```octave
%  Load data
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg 
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
% 利用optimset，找出最佳的X & Theta
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W 
% 將算出來的結果，拆成X & Theta 兩塊
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

```

* Recommendation for you

```octave
p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');

for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

```


### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/1.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/2.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/3.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/4.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/5.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/6.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week9/7.png)


