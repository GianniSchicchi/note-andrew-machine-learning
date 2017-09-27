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

### 作業

#### K-Means Clustering

* Find Closest Centroids

```octave
% Select an initial set of centroids
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using the initial_centroids
% idx = m x 1 vector of centroid assignments
idx = findClosestCentroids(X, initial_centroids); 

============================

% computes the centroid memberships for every example
function idx = findClosestCentroids(X, centroids)

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

for xi = 1:size(X,1) % for-loop所有的點
		x = X(xi,:);
		%找出x的closest centroid
	    best = Inf;
	    for ui = 1:K  %分成3個group
			  u = centroids(ui,:);
			  distance = dot(x-u, x-u);
		      if distance < best
			       best = distance;
			       idx(xi) = ui; % 替每個點給予分類標籤
		      end
	    end
end

```

* Compute Means

```octave
%  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K);

============================

% returns the new centroids by computing the means of the data points assigned to each centroid.
function centroids = computeCentroids(X, idx, K)

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

for ui = 1:K
	centroids(ui,:) = sum(X(find(idx == ui),:)) / sum(idx == ui);
end

```

* K-Means Clustering

```octave
% Settings for running K-Means
K = 3;
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);

[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);

============================

% This function initializes K centroids that are to be used in K-Means on the dataset X
function centroids = kMeansInitCentroids(X, K)

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% randomly redorder the indices of examples
randidx = randperm(size(X,1));
% take the first k examples as centroids
centroids = X(randidx(1:K),:);

============================

% runs the K-Means algorithm on data matrix X, where each row of X is a single example
function [centroids, idx] = runkMeans(X, initial_centroids, ...
                                      max_iters, plot_progress)
                                      
% Initialize values
[m n] = size(X);
K = size(initial_centroids, 1);
centroids = initial_centroids;
previous_centroids = centroids;
idx = zeros(m, 1);

% Run K-Means
for i=1:max_iters
	 % For each example in X, assign it to the closest centroid
    idx = findClosestCentroids(X, centroids);
    
    % Given the memberships, compute new centroids
    centroids = computeCentroids(X, idx, K);
end
```

#### Principle Component Analysis 

* Load Example Dataset

```octave
%  The following command loads the dataset. You should now have the variable X in your environment
load ('ex7data1.mat');
```

* Principal Component Analysis

```octave
%  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

============================

% Normalizes the features in X 
function [X_norm, mu, sigma] = featureNormalize(X)

mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

============================

% Run principal component analysis on the dataset X
function [U, S] = pca(X)

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

sigma = (1/m) * X' * X;
[U, S, V] = svd(sigma);

```

* Dimension Reduction

```octave
%  Project the data onto K = 1 dimension
K = 1;
Z = projectData(X_norm, U, K);
X_rec  = recoverData(Z, U, K);

============================

% Computes the reduced data representation when projecting only on to the top k eigenvectors
function Z = projectData(X, U, K)

Z = zeros(size(X, 1), K);

% take the first k directions
U_reduce = U(:, 1:K);
% computes the projected data points
Z = X * U_reduce;

============================

% Recovers an approximation of the original data when using the projected data
function X_rec = recoverData(Z, U, K)

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

U_reduce = U(:, 1:K);
X_rec = Z * U_reduce';

```

### Concept Graph

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week8/1.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week8/2.png)

![1](https://github.com/htaiwan/note-andrew-machine-learning/blob/master/Concept%20Graph/Week8/3.png)


