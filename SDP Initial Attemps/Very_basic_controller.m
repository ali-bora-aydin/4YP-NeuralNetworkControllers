%% Define system parameters
m = 3;
k = 3;
b = 6;
% Define mass- spring damper system
A = [ 0, 1; -k/m, -b/m];
B = [0 ; 1/m];
C = [1 0];
D = 0;

% LQR control parameters: State feedback controller
Q = 0.5 * [1 0; 0 0];
R = 1/16;

% Solve optimal control problem
[K, ~, ~] = lqr(A,B,Q,R);

%% Generate data to train a Neural Network with this
x1 = -10:0.1:10;  % Define the range of x1 values
x2 = -10:0.1:10;  % Define the range of x2 values

% Initialize an empty matrix to store the permutations
X = zeros(numel(x1) * numel(x2), 2); 

% Loop through each value of x1 and x2 to create permutations
for i = 1:length(x1)
    for j = 1:length(x2)
        X( (i-1)*numel(x1)+j, :) = [x1(i), x2(j)];
    end
end

y = K * X'; % results in a row vector

% Split the data into train - validation - test sets (70% - 15% - 15%)
rng(42)
[Xtrain, Xval, Xtest, ytrain, yval, ytest] = trainValTestSplit(X, y);





%% 2 hidden layer neural network (as in most papers)
numFeatures = size(X,2);
layers = [ ...
    sequenceInputLayer(numFeatures)
    fullyConnectedLayer(32, BiasInitializer = "zeros", BiasLearnRateFactor = 0)
    reluLayer
    fullyConnectedLayer(32, BiasInitializer = "zeros", BiasLearnRateFactor = 0)
    reluLayer
    fullyConnectedLayer(1, BiasInitializer = "zeros", BiasLearnRateFactor = 0)
    regressionLayer];

numEpochs = 1000;
options = trainingOptions('adam', ...
    MaxEpochs = numEpochs, ...
    MiniBatchSize = 400, ...
    Verbose= false, ...
    VerboseFrequency= numEpochs*50, ...
    ValidationData={Xval', yval}, ...
    ValidationPatience = 20, ...
    InitialLearnRate = 1e-2,...
    LearnRateSchedule = "piecewise", ...
    LearnRateDropPeriod = numEpochs/2, ...
    LearnRateDropFactor = 0.1, ...
    ExecutionEnvironment = "gpu", ...
    Plots='none');

[net, info] = trainNetwork(Xtrain', ytrain, layers, options);

W1 = net.Layers(2).Weights;
W2 = net.Layers(4).Weights;
W3 = net.Layers(6).Weights;
b1 = net.Layers(2).Bias;
b2 = net.Layers(4).Bias;
b3 = net.Layers(6).Bias;


% Predictions
X = [0;0];

y_pred =   W3 * reluu(W2 * reluu(W1*Xval' + b1) + b2) +b3;
predMSE = mse(y_pred, yval);

x = [1,1];
info.FinalValidationLoss
predict(net, [1;1])
result = W3 * reluu(W2 * reluu(W1*x' + b1) + b2) + b3






function [Xtrain, Xval, Xtest, ytrain, yval, ytest] = trainValTestSplit(X, ...
    y, trainRatio, valRatio, testRatio)

arguments
    X
    y
    trainRatio = 0.7;
    valRatio = 0.15;
    testRatio = 0.15;
end

[trainInd,valInd,testInd] = dividerand(numel(y),trainRatio,valRatio,testRatio);

Xtrain = X(trainInd, :);
Xval = X(valInd, :);
Xtest = X(testInd, :);
ytrain = y(trainInd);
yval = y(valInd);
ytest = y(testInd);

end
