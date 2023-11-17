% Define the unstable problem

% Simply a sign change compared to the past problem

A = [0 1; 1 0];
B = [0 ; 1]; 
C = [1 0];
D = 0;
%A = [0 1; -1 0 ];
Q = 0.5 * [1 0; 0 0];
R = 1/16;
[K, S, P] = lqr(A,B,Q,R);

sys1 = ss(A-B*K, B, C, D);
step(sys1)



%% Training Data Generation

% We need to collect different combinations of state vectors (x1,x2) 
% in a n by 2 X matrix.
% The output of the controller to these state vectors x will give
% the n by 1 y vector.

% Note: this is not a very smart way: I don't know yet what kind of
% sampling would work best. In this case, it is not even that necessary
% since the relationship is known to be linear. Could be useful when there
% is uncertainty/ robustness requirements? I don't know

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

%% Neural Network 
numHiddenLayers = 2;
neuronsPerLayer = 1;
numEpochs = 1000;
relu = true;
plotsFlag = true; % to disable the app from launching


net = createFeedForwardNet(neuronsPerLayer = neuronsPerLayer);

trainedNet = train(net, Xtrain', ytrain);

%Evaluate train loss
ytrainPred = sim(trainedNet, Xtrain')'; % column vector for all val datapoints
trainMSE = mse(ytrainPred', ytrain);

% Evaluate validation loss
yvalPred = sim(trainedNet, Xval')'; % column vector for all val datapoints
valMSE = mse(yvalPred', yval);





gensim(trainedNet)







%% function to create neural network

function  net = createFeedForwardNet(opts)

arguments
    opts.numHiddenLayers = 2;
    opts.neuronsPerLayer = 10; 
    opts.numEpochs = 1000;
    opts.reluFlag = true;
    opts.plotsFlag = true
    % Should I train with SGDM instead and define learning rate etc.?
end


% Define the network architecture here
netSize = zeros(1, opts.numHiddenLayers);
netSize(:) = opts.neuronsPerLayer;
net = feedforwardnet(netSize);

% Replace tanh activations with ReLU
if opts.reluFlag
    net.layers{1: end -1}.transferFcn = "poslin";
end
%else don't do anything and activations will be tanh

% Try to disable data split by the train function as I want to do this outside
net.divideParam.trainRatio = 1;
net.divideParam.valRatio   = 0;
net.divideParam.testRatio  = 0;

% If I ever need to change numEpochs
net.trainParam.epochs = opts.numEpochs;
% to disable training window
net.trainParam.showWindow = opts.plotsFlag;

end

% Function that creates train, test, val X and y matrices

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
