%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix

predicted = zeros(1, size(ytest, 2));
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    [max_prob, max_index] = max(P);
    predicted(i:i+99) = max_index;
end

C = confusionmat(ytest, predicted);
confusionchart(C);