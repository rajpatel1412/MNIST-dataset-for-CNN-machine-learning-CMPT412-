layers = get_lenet();
load lenet.mat
% load data
% Change the following value to true to load the entire dataset.
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);
xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
batch_size = 64;
 
 
layers{1}.batch_size = 1;
img = xtest(:, 1);
img = reshape(img, 28, 28);
imshow(img')
 
%[cp, ~, output] = conv_net_output(params, layers, xtest(:, 1), ytest(:, 1));
output = convnet_forward(params, layers, xtest(:, 1));
output_1 = reshape(output{1}.data, 28, 28);
% Fill in your code here to plot the features.

feature_maps = reshape(output{1, 2}.data, output{1, 2}.height, output{1, 2}.width, output{1, 2}.channel);
figure();
title("Conv_layer");
for i = 1:4
    for j = 1:5
        subplot(4, 5, (j + ((i - 1)*5)));
        imshow(feature_maps(:, :, (j + ((i - 1)*5)))')
    end
end

feature_maps = reshape(output{1, 3}.data, output{1, 3}.height, output{1, 3}.width, output{1, 3}.channel);
feature_maps(feature_maps < 0) = 0;
feature_maps(feature_maps > 0) = 1;
feature_maps = imcomplement(feature_maps);
figure();
title("Relu layer");
for i = 1:4
    for j = 1:5
        subplot(4, 5, (j + ((i - 1)*5)));
        imshow(feature_maps(:, :, (j + ((i - 1)*5)))')
    end
end