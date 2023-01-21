load lenet.mat
images = zeros(28*28, 5);
ims = zeros(28, 28, 5);

im = rgb2gray(im2double(imread('../images/Samples/sample_1.png')))';
images(:, 1) = reshape(im, [], 1);
ims(:, :, 1) = im';

im = rgb2gray(im2double(imread('../images/Samples/sample_2.png')))';
images(:, 2) = reshape(im, [], 1);
ims(:, :, 2) = im';

im = rgb2gray(im2double(imread('../images/Samples/sample_4.png')))';
images(:, 3) = reshape(im, [], 1);
ims(:, :, 3) = im';

im = rgb2gray(im2double(imread('../images/Samples/sample_8.png')))';
images(:, 4) = reshape(im, [], 1);
ims(:, :, 4) = im';

im = rgb2gray(im2double(imread('../images/Samples/sample_9.png')))';
images(:, 5) = reshape(im, [], 1);
ims(:, :, 5) = im';

layers = get_lenet();
layers{1}.batch_size = 1;
predicted = zeros(1, 5);

figure();
for i = 1:5
    [output, P] = convnet_forward(params, layers, images(:, i));
    [max_prob, max_index] = max(P);
    subplot(1, 5, i);
%     imshow(reshape(images(:, i), 28, 28));
    imshow(ims(:, :, i));
    xlabel(max_index - 1);
%     disp(max_index - 1);

    
end