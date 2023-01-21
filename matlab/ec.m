load lenet.mat;
layers = get_lenet();
layers{1}.batch_size = 1;

im1 = rgb2gray(imread('../images/image1.JPG'));
% imshow(im1);
level1 = graythresh(im1);
im1comp = imcomplement(im1);
im1bin = imbinarize(im1comp, level1);
% CC1 = bwconncomp(im1bin);
% L1 = bwlabel(im1bin);
stats1 = regionprops(im1bin, "BoundingBox");
num_bounding_boxes = size(stats1, 1);
predictions = zeros(1, num_bounding_boxes);
figure();

for i = 1:num_bounding_boxes
    box = imcrop(im1bin, stats1(i).BoundingBox);
    if size(box, 1) < 28 || size(box, 2) < 28
        continue;
    end
    input = imresize(box, [28 28]);
    input = input';
    input = reshape(input, [], 1);
    
    [output, P] = convnet_forward(params, layers, input);
    [max_prob, max_index] = max(P);
    max_index = max_index - 1;
    predictions(i) = max_index;
    subplot(1,num_bounding_boxes, i)
    imshow(box);
    xlabel(max_index);
end
% disp(predictions);

im2 = rgb2gray(imread('../images/image2.JPG'));
level2 = graythresh(im2);
im2comp = imcomplement(im2);
im2bin = imbinarize(im2comp, level2);
% CC1 = bwconncomp(im1bin);
% L1 = bwlabel(im1bin);
stats2 = regionprops(im2bin, "BoundingBox");
num_bounding_boxes2 = size(stats2, 1);
predictions2 = zeros(1, num_bounding_boxes2);
figure();

for i = 1:num_bounding_boxes2
    box2 = imcrop(im2bin, stats2(i).BoundingBox);
    if size(box2, 1) < 28 || size(box2, 2) < 28
        continue;
    end
    input2 = imresize(box2, [28 28]);
    input2 = input2';
    input2 = reshape(input2, [], 1);
    
    [output, P] = convnet_forward(params, layers, input2);
    [max_prob, max_index2] = max(P);
    max_index2 = max_index2 - 1;
    predictions2(i) = max_index2;
    subplot(1,num_bounding_boxes2, i)
    imshow(box2);
    xlabel(max_index2);
end
% disp(predictions);

im3 = rgb2gray(imread('../images/image3.png'));
level3 = graythresh(im3);
im3comp = imcomplement(im3);
im3bin = imbinarize(im3comp, level3);
% CC1 = bwconncomp(im1bin);
% L1 = bwlabel(im1bin);
stats3 = regionprops(im3bin, "BoundingBox");
num_bounding_boxes = size(stats3, 1);
predictions = zeros(1, num_bounding_boxes);
figure();

for i = 1:num_bounding_boxes
    box = imcrop(im3bin, stats3(i).BoundingBox);
    if size(box, 1) < 28 || size(box, 2) < 28
        continue;
    end
    input = imresize(box, [28 28]);
    input = input';
    input = reshape(input, [], 1);
    
    [output, P] = convnet_forward(params, layers, input);
    [max_prob, max_index] = max(P);
    max_index = max_index - 1;
    predictions(i) = max_index;
    subplot(1,num_bounding_boxes, i)
    imshow(box);
    xlabel(max_index);
end
% disp(predictions);

im4 = im2gray(imread('../images/image4.JPG'));
im4 = imresize(im4, 4);
% imshow(im1);
level4 = graythresh(im4);
im4comp = imcomplement(im4);
im4bin = imbinarize(im4comp, level4);
% CC1 = bwconncomp(im1bin);
% L1 = bwlabel(im1bin);
stats4 = regionprops(im4bin, "BoundingBox");
num_bounding_boxes = size(stats4, 1);
predictions = zeros(1, num_bounding_boxes);
figure();

for i = 1:num_bounding_boxes
    box = imcrop(im1bin, stats4(i).BoundingBox);

    input = padarray(box, [20, 20], 'both');
    input = imresize(input, [28 28]);
    
    input = input';
    input = reshape(input, [], 1);
    
    [output, P] = convnet_forward(params, layers, input);
    [max_prob, max_index] = max(P);
    max_index = max_index - 1;
    predictions(i) = max_index;
    subplot(1,num_bounding_boxes, i)
    imshow(box);
    xlabel(max_index);
end
% disp(predictions);
