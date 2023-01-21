function [output] = inner_product_forward(input, layer, param)

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

% Replace the following line with your implementation.


for b = 1:k
    output.data(:, b) = input.data(:, b)' * param.w + param.b;
end



output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;
end
