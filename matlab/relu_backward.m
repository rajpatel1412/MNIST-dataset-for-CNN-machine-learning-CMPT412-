function [input_od] = relu_backward(output, input, layer)

% Replace the following line with your implementation.
relu = relu_forward(input);
input_od = (relu.data == input.data) .* output.diff;
    

% input_od = zeros(size(input.data));
end
