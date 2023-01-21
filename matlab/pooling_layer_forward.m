function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    
    c_matrix = zeros([h_out,w_out,c]);
    r_out = zeros(1, w_out); 
    for b = 1:batch_size
        im = reshape(input.data(:, b), [h_in, w_in, c]);
        im = padarray(im, [pad, pad], 0);
        for ch = 1:c
            i = 1;
            for h = 1: stride: (2*pad) + h_in
               j = 1;
               for w = 1: stride : (2*pad) + w_in
                    r_out(j) = max(im(h: h+k-1, w: w+k-1, ch),[], 'all');
                    j = j + 1;
               end
               c_matrix(i, :, ch) = r_out;
               i = i + 1;
            end
        end
        output.data(:, b) = reshape(c_matrix, [], h_out*w_out*c);    
    end
end

