function [output, a1, z2] = feedforward(X, W1, W2, b1, b2)
    z1 = X * W1 + repmat(b1, size(X, 1), 1);
    a1 = customActivation(z1);
    z2 = a1 * W2 + repmat(b2, size(a1, 1), 1);
    output = softmax(z2);
end
