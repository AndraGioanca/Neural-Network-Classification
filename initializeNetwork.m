function [W1, W2, b1, b2] = initializeNetwork(inputSize, hiddenSize, outputSize)
    W1 = randn(inputSize, hiddenSize) * 0.01;
    W2 = randn(hiddenSize, outputSize) * 0.01;
    b1 = zeros(1, hiddenSize);
    b2 = zeros(1, outputSize);
end
