function [W1, W2, b1, b2, lossHistory, gradHistory, timeHistory] = trainGD(X, Y, W1, W2, b1, b2, epochs, learningRate)
    lossHistory = zeros(epochs, 1);
    gradHistory = zeros(epochs, 1);
    timeHistory = zeros(epochs, 1);  % Store timing for each epoch

    for epoch = 1:epochs
        tic;  % Start timing this epoch
        [output, a1, z2] = feedforward(X, W1, W2, b1, b2);
        delta2 = (output - Y);
        delta1 = (delta2 * W2') .* (1 - tanh(a1).^2);

        lossHistory(epoch) = -sum(Y .* log(output + 1e-12), 'all') / size(Y, 1);
        gradHistory(epoch) = norm(delta2, 'fro') / numel(delta2);  % Frobenius norm for matrix

        W2 = W2 - learningRate * (a1' * delta2);
        b2 = b2 - learningRate * sum(delta2, 1);
        W1 = W1 - learningRate * (X' * delta1);
        b1 = b1 - learningRate * sum(delta1, 1);

        timeHistory(epoch) = toc;  % End timing and record duration
        if mod(epoch, 10) == 0
            fprintf('Epoch %d, Loss: %f, Time: %fs\n', epoch, lossHistory(epoch), timeHistory(epoch));
        end
    end
end
