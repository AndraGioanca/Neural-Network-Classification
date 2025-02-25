function [W1, W2, b1, b2, lossHistory, gradHistory, timeHistory] = trainSGD(X, Y, W1, W2, b1, b2, epochs, learningRate)
    n = size(X, 1);  % Total number of examples
    lossHistory = zeros(epochs, 1);
    gradHistory = zeros(epochs, 1);
    timeHistory = zeros(epochs, 1);  % Store timing for each epoch

    for epoch = 1:epochs
        tic;  % Start timing this epoch
        indices = randperm(n);
        totalLoss = 0;  % To calculate mean loss per epoch
        totalGrad = 0;  % To calculate mean gradient magnitude per epoch
        
        for i = indices
            [output, a1, z2] = feedforward(X(i, :), W1, W2, b1, b2);
            delta2 = output - Y(i, :);
            delta1 = (delta2 * W2') .* (1 - tanh(a1).^2);

            totalLoss = totalLoss - sum(Y(i, :) .* log(output + 1e-12));
            totalGrad = totalGrad + norm(delta2);

            W2 = W2 - learningRate * (a1' * delta2);
            b2 = b2 - learningRate * sum(delta2, 1);
            W1 = W1 - learningRate * (X(i, :)' * delta1);
            b1 = b1 - learningRate * sum(delta1, 1);
        end

        timeHistory(epoch) = toc;  % End timing and record duration
        lossHistory(epoch) = totalLoss / n;
        gradHistory(epoch) = totalGrad / n;

        fprintf('Epoch %d, Loss: %f, Time: %fs\n', epoch, lossHistory(epoch), timeHistory(epoch));
    end
end
