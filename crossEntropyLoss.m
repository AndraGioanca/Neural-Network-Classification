function [loss, grad] = crossEntropyLoss(output, target)
    epsilon = 1e-12;  % To prevent log(0)
    output = max(epsilon, min(1 - epsilon, output));  % Clamp values
    loss = -sum(target .* log(output), 'all') / size(target, 1);
    grad = (output - target) / size(target, 1);  % Gradient for softmax cross-entropy
end
