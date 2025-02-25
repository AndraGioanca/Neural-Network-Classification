function y = softmax(z)
    exp_z = exp(z - max(z, [], 2)); % Stability improvement by subtracting max from z
    y = exp_z ./ sum(exp_z, 2); % Ensure the outputs sum to 1 across each row
end
