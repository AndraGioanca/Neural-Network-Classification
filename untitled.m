X = Iris{:,2:5}
y = Iris{:,end}
%%
[uniqueLabels, ~, labelIndices] = unique(y);
y = labelIndices;
y = dummyvar(labelIndices)
%%
nTrain = floor(0.8 * height(Iris))
randomIndices = randperm(height(Iris));
X = X(randomIndices,:)
y = y(randomIndices,:)
X_train = X(1:nTrain,:)
X_test = X(nTrain:end,:)
y_train = y(1:nTrain,:)
y_test = y(nTrain:end,:)
%%
inputSize = 4
hiddenSize = 15
outputSize = 3
% net = SimpleNN(inputSize, hiddenSize, outputSize)
[W1, W2, b1, b2] = initializeNetwork(inputSize, hiddenSize, outputSize)
%%
epochs = 2000;
learningRate = 0.001;

% net = net.trainGD(X_train,y_train,epochs,learningRate);
[W1, W2, b1, b2, lossHistory, gradHistory, timeHistory] = trainGD(X_train, y_train, W1, W2, b1, b2, epochs, learningRate)
figure;
subplot(3,1,1);
plot(lossHistory);
title('Loss History');
xlabel('Epoch');
ylabel('Loss');

subplot(3,1,2);
plot(gradHistory);
title('Gradient Magnitude History');
xlabel('Epoch');
ylabel('Gradient Magnitude');

subplot(3,1,3);
plot(timeHistory);
title('Time per Epoch');
xlabel('Epoch');
ylabel('Time (seconds)');
%%
% [output, ~, ~] = net.feedforward(X_test)
[output, a1, z2] = feedforward(X_test, W1, W2, b1, b2)
[~, predictedLabels] = max(output, [], 2);
[~, actualLabels] = max(y_test, [], 2);
accuracy = sum(predictedLabels == actualLabels) / length(actualLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
C = confusionmat(actualLabels, predictedLabels);
figure;
confusionchart(C);
title('Confusion Matrix');
%%
epochs = 1000;
learningRate = 0.001;
[W1, W2, b1, b2] = initializeNetwork(inputSize, hiddenSize, outputSize)
[W1, W2, b1, b2, lossHistory, gradHistory, timeHistory] = trainSGD(X_train, y_train, W1, W2, b1, b2, epochs, learningRate)
figure;
subplot(3,1,1);
plot(lossHistory);
title('Loss History');
xlabel('Epoch');
ylabel('Loss');

subplot(3,1,2);
plot(gradHistory);
title('Gradient Magnitude History');
xlabel('Epoch');
ylabel('Gradient Magnitude');

subplot(3,1,3);
plot(timeHistory);
title('Time per Epoch');
xlabel('Epoch');
ylabel('Time (seconds)'); 
%%

[output, a1, z2] = feedforward(X_test, W1, W2, b1, b2)
[~, predictedLabels] = max(output, [], 2);
[~, actualLabels] = max(y_test, [], 2);
accuracy_sgd = sum(predictedLabels == actualLabels) / length(actualLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy_sgd * 100);
%%
C = confusionmat(actualLabels, predictedLabels);
figure;
confusionchart(C);
title('Confusion Matrix');