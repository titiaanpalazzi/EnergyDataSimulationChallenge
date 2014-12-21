%% Initialization for cambridge energy exercise

%% ================ Part 1: Feature Normalization ================
%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load data
% Load training data
training_data = csvread('training_dataset_500.csv');
X = training_data(1:end,6:7);
y = training_data(1:end,8);
m_training = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m_training, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


%% Part 3: predict the energy production
% Load test data
test_data = csvread('test_dataset_500.csv');
X_test = test_data(1:end,6:7);
y_test = test_data(1:end,8);
m_test = length(X_test);
[X_test, mu, sigma] = featureNormalize(X_test);
% num_features = size(X_test,2);
% num_trainings = size(X_test,1);
% 
% for i = 1:num_trainings;
%     for j = 1:num_features;
%         X_test(i,j) = (X(i,j) - mu(j)) / sigma(j);
%     end
% end

% Add intercept term to X_test
X_test = [ones(m_test, 1) X_test];

% Calculate the predicted energy values
% test_theta = repmat(transpose(theta),499,1);
h_test = (X_test)*theta;
predicted_energy_production = h_test;

%% Part 4: calculate the MAPE
abs_cost = abs(y_test-h_test);
rel_cost = abs_cost./y_test;
MAPE = 1/m_test * sum(rel_cost);

    
