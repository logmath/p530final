% This script will perform a lasso regression using the rocket data as a
% function of weather parameters.

% initialize
clear; clc; close all;

% define constants
lamrange = logspace(-3,3,100)'; % lambda values for the LASSO process
tol = 1e-8; % when is a parameter essentially zero?
maxer = 1.5; % maximum acceptable error, used to figure out which parameter to use.

parstrs = {'Avg Press';'Avg Temp';'Wind Dir';'Avg Wind Speed';'Prop Speed';'Mic Press';'Mic Temp';'Mic Wind Speed'};
% read in the weather data, N is the number of measurements and M is the number of weather parameters
load("AverageFalcon9WeatherMetrics.mat")

WeatherData = [AvgPress;AvgTemp;mod(AvgWdir,360);AvgWs;EffPropSpeed';ReceiverPress; ReceiverTemp; ReceiverWs]'; % M x N matrix containing the weather parameter for each measurement
% which data to fit to. N x 1 vector
y = importdL('530 Project.xlsx'); % difference in predicted level
%y = importOASPL('530 Project.xlsx'); % raw OASPL
%y = importOASPLA('530 Project.xlsx'); % A weighted OASPL

% remove transporter 8
WeatherData(5,:) = [];
y(5) = [];

N = size(WeatherData,1);M = size(WeatherData,2);
par0 = ones(M,1); % initial parameter values

% define LASSO functions

A = log10(WeatherData); % create model map

fhat = @(params) A*params; % predict the dB difference
ferror = @(params) norm(fhat(params) - y)^2; % cost of this set of parameters

parfits = zeros([M,length(lamrange)]); % place to keep the parameters
errors = zeros([length(lamrange),1]); % place to keep the erorrs

for i = 1:length(lamrange)
    lam = lamrange(i);
    fLasso = @(params) ferror(params) + lam*sum(abs(params)); % function for lass regression
    
    parfit = fmincon(fLasso,par0,[],[]);

    % save the data
    parfits(:,i) = parfit;
    errors(i) = ferror(parfit);

end % i = 1:length(lamrange)
%%
nonzero = sum(abs(parfits)>tol);

% find the lamnda value for the best parameter fit within our error
erind= find(errors>1.5,1);
lamfit = lamrange(erind);
parfit = parfits(:,erind);

% plot the results
figure()
semilogx(lamrange,errors)
xlabel('\lambda')
ylabel('error, dB')
xline(lamfit)

yyaxis right
semilogx(lamrange,nonzero)
ylabel('Number of nonzero parameters')

% colormap
load('diffcmap.mat')
figure()
imagesc(lamrange,1:M,parfits);
xline(lamfit)
ax = gca;
ax.XScale = 'log';
ax.YTickLabel = parstrs;
cb = colorbar;
colormap(diffcmap)
ax.CLim = [-1,1]*abs(max(clim));
xlabel('\lambda')
