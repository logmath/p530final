% This script will perform a lasso regression using the rocket data as a
% function of weather parameters.

% You will need to download the appropriate files from box since data won't
% be stored on git.

% initialize
clear; clc; close all;

% define constants
lamrange = logspace(-3,3,100)'; % lambda values for the LASSO process
tol = 1e-8; % when is a parameter essentially zero?
maxer = [1.5,2]; % maximum acceptable error, used to figure out which parameter to use.
ercolvec = ['r','g'];

parstrs = {'Avg Press';'Avg Temp';'Avg Wind';'Prop Speed';'Mic Temp';'Mic Wind';'Rocket Temp';'Rocket Wind'};
% read in the weather data, N is the number of measurements and M is the number of weather parameters
load("AverageFalcon9WeatherMetrics.mat")

WeatherData = [AvgPress; AvgTemp+273; AvgWs; EffPropSpeed'; ReceiverTemp+273; ReceiverWs; SourceTemp+273; SourceWs]'; % M x N matrix containing the weather parameter for each measurement
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
ferror = @(params) norm(fhat(params) - y); % cost of this set of parameters

parfits = zeros([M,length(lamrange)]); % place to keep the parameters
errors = zeros([length(lamrange),1]); % place to keep the erorrs

for i = 1:length(lamrange)
    lam = lamrange(i);
    fLasso = @(params) ferror(params)^2 + lam*sum(abs(params)); % function for lass regression
    
    parfit = fmincon(fLasso,par0,[],[]);

    % save the data
    parfits(:,i) = parfit;
    errors(i) = ferror(parfit);

end % i = 1:length(lamrange)
%%
nonzero = sum(abs(parfits)>tol);

% find the lamnda value for the best parameter fit within our error

lamfits = zeros(size(maxer));
parlam = zeros([length(par0),length(maxer)]);
for i = 1:length(maxer)
    erind= find(errors>maxer(i),1);
    lamfit(i) = lamrange(erind);
    parlam(:,i) = parfits(:,erind);
end

% plot the results
figure()
semilogx(lamrange,errors)
xlabel('\lambda')
ylabel('error, dB')
xline(lamfit)

yyaxis right
semilogx(lamrange,nonzero)
ylabel('Number of nonzero parameters')

% value colormap
load('diffcmap.mat')
figure()
imagesc(lamrange,1:M,parfits);
xline(lamfit)
ax = gca;
ax.XScale = 'log';
ax.YTickLabel = parstrs;
cb = colorbar;
cb.Label.String = 'Parameter value';
colormap(diffcmap)
ax.CLim = [-1,1]*abs(max(clim));
xlabel('\lambda')

% significance colormap
sig = parfits./max(abs(parfits'))'; % normalized parameter values
sig = log10(abs(parfits));

figure()
imagesc(lamrange,1:M,sig);
xline(lamfit)
ax = gca;
ax.XScale = 'log';
ax.YTickLabel = parstrs;
cb = colorbar;
cb.Label.String = 'log_{10} |value|';
colormap sky
%colormap(diffcmap)
%ax.CLim = [-1,1];%[-1,1]*abs(max(clim));
xlabel('\lambda')


% create the compared results
figure()
plot(y,'bo','MarkerFaceColor','b','DisplayName','Measured')
hold on
for i = 1:length(maxer)
    plot(fhat(parlam(:,i)),['o',ercolvec(i)],'MarkerFaceColor',ercolvec(i),'DisplayName',sprintf('Predicted (%0.1f dB)',maxer(i)))
end
legend('location','best')
xlabel('Measurement')
ylabel('\Delta dB')