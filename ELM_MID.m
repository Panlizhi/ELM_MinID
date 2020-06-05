function [Output, TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM_MID(train_x, train_y, test_x, test_y, Elm_Type, NumberofHiddenNeurons, ActivationFunction, yita, nbIters,sigma)
%% Input:
% train_x:   training input 
% train_y:   training output 
% test_x:    testing input 
% test_y:    testing output 
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
% yita: step size for gradient method
% nbIters: number of iterations
%
%% Output:
% Output: the predicted output
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification



%% DATA LOAD
REGRESSION=0;
CLASSIFIER=1;
%TRAIN DATA
T=train_y';
P=train_x';
%TEST DATA
TV.T=test_y';
TV.P=test_x';

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

parpool('local',4);
%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;
tic;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'tansig'}
        %%%%%%%% tansig
        H = tansig(tempH);
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;   %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
   
   OutputWeight=zeros(NumberofHiddenNeurons,1);
   
   
   %parameter of Momentum
   V_last=0;
   beta=0.9;
   

   wb=waitbar(0,'Please waiting of ELMMID');


for iter=1:nbIters  
 
   waitbar(iter/nbIters);
   %calculate error
   e=(H'*OutputWeight-T'); 
   a=1/(sqrt(2*pi)*sigma);
   L=NumberofTrainingData;
   dJ_temp1=zeros(NumberofHiddenNeurons,1);
   
   %Calculation of divergence
   parfor j=1:L
       xx = exp(-(e(j,1)-e).^2./(2*sigma^2));
       dJ_temp1 = dJ_temp1 + (H-H(:,j))*(xx.*(e(j,1)-e)./(sigma^2));
   end

   xx = exp(-(e.^2)./(2*sigma^2));
   dJ_temp2 = H*(xx.*e./(-sigma^2));
   dJ=dJ_temp1*a*(1/(L^2))+dJ_temp2*a*(-2)/L;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%Momentum Gradient descent
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   %  1.SGD   
   %  OutputWeight=OutputWeight-yita*dJ;   

   %  2.Momentum
      V_now=beta*V_last+(1-beta)*dJ;
      OutputWeight=OutputWeight-yita*V_now;
      V_last=V_now
   
   %Early stop
   if iter>=2 && min(dJ)<=0.0001
       break
   end
  
end

%%
close(wb);
end_time_train=cputime;

% training time
TrainingTime=end_time_train-start_time_train;      
TrainingTime = toc;

delete(gcp('nocreate'));
% training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(sum((T - Y).^2)/length(T));              %   Calculate training accuracy (RMSE) for regression case
end
clear H;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Calculate the output of testing input
tic;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'tansig'}
        %%%%%%%% tansig 
        H_test = tansig(tempH_test);
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data

 %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
TestingTime = toc;
if Elm_Type == REGRESSION  
    TestingAccuracy=sqrt(sum((TV.T - TY).^2)/length(TY));  %   Calculate testing accuracy (RMSE) for regression case
    Output=TY';
        
end

