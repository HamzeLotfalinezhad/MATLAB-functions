function [Accuracy, Results]=SVM_GridSearch(FM,ClassNum,KFold,Optimize,KernelFunction,Num_Grid,Iteration)

% BoxConstraint is "C" parameter in papers
%________________________________________________________________________________________
 % for "KernelScale" If you specify 'auto', then the software selects an appropriate scale factor using
 % a heuristic procedure. This heuristic procedure uses subsampling, so estimates can 
 % vary from one call to another. Therefore, to reproduce results, set a random number 
 % seed using rng before training.
%________________________________________________________________________________________

% Optimize='2'  just optimize KernelScale and BoxConstraint(C), you must
% define KernelFunction (rbf(gausion), linear, polynominal)
% Optimize='3'  optimize KernelScale, BoxConstraint(C) and KernelFunction (rbf(gausion), linear, polynominal)
 
%% if you are not using BFS_GA_.. functions, uncomment next line (rng(1))
% rng(1); % random number generator, if (0), then ovo and ova and other parameters doesnot choose with same probability
rng('default')

[~, InputNum]= size(FM);
if length(ClassNum)==1
Total_ClassNum=ClassNum;
else
Total_ClassNum=length(ClassNum);
end

FM2=DefineClass(FM,ClassNum);
X=FM2(:,1:InputNum);
Y=FM2(:,end);
%%

if Optimize=='2'
 template = templateSVM( 'Standardize',1,'KernelFunction',KernelFunction);
 
trainedClassifier = fitcecoc(X,Y,'Learners',template,...
    'ObservationsIn','rows',...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus',...
    'optimizer','gridsearch',...
    'NumGridDivisions',Num_Grid,...
    'Verbose',0,...
    'ShowPlots',0,...
    'MaxObjectiveEvaluations',Iteration,...
    'SaveIntermediateResults',0)); 

elseif Optimize=='3'
 
     template = templateSVM( 'Standardize',1);
 
trainedClassifier = fitcecoc(X,Y,'Learners',template,...
    'ObservationsIn','rows',...
    'OptimizeHyperparameters','all',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus',...
    'optimizer','gridsearch',...
    'NumGridDivisions',Num_Grid,...
    'Verbose',0,...
    'ShowPlots',0,...
    'MaxObjectiveEvaluations',Iteration,...
    'SaveIntermediateResults',0));     
end
%% Results
% set(groot,'defaultFigureVisible','on') % figure of iteration is on

partModel = crossval(trainedClassifier,'KFold',KFold);
Accuracy = (1-kfoldLoss(partModel,'LossFun', 'ClassifError','mode','average'))*100;
 [validationPredictions] = kfoldPredict(partModel);
 ConfMat=zeros(Total_ClassNum);
 for i=1:length(validationPredictions)
    ConfMat(Y(i),validationPredictions(i))=ConfMat(Y(i),validationPredictions(i))+1;   
 end

Results.Accuracy=Accuracy;
Results.ConfusionMatrix=ConfMat;
if Optimize=='3'
Results.BestResults=sortrows(trainedClassifier.HyperparameterOptimizationResults,8);
elseif Optimize=='2'
Results.BestResults=sortrows(trainedClassifier.HyperparameterOptimizationResults,5);
end
Results.AllResult_TrainedClassifiers=trainedClassifier;
Results.AllResult_CrossValidation=partModel;


end
%% |||||||||||||||||||||||||||||||||||||||||||||||||||||  SUB FUNCTIONS |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\
function Features=DefineClass(Features0,ClassNum)
sf=size(Features0,2);
if length(ClassNum)>1 % if class sizes dont have same number of Trials
    kj=0;
    for i2=1:length(ClassNum)
        for ii=1:ClassNum(i2)
            kj=kj+1;
Features0(kj,sf+1)=i2;
        end
    end
    
else
FMC=size(Features0,1)/ClassNum;
kj=0;
for i2=1:ClassNum
    for j=1:FMC
        kj=kj+1;
Features0(kj,sf+1)=i2;
    end
end

end
Features=Features0;
end