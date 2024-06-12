function [Accuracy, Results]=LQDA_GridSearch(FM,ClassNum,KFold,Optimize,DiscrimType,Num_Grid,Iteration)

% DiscrimType=(linear, diaglinear, quadratic, diagquadratic)
 
% Optimize='2'  just optimize gamma and delta

% Optimize='3'  optimize delta, gamma and DiscrimType (linear, diaglinear, quadratic, diagquadratic)
 



[~, InputNum]= size(FM);
if length(ClassNum)==1
Total_ClassNum=ClassNum;
else
Total_ClassNum=length(ClassNum);
end

FM2=DefineClass(FM,ClassNum);
X=FM2(:,1:InputNum);
Y=FM2(:,end);

rng(0,'twister'); % For reproducibility

if Optimize=='2'
 
trainedClassifier = fitcdiscr(X,Y,'DiscrimType',DiscrimType,...
    'ObservationsIn','rows',...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus',...
    'optimizer','randomsearch',...
    'NumGridDivisions',Num_Grid,...
    'Verbose',0,...
    'ShowPlots',0,...
    'MaxObjectiveEvaluations',Iteration,...
    'SaveIntermediateResults',0)); 

elseif Optimize=='3'
  
trainedClassifier = fitcdiscr(X,Y,...
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

partModel = crossval(trainedClassifier,'KFold',KFold);
Accuracy = (1-kfoldLoss(partModel,'LossFun', 'ClassifError','mode','average'))*100;
 [validationPredictions] = kfoldPredict(partModel);
 ConfMat=zeros(Total_ClassNum);
 for i=1:length(validationPredictions)
    ConfMat(Y(i),validationPredictions(i))=ConfMat(Y(i),validationPredictions(i))+1;   
 end

Results.Accuracy=Accuracy;
Results.ConfusionMatrix=ConfMat;


if Optimize=='2'
Results.BestResults=sortrows(trainedClassifier.HyperparameterOptimizationResults,4);
elseif Optimize=='3'
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