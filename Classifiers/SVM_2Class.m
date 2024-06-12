function [Accuracy, Results]=SVM_2Class(FM,ClassNum,KFold,...
    KernelFunction,C,S,Optimize,Num_Grid,Iteration)


% BoxConstraint is "C" parameter in papers
%________________________________________________________________________________________
 % for "KernelScale" If you specify 'auto', then the software selects an appropriate scale factor using
 % a heuristic procedure. This heuristic procedure uses subsampling, so estimates can 
 % vary from one call to another. Therefore, to reproduce results, set a random number 
 % seed using rng before training.
%________________________________________________________________________________________

 rng default

 
 [~, InputNum]= size(FM);
if length(ClassNum)==1
Total_ClassNum=ClassNum;
else
Total_ClassNum=length(ClassNum);
end

FM2=DefineClass(FM,ClassNum);
X=FM2(:,1:InputNum);
Y=FM2(:,end);

% rng(1); % For reproducibility
if isempty(Optimize)==1

        switch KernelFunction
%___________________________________________________________________________________
            case 'linear'
                if isempty(S)==1;S='auto';end
                if isempty(C)==1;C=1;end
%  template = templateSVM( 'Standardize',1,...
%      'KernelFunction',KernelFunction,'KernelScale',S,'BoxConstraint',C);
 trainedClassifier = fitcsvm(X,Y,'Standardize',1,...
     'KernelFunction',KernelFunction,'KernelScale',S,'BoxConstraint',C);

%_______________________________________________________________________________
            case 'rbf'
                
                if isempty(S)==1;S='auto';end
                if isempty(C)==1;C=1;end
%  template = templateSVM( 'Standardize',1,...
%      'KernelFunction',KernelFunction,'KernelScale',S,'BoxConstraint',C);
  trainedClassifier = fitcsvm(X,Y,'Standardize',1,...
     'KernelFunction',KernelFunction,'KernelScale',S,'BoxConstraint',C);
 %___________________________________________________________________________________       
        end  


elseif isempty(Optimize)==0
    
% trainedClassifier = fitcsvm(X,Y,'Learners', template,'Standardize',1);
% else
trainedClassifier = fitcsvm(X,Y,'Standardize',1,...
    'KernelFunction',KernelFunction,...
    'KernelScale','auto',...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName',...
    'expected-improvement-plus',...
    'optimizer','gridsearch',...
    'NumGridDivisions',Num_Grid,...
    'Verbose',0,...
    'ShowPlots',0,...
    'MaxObjectiveEvaluations',Iteration));
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
% Results.KernelScales=trainedClassifier.BinaryLearners;
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