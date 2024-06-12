function [Accuracy, Results, validationPredictions]=SVM(FM,ClassNum,KFold,C,S,...
    Optimize,KernelFunction,Num_Grid,Iteration)

% Optimize=='1' --> optimize 2 class
% Optimize=='2' --> optimize multi class auto
% Optimize=='3' --> optimize multi class all



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

ClassNum2=ClassNum;
ClassNum2(ClassNum2==0)=[];
MissCOST=Define_MissCost(ClassNum2);
% MissCOST=ones(Total_ClassNum)-eye(Total_ClassNum);
%%
if (Total_ClassNum==2)  % if 2 class
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(Optimize)==1 %  No optimize 2 class
       
                if isempty(S)==1;S='auto';end
                if isempty(C)==1;C=1;end
  trainedClassifier = fitcsvm(X,Y,'Standardize',1,...
     'KernelFunction',KernelFunction,...
     'KernelScale',S,...
     'BoxConstraint',C,...
     'Cost',MissCOST);


elseif Optimize=='1' %  optimize 2 class
    
trainedClassifier = fitcsvm(X,Y,'Standardize',1,...
    'KernelFunction',KernelFunction,...
    'Cost',MissCOST,...
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif  (Total_ClassNum>2)

if isempty(Optimize)==1 % No optimize multiclass
                if isempty(S)==1;S='auto';end
                if isempty(C)==1;C=1;end 
template = templateSVM( 'Standardize',1,...
    'KernelFunction',KernelFunction,...
    'KernelScale',S,...
    'BoxConstraint',C);

trainedClassifier = fitcecoc(X,Y,'Learners',template,...
    'ObservationsIn','rows',...
    'Cost',MissCOST);

elseif Optimize=='2'  % optimize multiclass
 template = templateSVM( 'Standardize',1,...
     'KernelFunction',KernelFunction);
 
trainedClassifier = fitcecoc(X,Y,'Learners',template,...
    'ObservationsIn','rows',...
    'Cost',MissCOST,...
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
    'Cost',MissCOST,...
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
end
%% Results
% set(groot,'defaultFigureVisible','on') % figure of iteration is on
if KFold=='leaveout'
    partModel = crossval(trainedClassifier,'leaveout','on');
else
    partModel = crossval(trainedClassifier,'KFold',KFold);
end
Accuracy = (1-kfoldLoss(partModel,'LossFun', 'ClassifError','mode','average'))*100;
% Accuracy = (1-kfoldLoss(partModel,'LossFun', 'ClassifError','mode','individual'))*100;
   
 [validationPredictions] = kfoldPredict(partModel);
 ConfMat=zeros(Total_ClassNum);
 for i=1:length(validationPredictions)
    ConfMat(Y(i),validationPredictions(i))=ConfMat(Y(i),validationPredictions(i))+1;   
 end
%  validationPredictions=[validationPredictions,Y];
 
Results.Accuracy=Accuracy;
Results.ConfusionMatrix=ConfMat;
if Optimize=='3'
Results.BestResults=sortrows(trainedClassifier.HyperparameterOptimizationResults,8);
elseif Optimize=='2'
Results.BestResults=sortrows(trainedClassifier.HyperparameterOptimizationResults,5);
elseif Optimize=='1'
Results.BestResults=sortrows(trainedClassifier.HyperparameterOptimizationResults,4);
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