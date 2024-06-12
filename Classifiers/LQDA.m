function [Accuracy, Results]=LQDA(FM,ClassNum,KFold,DiscrimType,Gamma,Delta)

% DiscrimType=(linear, diaglinear, quadratic, diagquadratic)
% gamma=scalar value in the interval [0,1]
 
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
rng(0,'twister'); % For reproducibility

                if isempty(Gamma)==1;Gamma=1;end
                if isempty(Delta)==1;Delta=0;end
trainedClassifier = fitcdiscr(X,Y,'DiscrimType',DiscrimType...
    ,'gamma',Gamma,'delta',Delta);


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