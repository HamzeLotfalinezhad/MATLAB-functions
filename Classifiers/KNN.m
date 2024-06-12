function [Accuracy, Results, validationPredictions]=KNN(FM,ClassNum,KFold,Distance,NumNeighbors)

% Distance=
% 'cityblock'	City block distance.
% 'chebychev'	Chebychev distance (maximum coordinate difference).
% 'correlation'	One minus the sample linear correlation between observations (treated as sequences of values).
% 'cosine'	One minus the cosine of the included angle between observations (treated as vectors).
% 'euclidean'	Euclidean distance.
% 'hamming'	Hamming distance, percentage of coordinates that differ.
% 'jaccard'	One minus the Jaccard coefficient, the percentage of nonzero coordinates that differ.
% 'mahalanobis'	Mahalanobis distance, computed using a positive definite covariance matrix C. The default value of C is the sample covariance matrix of X, as computed by nancov(X). To specify a different value for C, use the 'Cov' name-value pair argument.
% 'minkowski'	Minkowski distance. The default exponent is 2. To specify a different exponent, use the 'Exponent' name-value pair argument.
% 'seuclidean'	Standardized Euclidean distance. Each coordinate difference between X and a query point is scaled, meaning divided by a scale value S. The default value of S is the standard deviation computed from X, S = nanstd(X). To specify another value for S, use the Scale name-value pair argument.
% 'spearman'	One minus the sample Spearman's rank correlation between observations (treated as sequences of values).
% @distfun	

 
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

                if isempty(Distance)==1;Distance='euclidean';end
                if isempty(NumNeighbors)==1;NumNeighbors=1;end
trainedClassifier = fitcknn(X,Y,'Standardize',1,'Distance',Distance,...
    'NumNeighbors',NumNeighbors);


%% Results

% partModel = crossval(trainedClassifier,'KFold',KFold);
if KFold=='leaveout'
    partModel = crossval(trainedClassifier,'leaveout','on');
else
    partModel = crossval(trainedClassifier,'KFold',KFold);
end
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