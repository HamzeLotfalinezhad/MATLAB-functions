function [Accuracy Sensitivity Specificity ConfM1 Kappa] = SVM_Kfold(FM0,ClassNum,K_Fold,Kernel_Function,SVMparam)

% rows: number of participates of different class 
% columns: number of features
% last column of Features must be Target class codes form 1 to ....
%% define class
[~, InputNum]= size(FM0);

if length(ClassNum)==1
Total_ClassNum=ClassNum;
else
Total_ClassNum=length(ClassNum);
end

Features=DefineClass(FM0,ClassNum);
KFold_Data=K_FOLDcrossValidation(Features,K_Fold);
%% ||||||||||||||||||||||||||||||||||||||||||| K_FOLD |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
for i=1:K_Fold
FKtrain=find(KFold_Data(:,end)~=i);
FKtest=find(KFold_Data(:,end)==i);

%    |||||||||||||||||||||||||||||||   Train and Test
Train1 = KFold_Data(FKtrain,1:end-2);
Test1 = KFold_Data(FKtest,1:end-2);
% ||||||||||||||||||||||||||||||||||||   Target
TrainTarget_uc  = KFold_Data(FKtrain,end-1);
TestTarget_uc = KFold_Data(FKtest,end-1);
% ||||||||||||||||||||||||||||||||||||||| Normalization and ClassCoding
Train2=Train1;
Test2=Test1;
%% |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   Training Classifiers |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
% ******************** SVM
% Train
% SVM
TRD=Train2;TSD=Test2;
switch Kernel_Function
    case 'linear'
SVMStruct = svmtrain(TRD,TrainTarget_uc,'Kernel_Function','linear');
    case 'mlp'
SVMStruct = svmtrain(TRD,TrainTarget_uc,'Kernel_Function','mlp','mlp_params',SVMparam);
    case 'rbf'
SVMStruct = svmtrain(TRD,TrainTarget_uc,'Kernel_Function','rbf','boxconstraint',1,'rbf_sigma',SVMparam);
% SVMStruct = svmtrain(TRD,TrainTarget_uc,'Kernel_Function',@MyKernelsvm);SVMStruct = svmtrain(TRD,TrainTarget_uc,'Kernel_Function','rbf','boxconstraint',1,'rbf_sigma',SVMparam);

%%%%% matlab 2017
% SVMStruct = fitcsvm(TRD,TrainTarget_uc,'Standardize',true,'KernelFunction','RBF',...
%     'KernelScale','auto');

end
Predicted_Class = svmclassify(SVMStruct,TSD);

Actual_Class=TestTarget_uc;
CM=confusionmat(Actual_Class,Predicted_Class,'order',1:Total_ClassNum);

ConfM{i,1}=CM;
end
ConfM1=0;
for i=1:K_Fold
ConfM1=ConfM{i,1}+ConfM1;
end
[Accuracy Sensitivity Specificity Kappa]=Classifier_Properties(ConfM1);
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
function D5=K_FOLDcrossValidation(input,K1)
s=size(input,1);
if s<K1 % if k-fold is greater than size of data
  K1=s;  
end
INd=crossvalind('Kfold', s, K1);
D4=input;
    for k=1:s
        D4(k,size(input,2)+1)=INd(k);
    end
D5=D4;
    
end
function [Accuracy Sensitivity Specificity Kappa]=Classifier_Properties(CM)
% accuracy
Accuracy=100*nansum(diag(CM)) / (sum(sum(CM)));
Accuracy(isnan(Accuracy))=0;
% other properties
if length(CM)>2
TP=diag(CM);
FP=(sum(CM)-TP')';
FN=sum(CM,2)-TP;
for i=1:length(CM)
    tn=CM;
    tn(i,:)=[];
    tn(:,i)=[];
    TN(i,1)=sum(sum(tn));
end
TPR=TP./(TP+FN);
TNR=TN./(TN+FP);
Sensitivity=100*(mean(TPR));
Specificity=100*(mean(TNR));
else
    
Sensitivity=100*CM(1,1)/(sum(CM(1,:)));
Specificity=100*CM(2,2)/(sum(CM(2,:)));

end

Predicted_Acc=(sum(CM)*sum(CM,2))/(sum(sum(CM)).^2);
Kappa=100*(0.01*Accuracy-Predicted_Acc)/(1-Predicted_Acc);
end
