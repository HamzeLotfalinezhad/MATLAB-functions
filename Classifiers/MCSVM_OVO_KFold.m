function [Accuracy Sensitivity Specificity ConfM1 Kappa] = MCSVM_OVO_KFold(FM,ClassNum,K_Fold,Kernel_Function,SVMparam)

[Dn, InputNum]= size(FM);
if length(ClassNum)==1
Total_ClassNum=ClassNum;
else
Total_ClassNum=length(ClassNum);
end

Vote=zeros(size(FM,1),Total_ClassNum);

FM2=DefineClass(FM,ClassNum);
count=(1:Dn)';
FM2=[FM2,count];
%%%%%+++++++++++++++++++++++++%%%%%%%%%%%%%%%%%%%%%%
for Cn_1=1:Total_ClassNum
    for Cn_2=Cn_1+1:Total_ClassNum

Class1_ind=0;Class2_ind=0;
Class1_ind=find(FM2(:,end-1)==Cn_1);
Class2_ind=find(FM2(:,end-1)==Cn_2);

S(1)=size(Class1_ind,1);
S(2)=size(Class2_ind,1);

CI(1)=Class1_ind(1,1);
CI(2)=Class2_ind(1,1);

F3=0;
i7=0;
for i5=1:2
    for i6=1:S(i5)
        i7=i7+1;
        F3(i7,1:InputNum+2)=FM2(CI(i5)+i6-1,1:InputNum+2);
    end
end
F4=0;
F4=F3;%DefineClass(F3,S);
KFold_Data(Cn_1,Cn_2).F4=K_FOLDcrossValidation(F4,K_Fold);
    end
end
%% Train all SVMs with k-fold
for Kfold=1:K_Fold
    Cn3=0;
    T=[];
for Cn_1=1:Total_ClassNum
    for Cn_2=Cn_1+1:Total_ClassNum
FKtrain=0;FKtest=0;
FKtrain=find(KFold_Data(Cn_1,Cn_2).F4(:,end)~=Kfold);
FKtest=find(KFold_Data(Cn_1,Cn_2).F4(:,end)==Kfold);

% |||||||||||||||||||||||||||||||   Train and Test
XtrUN=0;XtsUN=0;
XtrUN = KFold_Data(Cn_1,Cn_2).F4(FKtrain,1:end-3);
XtsUN = KFold_Data(Cn_1,Cn_2).F4(FKtest,1:end-3);
%XvlUN = KFold_Data(vlIndex,:);
% ||||||||||||||||||||||||||||||||||||   Target
YtrUC  = KFold_Data(Cn_1,Cn_2).F4(FKtrain,end-2);
YtsUC = KFold_Data(Cn_1,Cn_2).F4(FKtest,end-2:end-1);
% ||||||||||||||||||||||||||||||||||||||| Normalization and ClassCoding
Xtr=XtrUN;
Xts=XtsUN;
%% |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   Training Classifiers |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
% ******************** SVM
% Train
% SVM
TRD=0;TSD=0;
TRD=Xtr;TSD=Xts;
switch Kernel_Function
    case 'linear'
SVMStruct = svmtrain(TRD,YtrUC,'Kernel_Function','linear');
    case 'mlp'
SVMStruct = svmtrain(TRD,YtrUC,'Kernel_Function','mlp','mlp_params',SVMparam);
    case 'rbf'
SVMStruct = svmtrain(TRD,YtrUC,'Kernel_Function','rbf','rbf_sigma',0.5);
end

TSD=[TSD,YtsUC];
SVMs(Cn_1,Cn_2).Kfold(Kfold,1).Net=SVMStruct;
SVMs(Cn_1,Cn_2).Kfold(Kfold,1).test=TSD;

    end
end

end
%% Combine all Test data together
for Kfold=1:K_Fold
    TEST0=[];
    TEST=0;
for Cn_1=1:Total_ClassNum
    for Cn_2=Cn_1+1:Total_ClassNum
        TEST=SVMs(Cn_1,Cn_2).Kfold(Kfold,1).test;
        TEST=[TEST;TEST0];
        TEST0=TEST;
    end
end
for c1=1:Total_ClassNum
    for c2=c1+1:Total_ClassNum
SVMs(c1,c2).Kfold(Kfold,1).testall=TEST;
    end
end

end
%% Test
for Kfold=1:K_Fold
    TEST0=[];
    TestAll=0;
for Cn_1=1:Total_ClassNum
    for Cn_2=Cn_1+1:Total_ClassNum
        
        NET=SVMs(Cn_1,Cn_2).Kfold(Kfold,1).Net;
        TestAll=SVMs(Cn_1,Cn_2).Kfold(Kfold,1).testall;
Ph=0;
Ph = svmclassify(NET,TestAll(:,1:end-2));
for i=1:size(Ph,1);
Vote(TestAll(i,end),Ph(i,1))=Vote(TestAll(i,end),Ph(i,1))+1;
end

    end
end

end
%% Confusion Matrix
Actual_Class=FM2(:,end-1);
CM=zeros(Total_ClassNum);
for i=1:size(Vote,1)
    [~,Predicted_Class]=max(Vote(i,:));
    
    CM(Actual_Class(i,1),Predicted_Class)=CM(Actual_Class(i,1),Predicted_Class)+1;  
end
%% Accuracy and ....
[Accuracy Sensitivity Specificity Kappa]=Classifier_Properties(CM);

ConfM1=CM;
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

