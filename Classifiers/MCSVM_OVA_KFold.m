function [Accuracy Sensitivity Specificity ConfM1 Kappa] = MCSVM_OVA_KFold(FM,ClassNum,K_Fold,Kernel_Function,SVMparam)

% ARLAN Codes: Ove versus All
[Dn, InputNum]= size(FM);
if length(ClassNum)==1
Total_ClassNum=ClassNum;
else
Total_ClassNum=length(ClassNum);
end
Vote=zeros(size(FM,1),1+Total_ClassNum);

FM2=DefineClass(FM,ClassNum);
count=(1:Dn)';
FM2=[FM2,count];
%%%%%+++++++++++++++++++++++++%%%%%%%%%%%%%%%%%%%%%%
for Cn=1:Total_ClassNum

F6=FM2;
F5=FM2(:,end-1);
F5(F5~=Cn)=-Cn;
F6=[F6,F5];
KFold_Data(Cn,1).F4=K_FOLDcrossValidation(F6,K_Fold);   

end
%+++++++++++++++++++++++++++++++%%%%%%%%%%%%
%% Train all SVMs with k-fold
for Kfold=1:K_Fold
    Cn3=0;
for Cn=1:Total_ClassNum

FKtrain=find(KFold_Data(Cn,1).F4(:,end)~=Kfold);
FKtest=find(KFold_Data(Cn,1).F4(:,end)==Kfold);

% |||||||||||||||||||||||||||||||   Train and Test
XtrUN = KFold_Data(Cn,1).F4(FKtrain,1:end-4);
XtsUN = KFold_Data(Cn,1).F4(FKtest,1:end-4);
%XvlUN = KFold_Data(vlIndex,:);
% ||||||||||||||||||||||||||||||||||||   Target
YtrUC  = KFold_Data(Cn,1).F4(FKtrain,end-1);
YtsUC = KFold_Data(Cn,1).F4(FKtest,end-1);
% ||||||||||||||||||||||||||||||||||||||| Normalization and ClassCoding
Xtr=XtrUN;
Xts=XtsUN;
%% |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   Training Classifiers |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
% ******************** SVM
% Train
% SVM
TRD=Xtr;TSD=Xts;
switch Kernel_Function
    case 'linear'
SVMStruct = svmtrain(TRD,YtrUC,'Kernel_Function','linear');
    case 'mlp'
SVMStruct = svmtrain(TRD,YtrUC,'Kernel_Function','mlp','mlp_params',SVMparam);
    case 'rbf'
SVMStruct = svmtrain(TRD,YtrUC,'Kernel_Function','rbf','rbf_sigma',SVMparam);
end

TSD=[TSD,YtsUC];
Vn = KFold_Data(Cn,1).F4(FKtest,end-3:end-2);
TSD=[TSD,Vn];
SVMs(Cn,1).Kfold(Kfold,1).Net=SVMStruct;
SVMs(Cn,1).Kfold(Kfold,1).test=TSD;
end

end

%% Combine all Test data together
for Kfold=1:K_Fold
    TEST0=[];
    TEST=0;
for Cn=1:Total_ClassNum
        TEST=SVMs(Cn,1).Kfold(Kfold,1).test;
        TEST=[TEST;TEST0];
        TEST0=TEST;
end
for c1=1:Total_ClassNum
SVMs(c1,1).Kfold(Kfold,1).testall=TEST;
end

end
%% Test
for Kfold=1:K_Fold
    TEST0=[];
    TestAll=0;
for Cn=1:Total_ClassNum
        
        NET=SVMs(Cn,1).Kfold(Kfold,1).Net;
        TestAll=SVMs(Cn,1).Kfold(Kfold,1).testall;
Ph=0;
Ph = svmclassify(NET,TestAll(:,1:end-3));
for i=1:size(Ph,1);
    H=1:Total_ClassNum;
    if Ph(i,1)<0
    H(-Ph(i,1))=[];    
%     else
        
    end
Vote(TestAll(i,end),H)=Vote(TestAll(i,end),H)+1;
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

