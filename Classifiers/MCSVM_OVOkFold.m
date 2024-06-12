function [Accuracy Sensitivity Specificity ConfM1] = MCSVM_OVOkFold(FM,ClassNum,K_Fold,Kernel_Function,SVMparam)


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

Ph=0;
Ph = svmclassify(SVMStruct,TSD(:,1:end-2));
for i=1:size(Ph,1);
Vote(TSD(i,end),Ph(i,1))=Vote(TSD(i,end),Ph(i,1))+1;
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

Accuracy=nansum(diag(CM)) / (sum(sum(CM)))*100;
Accuracy(isnan(Accuracy))=0;
% other properties sensitivity and ....
for i=1:length(CM)
    TP(i)=CM(i,i);
    FP(i)=abs(sum(CM(:,i))-TP(i));
    FN(i)=abs(sum(CM(i,:))-TP(i));
    tn=CM;
    tn(i,:)=[];tn(:,i)=[];
    TN(i)=sum(sum(tn));
end
for i=1:length(CM)
    TPR(i)=TP(i)/(TP(i)+FN(i));
    TNR(i)=TN(i)/(TN(i)+FP(i));
    F_score(i)=2*TP(i)/(2*TP(i)+TN(i)+FP(i));
end
if length(CM)>2
Sensitivity=(nansum(TPR)/length(CM))*100;
Specificity=(nansum(TNR)/length(CM))*100;
else
Sensitivity=(max(TPR))*100;
Specificity=(min(TNR))*100;   
end
Sensitivity(isnan(Sensitivity))=0;
Specificity(isnan(Specificity))=0;

% F_Score=(nansum(F_score)/length(CM))*100;
% CP = classperf(Actual_Class, Predicted_Class);
% Sensitivity=CP.Sensitivity*100;
% Specificity=CP.Specificity*100;
ConfM1=CM;
%% Worst Case Accuracy between pair of Classes
% k=0;
% for i=1:Total_ClassNum
%     k=k+1; 
%     Acc(k)=100*ConfM1(i,i)/length(find(FM2(:,end-1)==i));
% end
% WorstCase_Accuracy=min(Acc);
% Accuracy=WorstCase_Accuracy;

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

