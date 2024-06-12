function [Accuracy Sensitivity Specificity ConfM1] = RBF_Kfold(FM0,ClassNum,K_Fold,Goal,Spread,MaxNeorun)
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
[Train2 Test2]=Normalization(Train1,Test1,InputNum);
[TrainTarget_c TestTarget_c]=ClassCoding(TrainTarget_uc,TestTarget_uc,ClassNum);
%% |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   Training Classifiers |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
% ******************** RBF
DisplayAt=2000;
Network=newrb(Train2',TrainTarget_c',Goal,Spread,MaxNeorun,DisplayAt);
% Test
% |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| train results
TrainTarget_Network = sim(Network,Train2')';
Classes_tr = CalcClasses_Fcn(TrainTarget_Network);
%||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| test results
TestTarget_Network = sim(Network,Test2')';
Classes_ts = CalcClasses_Fcn(TestTarget_Network);
%|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
Actual_Class_Train=TrainTarget_uc;
CM_Train=confusionmat(Actual_Class_Train,Classes_tr,'order',1:Total_ClassNum);
Actual_Class=TestTarget_uc;
CM=confusionmat(Actual_Class,Classes_ts,'order',1:Total_ClassNum);

ConfM_Train{i,1}=CM_Train;
ConfM{i,1}=CM;
end
ConfM1_Train=0;
for i=1:K_Fold
ConfM1_Train=ConfM_Train{i,1}+ConfM1_Train;
end
Accuracy(1)=100*nansum(diag(ConfM1_Train)) / (sum(sum(ConfM1_Train)));
% Test
ConfM1=0;
for i=1:K_Fold
ConfM1=ConfM{i,1}+ConfM1;
end
[Accuracy(2) Sensitivity Specificity Kappa]=Classifier_Properties(ConfM1);

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
function [Train_N Test_N]=Normalization(Train_NotN,Test_NotN,InputNum)

Min_Train_NotN = min(Train_NotN);Max_Train_NotN = max(Train_NotN); 
Min_Test_NotN = min(Test_NotN);Max_Test_NotN = max(Test_NotN);

Train_N=Train_NotN;
Test_N=Test_NotN;
for ii = 1:InputNum
    Train_N(:,ii) = NORMF(Train_NotN(:,ii),Min_Train_NotN(ii),Max_Train_NotN(ii));
end  
if size(Test_N,1)==1
 Test_N = NORMF(Test_NotN,Min_Test_NotN,Max_Test_NotN);
else
    for ii = 1:InputNum   
    Test_N(:,ii) = NORMF(Test_NotN(:,ii),Min_Test_NotN(ii),Max_Test_NotN(ii));
    end
end
 
end
function xN = NORMF(x,MinX,MaxX)
 
xN = (x - MinX) / (MaxX - MinX) * 2 - 1;
 
end
function [Train_YO Test_YO]=ClassCoding(Train_NC,Test_NC,ClassNum)
if length(ClassNum)>1
CNum=length(ClassNum); % for non equal trials of classes
else
CNum=ClassNum;    % for equal trials of classes
end
Train_YO = zeros(size(Train_NC,1),CNum);
Test_YO = zeros(size(Test_NC,1),CNum);

for ii=1: size(Train_YO,1);
    Train_YO(ii,Train_NC(ii))=1;
end

for ii=1: size(Test_YO,1);
    Test_YO(ii,Test_NC(ii))=1;
end
end
function Classes = CalcClasses_Fcn(NetOut)
Classes = zeros(size(NetOut,1),1);

for ii = 1:numel(Classes)
    NO = NetOut(ii,:);
    I = find(NO==max(NO));
    I = I(1);
    Classes(ii) = I;
end

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
