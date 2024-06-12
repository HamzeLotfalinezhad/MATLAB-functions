function [BestF, all_BACC, all_BestF, SVM_Results2,ARACC]=GSBS_svmOptimize(FM,...
    Wanted_F,Combination,ClassNum,K_Fold,...
    Kernel_F,BestFComb,P,itr)


% generalized sequential backward search
% if Combination=1 ==>  SBS
All_FeaturesDim=size(FM,2);
if nargin<8
BestFComb=1:All_FeaturesDim;
end
k=0;
all_BACC=0;
RACC=-100;
ACC=0;
LL=All_FeaturesDim;

while (LL>Wanted_F)
    
SeFe=0;
Cost_ACC=0;    
SeFe=BPerms(BestFComb,Combination);

for c=1:size(SeFe,1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if (ClassNum==2 | length(ClassNum)==2)
% [Accuracy, Results]=SVM(FM(:,SeFe(c,:)),ClassNum,K_Fold,[],[],'1',Kernel_F,10,itr);
% else
% [Accuracy, Results]=SVM(FM(:,SeFe(c,:)),ClassNum,K_Fold,[],[],'2',Kernel_F,10,itr);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (ClassNum==2 | length(ClassNum)==2)
[Accuracy, Results]=SVM(FM(:,SeFe(c,:)),ClassNum,K_Fold,1,[],[],Kernel_F,10,itr);
% [Accuracy, Results]=KNN(FM(:,SeFe(c,:)),ClassNum,K_Fold,[],3);

% for nn=1:4
% [acc, Results]=KNN(FM(:,SeFe(c,:)),ClassNum,K_Fold,[],nn);
% acc2(nn)=acc;
% r(nn).rr=Results;
% end
% [MAXX,MAXXInd]=max(acc2);
% Accuracy=MAXX;
% Results=r(MAXXInd).rr;

else
[Accuracy, Results]=SVM(FM(:,SeFe(c,:)),ClassNum,K_Fold,1,[],[],Kernel_F,10,itr);
% [Accuracy, Results]=KNN(FM(:,SeFe(c,:)),ClassNum,K_Fold,[],3);

% for nn=1:4
% [acc, Results]=KNN(FM(:,SeFe(c,:)),ClassNum,K_Fold,[],nn);
% acc2(nn)=acc;
% r(nn).rr=Results;
% end
% [MAXX,MAXXInd]=max(acc2);
% Accuracy=MAXX;
% Results=r(MAXXInd).rr;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


SVM_Results0=Results;

Cost_ACC(c,1)=Accuracy;
SVM_Results00{c,1}=SVM_Results0;

if P==1
disp(['Searching Combination of ',num2str(size(SeFe,2)),' Features, Best Cost ',num2str(all_BACC),', Cost ',num2str(Accuracy)])
end

end
% clc
LL=size(SeFe,2);
[ACC, MaxInd]=max(Cost_ACC);
BestFComb=SeFe(MaxInd,:);
SVM_Results1=SVM_Results00{MaxInd,1};

if ACC >RACC
    RACC=ACC;
    all_BACC=ACC;
    all_BestF=BestFComb;
    SVM_Results2=SVM_Results1;
end
k=k+1;
ARACC(k)=RACC;
disp(['Best Features: ',num2str(BestFComb)])
end
BestF=BestFComb;
if P==1
disp(['Overal Best Features: [',...
    num2str(all_BestF),'], Overal Best Cost ',num2str(all_BACC)])
end

end
%%
function OUT=BPerms(BestF_YouKnow,Combination)

OUT=nchoosek(BestF_YouKnow,length(BestF_YouKnow)-Combination);

end