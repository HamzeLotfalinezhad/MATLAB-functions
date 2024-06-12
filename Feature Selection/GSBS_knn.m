function [BestF all_BACC all_BestF Results2]=GSBS_knn(Feature_Matrix,...
    Wanted_F,Combination,ClassNum,K_Fold,NumNeighbor,BestFComb,P,itr)


% generalized sequential backward search
% if Combination=1 ==>  SBS
All_FeaturesDim=size(Feature_Matrix,2);
if nargin<7
BestFComb=1:All_FeaturesDim;
end

all_BACC=0;
RACC=-100;
ACC=0;
LL=All_FeaturesDim;

while (LL>Wanted_F)

SeFe=0;
Cost_ACC=0;    
SeFe=BPerms(BestFComb,Combination);

for c=1:size(SeFe,1)
    

% [Accuracy,~,~, COM] = KNN_Kfold(Feature_Matrix(:,SeFe(c,:)),ClassNum,K_Fold,NumNeighbor);        
% [Accuracy, Results]=KNN(Feature_Matrix(:,SeFe(c,:)),ClassNum,K_Fold,[],NumNeighbor);
[Accuracy, Results]=KNN_GridSearch(Feature_Matrix(:,SeFe(c,:)),ClassNum,K_Fold,'2',10,itr);

Cost_ACC(c,1)=Accuracy;
Results0{c,1}=Results;

if P==1
disp(['Searching Combination of ',num2str(size(SeFe,2)),' Features, Best Cost ',num2str(all_BACC),', Cost ',num2str(Accuracy)])
end

end
% clc
LL=size(SeFe,2);
[ACC MaxInd]=max(Cost_ACC);
BestFComb=SeFe(MaxInd,:);
Results1=Results0{MaxInd,1};

if ACC >RACC
    RACC=ACC;
    all_BACC=ACC;
    all_BestF=BestFComb;
    Results2=Results1;
end
disp(['Best Features: ',num2str(BestFComb)])
end
BestF=BestFComb;

if P==1
disp(['Overal Best Features: [',num2str(all_BestF),'], Overal Best Cost ',num2str(all_BACC)])
end




end
%%
function OUT=BPerms(BestF_YouKnow,Combination)

OUT=nchoosek(BestF_YouKnow,length(BestF_YouKnow)-Combination);

end