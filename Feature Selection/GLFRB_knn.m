function [all_BACC, all_BestF, Results]=GLFRB_knn(FM,LR,Wanted_F,ClassNum,K_Fold,NumNeighbor,itr)
%% GLFRB first forward then backward

S=size(FM,2);
if Wanted_F>S
    Wanted_F=S;
end
FM_F=FM;
WF_F=LR(1);
all_BestF_B=0;
Comb_F=1;
Comb_B=1;
BestFComb_F=0;
BestFComb_B=0;
Acc=-100;
%% LOOP
while (size(all_BestF_B,2)<Wanted_F)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SFS 
[BestF_F, all_BACC_F, all_BestF_F, Results_F]=GSFS_knn(FM_F,WF_F,Comb_F,...
    ClassNum,K_Fold,NumNeighbor,BestFComb_F,0,itr);

if  all_BACC_F>Acc
    all_BestF=0;
    all_BACC=all_BACC_F;
    all_BestF=all_BestF_F;
    Results=Results_F;
    Acc=all_BACC_F;
end


if WF_F==Wanted_F %%%%%%%%%%%%%%%%%%%%%%%
   break 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SBS
if WF_F==S
WF_B=0;
break
else
WF_B=size(BestF_F,2)-LR(2);    
end
BestFComb_B=BestF_F;
[BestF_B, all_BACC_B, all_BestF_B, Results_B]=GSBS_knn(FM,WF_B,Comb_B,...
    ClassNum,K_Fold,NumNeighbor,BestFComb_B,0,itr);
BestFComb_F=BestF_B;


WF_F=length(BestF_B)+LR(1);
if WF_F>Wanted_F %%%%%%%%%%%%%%%%%%
   WF_F= Wanted_F;
end


if WF_F>S
    WF_F=S;
end

% save Best    
if  all_BACC_B>Acc
    all_BestF=0;
    all_BACC=all_BACC_B;
    all_BestF=all_BestF_B;
    Results=Results_B;
    Acc=all_BACC_B;
end

end

end



