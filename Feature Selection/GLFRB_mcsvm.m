function [all_BACC all_BestF COM2]=GLFRB_mcsvm(FM,LR,Wanted_F,ClassNum,K_Fold,Kernel_F,Kernel_P,SvmType)
%% GLFRB first forward then backward
if length(LR)==2
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
while (size(all_BestF_B,2)<Wanted_F)
% SFS 
[BestF_F all_BACC_F all_BestF_F COM_F]=GSFS_mcsvm(FM_F,WF_F,Comb_F,...
    ClassNum,K_Fold,Kernel_F,Kernel_P,BestFComb_F,0,SvmType);

if  all_BACC_F>Acc
    all_BestF=0;
    all_BACC=all_BACC_F;
    all_BestF=all_BestF_F;
    COM2=COM_F;
    Acc=all_BACC_F;
end
% SBS
if WF_F==S
WF_B=0;
break
else
WF_B=size(BestF_F,2)-LR(2);    
end
BestFComb_B=BestF_F;
[BestF_B all_BACC_B all_BestF_B COM_B]=GSBS_mcsvm(FM,WF_B,Comb_B,...
    ClassNum,K_Fold,Kernel_F,Kernel_P,BestFComb_B,0,SvmType);

BestFComb_F=BestF_B;

WF_F=length(BestF_B)+LR(1);
if WF_F>S
    WF_F=S;
end

% save Best    
if  all_BACC_B>Acc
    all_BestF=0;
    all_BACC=all_BACC_B;
    all_BestF=all_BestF_B;
    COM2=COM_B;
    Acc=all_BACC_B;
end

end
elseif length(LR)>2
%% Floating search, first forward then backward random changing R and L
% S=size(FM,2);
% if Wanted_F>S
%     Wanted_F=S;
% end
% FM_F=FM;
% all_BestF_B=0;
% Comb_F=1;
% Comb_B=1;
% BestFComb_F=0;
% BestFComb_B=0;
% Acc=-100;
% WF_B=0;
% while (size(all_BestF_B,2)<=Wanted_F)
% % SFS 
% WF_F=WF_B+randi(LR(1:2));
% WF_B=randi(LR(3:4));
% 
% [~, BestF_F all_BACC_F all_BestF_F COM_F]=GSFS_svm(FM_F,WF_F,Comb_F,...
%     ClassNum,K_Fold,Kernel_F,Kernel_P,BestFComb_F);
% q=length(BestF_F);
% if  all_BACC_F>Acc
%     all_BestF=0;
%     all_BACC=all_BACC_F;
%     all_BestF=all_BestF_F;
%     COM2=COM_F;
%     Acc=all_BACC_F;
% end
% % SBS
% if WF_F==S
% WF_B=0;
% break
% else
% WF_B=size(BestF_F,2)-WF_B;    
% end
% BestFComb_B=BestF_F;
% [~, BestF_B all_BACC_B all_BestF_B COM_B]=GSBS_svm(FM,WF_B,Comb_B,...
%     ClassNum,K_Fold,Kernel_F,Kernel_P,BestFComb_B);
% 
% BestFComb_F=BestF_B;
% 
% WF_F=length(BestF_B)+WF_F;
% if WF_F>S
%     WF_F=S;
% end
% 
% % save Best    
% if  all_BACC_B>Acc
%     all_BestF=0;
%     all_BACC=all_BACC_B;
%     all_BestF=all_BestF_B;
%     COM2=COM_B;
%     Acc=all_BACC_B;
% end

% end

end
end
