function IG=InfoGain(x,ClassNum,Children_Num)
if nargin <3
    Children_Num=2;
end
% InfoGain.pdf
R=size(x,1);
x=NORM01(x);
x2=DefineClass(x,ClassNum);

Child_num=0:1/Children_Num:1;

[~,C_ind]=histc(x(:,1),Child_num);
C_ind(C_ind==length(Child_num))=length(Child_num)-1;

% find children
L=0;
y=x2(:,end);
for i=1:Children_Num
    ChildC=find(C_ind==i);
    CL(1,i)=length(ChildC);
    Children(i,1).Child_Class=y(ChildC);
    for ii=1:ClassNum
    L(i,ii)=length(find(Children(i,1).Child_Class==ii))/CL(1,i);
    end
end
EL=isempty(L);
if EL==1
    EL=0;
end
% children entropy
En=-L.*log2(L);
Child_En=(CL/R)'.*nansum(En,2);
 % parent entropy
if length(ClassNum)==1
    k=R/ClassNum;
    Parent_En=-(k/R)*log2((k/R))*ClassNum;
else
Parent_En=sum(-(ClassNum/R).*log2(ClassNum/R));
end
% information Gain
IG=Parent_En-sum(Child_En);


end
%% ||||||||||||||||||||||||||||||||||sub function
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


