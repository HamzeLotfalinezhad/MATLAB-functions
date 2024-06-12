function FD_B=Box_FD(x)
%%
L=length(x); % size of the boxes


%%
y=NORM01(x);
x=1/length(x):1/length(x):1;
%% plot
% L=0.05;
% sp=0;
% c=0:L:1;
% col=[0.5 0.5 0.5];
% for j=1:1/L
% fy=find((j-1)*L<=x & x<=(j)*L);
%         xx=find(min(y(fy))<c & c<max(y(fy)));
%        fract1=min(y(fy))-floor(min(y(fy)));
%        fract2=max(y(fy))-floor(max(y(fy)));
%         if fract1>0.5 %|| fract2<0.5 
%     rectangle('Position',[sp,c(min(xx)),L,c(max(xx))-c(min(xx))],'facecolor',col)
%         else
%     rectangle('Position',[sp,c(min(xx))-L,L,c(max(xx))-c(min(xx))],'facecolor',col)            
%         end
%         
%          if fract2>0.5 || max(y(fy))==1
%     rectangle('Position',[sp,c(min(xx)),L,c(max(xx))-c(min(xx))+L],'facecolor',col)
%         else
%     rectangle('Position',[sp,c(min(xx)),L,c(max(xx))-c(min(xx))],'facecolor',col)            
%         end
%         
%     sp=sp+L;
% end
%     hold on;plot(x,y,'-y','linewidth',1,'markersize',1,'markerfacecolor','r','markeredgecolor','r')
%     for i=0:L:1
%     line([0 1],[i i],'color',[0 0 0]);hold on
%     line([i i],[0 1],'color',[0 0 0]);hold on
%     end
% axis([0 1 0 1])
%%

k=0;
for ii=0:100000000000000;%i=2:2:length(x)/4;
%     i=ii;
    i=2.^ii;
    if i>=length(y)
        break
    end
k=k+1;
L=1;
L=L/i;
    for j=1:1/L
        fy=find((j-1)*L<=x & x<=(j)*L);
        my=abs(max(y(fy))-min(y(fy)));
        NumN(j)=ceil(my/L); 
    end
        BoxN(k)=sum(NumN);
        BoxL(k)=(L);
end
%%

p=polyfit(-log(BoxL),log(BoxN),1);
FD_B=p(1);

if nargout==0
    f=polyval(p,-log(BoxL));
plot(-log(BoxL),log(BoxN),'-ob','markersize',3,'markerfacecolor','b')
hold on;
plot(-log(BoxL),f,'r','linewidth',2)
xlabel('ln (1/L)')
ylabel('ln (N)')
grid on
end
end