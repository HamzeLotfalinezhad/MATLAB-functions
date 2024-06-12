function FD_B=Box_FDI(BW)

% BW=gifimage;
% BW=im2bw(BW,0.5);
% BW=~BW;

k=1;
BS0=length(BW);
%  imshow(BW)
BW0=ones(length(BW));
sl=0;
for i=0:1000000000000000
    sl=2.^i;
    if sl>=length(BW)
        break
    end
    
    BC=0;
    BS=fix(BS0/sl);
%     subplot(3,3,k)
    for r=1:sl
    for c=1:sl
        y=BW(BW(  1+r*BS-BS:r*BS, 1+c*BS-BS:c*BS )<1);
    if 0==isempty(y);
        BC=BC+1;
%% plot \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ 
% bwim=BW(  1+r*BS-BS:r*BS, 1+c*BS-BS:c*BS );
% for o=1:length(bwim)
%     for or=1:length(bwim)
%         if bwim(o,or)==1;
%             BW0(  o+r*BS-BS:r*BS, or+c*BS-BS:c*BS )=0.5;
%         end
%     end
% end
%%    
    end
    end
    end
    
    BoxSize1(k)=BS;
    BoxNum1(k)=BC;
    k=k+1;
%     imshow(BW0)
end
% cc=imfuse(BW,BW0);imshow(cc)
% imshow((BW+BW0));
%%
% a=(1:50);
% BoxSize1=BoxSize(a);
% BoxNum1=BoxNum(a);

figure
plot(-log(BoxSize1),log(BoxNum1),'-ob','linewidth',1,'markersize',3,'markerfacecolor','b')
p=polyfit(-log(BoxSize1),log(BoxNum1),1);
FD_B=p(1);
f=polyval(p,-log(BoxSize1));
hold on;
plot(-log(BoxSize1),f,'r','linewidth',2)
xlabel('ln (1/L)')
ylabel('ln (N)')
grid on
end