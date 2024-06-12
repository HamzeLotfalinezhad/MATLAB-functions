function MSE=MS_PermEn(x,MAXscale,order)

L=length(x);

%%
k=0;
for scale=1:MAXscale
k=k+1;
    j=fix(L/scale);

ys=reshape(x(1:scale*j),scale,j)';
my=mean(ys,2);
if length(my)<2
    break
end
y=my';
%% now you should calculate ENTROPY of each rows of "y"sepreatly
z=PermEn(y,order);
% [z ~]=ShannonEn(y,1,10);

MSE(k)=z;

end
if nargout==0
plot(MSE,'--rs','LineWidth',2,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','g',...
                'MarkerSize',5)
            title('MS PermEn')
            xlabel('Scales')
            ylabel('Permutation Entropy')
end
end