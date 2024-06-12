function RecurencePlot(x,dimention,delay,threshold)
Y=psr_deneme(x,dimention,delay)';
DM=dist(Y); % DM: distance matrix

% rescaling mean distance or maximum distance
DM=DM./max(max(DM)); % MAX distance rescaling
% DM=DM./mean(mean(DM)); % MAX distance rescaling

if nargin==4
    dx1=DM<threshold;
imagesc(dx1)
% colorbar; axis square;
colormap([1 1 1; 0 0 0]); axis square;
else
    imagesc(DM)
% colorbar; 
axis square;
end
xlim([0,size(DM,2)]);ylim([0,size(DM,1)]); 
    xlabel('Time','FontSize',8,'FontWeight','bold');ylabel('Time ','FontSize',8,'FontWeight','bold');
    title('Recurrence Plot','FontSize',10,'FontWeight','bold');
    get(gcf,'CurrentAxes');
%     set(gca,'Visible','off');
    set(gca,'YDir','normal') % this flip the plot of RP
    set(gca,'LineWidth',2,'FontSize',10,'FontWeight','bold');
end