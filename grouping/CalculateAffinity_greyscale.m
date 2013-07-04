function [affinity] = CalculateAffinity_greyscale(data)

% set the parameters
sigmaxy = 150;
sigmai  = 0.1;

% calculate the affinity based on spatial and intensity parameters
for i=1:size(data,1)    
    for j=1:size(data,1)
        distxy = ((data(i,1) - data(j,1))^2 + (data(i,2) - data(j,2))^2 );
        disti = ((data(i,3) - data(j,3))^2 );
        affinity(i,j) = exp(-distxy/(2*sigmaxy^2))*exp(-disti/(2*sigmai^2));
    end
end


