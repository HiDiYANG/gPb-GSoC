function [affinity] = CalculateAffinity_greyscale(data)

% set the parameters
sigmaxy = 150;
sigmai  = 0.1;

[rows, cols] = size(data);

for i = 1:rows
    for j = 1:cols
        map = zeros(1,25);
        c = 0;
        for n = -2:2
            for m = -2:2
                indx = i+n;
                indy = j+m;
                if( (indx >= 1 && indx <=rows)&&(indy >=1 && indy <=cols) ) 
                    map(c) = data(indx, indy);
                    c = c+1;
                end

% calculate the affinity based on spatial and intensity parameters
for i=1:size(data,1)    
    for j=1:size(data,1)
        distxy = ((data(i,1) - data(j,1))^2 + (data(i,2) - data(j,2))^2 );
        if distxy <=25
            affinity(i,j) = exp(-max(data(i,3), data(j,3))/0.1);
        else
            affinity(i,j) = 0;
        end
    end
end