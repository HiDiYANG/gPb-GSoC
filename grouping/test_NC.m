load mPb.mat
mPb = imresize(mPb, 0.5);
mPb = max(0, min(1, 1.2*mPb));


[ mmax nmax ] = size(mPb);

%Transform image into data for N-cuts
[indx, indy] = meshgrid(1:nmax,1:mmax);
data = [reshape(indx,[],1), reshape(indy,[],1), reshape(mPb,[],1)];

%Normalise the data
original_data = data;
maxdata = max(data(:));
data = data/maxdata;

%Calculate the affinity mnatrix 'W' for the data
W = CalculateAffinity_greyscale(data);
%figure,imshow(W,[]), title('Affinity Matrix');
 
% compute the degree matrix D 
N = length(W);
d = sum(W, 2);
D = spdiags(d, 0, N, N); % diagonal matrix


% compute the unnormalized graph laplacian matrix

opts.issym=1;
opts.isreal = 1;
opts.disp=2;
%Find the 2 smallest eigenvectors and corresponding values
[eigVectors,eigValues] = eigs(D - W, D, 17, 'sm', opts );

%save the 2nd smallest eigenvector
U2 = eigVectors(:, 1);


% plot the eigen vector corresponding to the 2nd smallest eigen value
figure,plot( U2,'r*'),title('2nd Smallest Eigenvector');

%Manually set the threshold to parition clusters based on the second smallest eigenvector
[xx1,yy1,val1] = find(U2 > 0);
[xx2,yy2,val2] = find(U2 <= 0 );

%Plot the resulting segmentation
figure,
hold on;
plot(data(xx1,1),data(xx1,2),'r*');
plot(data(xx2,1),data(xx2,2),'b*');
hold off;
title('Clustering Results using the 2nd smallest Eigen Vector');
grid on;shg

segmented_image = zeros(mmax, nmax);
for i=1:length(xx1)
    segmented_image(original_data(xx1(i),2),original_data(xx1(i),1)) = 1;
end
for i=1:length(xx2)
    segmented_image(original_data(xx2(i),2),original_data(xx2(i),1)) = 2;
end

figure;
imagesc(segmented_image);





