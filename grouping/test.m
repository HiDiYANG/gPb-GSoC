clc;
clear all;
close all;

% imgFile = 'data/101087.jpg';
% I = imread(imgFile);
% I = im2double(I);
% I = I.^(2.2);
% [L, a, b] = RGB2Lab(I(:,:,1), I(:,:,2), I(:,:,3));
% L = L./100;
% L(L<0) = 0;
% L(L>1) = 1;
% a = (a+73)./168;
% a(a<0) = 0;
% a(a>1) = 1;
% b = (b+73)./168;
% b(b<0) = 0;
% b(b>1) = 1;
% 
% L = floor(L.*25);
% a = floor(a.*25);
% b = floor(b.*25);
% L(L==25) = 24;
% a(a==25) = 24;
% b(b==25) = 24;

II = double(imread('source/opencv_gpb/src/Lab.png'));
L2 = II(:,:,3);
a2 = II(:,:,2);
b2 = II(:,:,1);

r = 3;
weight = zeros(2*r+1, 2*r+1);
for i = -r:r
    for j = -r:r
        rad = i^2+j^2;
        if rad <= r^2
            weight(i+r+1, j+r+1) = 1;
        end
    end
end
weight(r+1, r+1) = 0;

ang = zeros(2*r+1, 2*r+1);
for i = -r:r
    for j = -r:r
        angle = atan2(-i, j)/pi*180;
        if angle == -180
            angle = 180;
        end
        ang(i+r+1, j+r+1) = angle;
    end
end

ori = 0:7;
angs = ori'*180/8;
angs = [angs-180, angs];

[row, col] = size(L2);
sigma = 0.1;
st = floor(3*sigma*25);
g = normpdf(-st:st,0,sigma*25);
g = g./sum(g);

cont = 1;
for n = 1:1%size(angs,1)
J = zeros(row,col);
for i=r+1:row-r
    for j=r+1:col-r
        hist_r = zeros(1, 25);
        hist_l = zeros(1, 25);
        for x = -r:r
            for y = -r:r
                bin = L2(i+x,j+y)+1;
                if ang(x+r+1,y+r+1) > angs(n,1) && ang(x+r+1, y+r+1) <= angs(n,2)
                    hist_r(bin) = hist_r(bin) + weight(x+r+1, y+r+1);
                else
                    hist_l(bin) = hist_l(bin) + weight(x+r+1, y+r+1);
                end
            end
        end
         
        hist_r = conv(hist_r, g, 'same');
        hist_l = conv(hist_l, g, 'same');
        
        
        
        sum_r = sum(hist_r);
        sum_l = sum(hist_l);
        if sum_r ~= 0
           hist_r = hist_r./sum_r;
        end
        if sum_l ~= 0
           hist_l = hist_l./sum_l;
        end
        
        temp = hist_r+hist_l;
        temp(temp == 0) = 1;
     
        J(i,j)=sum(0.5*(hist_r - hist_l).^2./temp);
        
        HIST_R(cont,:) = hist_r - hist_l;
        HIST_L(cont,:) = temp;
        cont =cont+1;
        
    end
end
RR{n} = J;
figure,imshow(J,[]);
end        



% hist_l = dlmread('source/opencv_gpb/src/hist_left.txt');
% hist_r = dlmread('source/opencv_gpb/src/hist_right.txt');
% 
% for i=1: size(hist_l,1)
%     sum_r = sum(hist_r(i,:));
%     sum_l = sum(hist_l(i,:));
%     if sum_r ~= 0
%         hist_r(i,:) = hist_r(i,:)./sum_r;
%     end
%     if sum_l ~= 0
%         hist_l(i,:) = hist_l(i,:)./sum_l;
%     end
%     temp = hist_r(i,:)+hist_l(i,:);
%     temp(temp == 0) = 1;
%     
%     
%     
%     NNN(i)=sum(0.5*(hist_r(i,:) - hist_l(i,:)).^2./temp);
% end
        
        
        
        
        



        