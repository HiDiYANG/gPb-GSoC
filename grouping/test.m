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

RR = zeros(row, col, size(angs,1));
cont = 1;
for n = 1:size(angs,1)
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
        
        
        
%         sum_r = sum(hist_r);
%         sum_l = sum(hist_l);
%         if sum_r ~= 0
%            hist_r = hist_r./sum_r;
%         end
%         if sum_l ~= 0
%            hist_l = hist_l./sum_l;
%         end
        
        temp = hist_r+hist_l;
        temp(temp == 0) = 1;
     
        J(i,j)=sum(0.5*(hist_r - hist_l).^2./temp);
        
        HIST_R(cont,:) = hist_r - hist_l;
        HIST_L(cont,:) = temp;
        cont =cont+1;
        
    end
end
RR(:,:,n) = J;
figure,imshow(J,[]);
end        

bbg1 = dlmread('./source/opencv_gpb/src/bg1.txt');
bbg2 = dlmread('./source/opencv_gpb/src/bg2.txt');
bbg3 = dlmread('./source/opencv_gpb/src/bg3.txt');

ccga1 = dlmread('./source/opencv_gpb/src/cga1.txt');
ccga2 = dlmread('./source/opencv_gpb/src/cga2.txt');
ccga3 = dlmread('./source/opencv_gpb/src/cga3.txt');

ccgb1 = dlmread('./source/opencv_gpb/src/cgb1.txt');
ccgb2 = dlmread('./source/opencv_gpb/src/cgb2.txt');
ccgb3 = dlmread('./source/opencv_gpb/src/cgb3.txt');

ttg1 = dlmread('./source/opencv_gpb/src/tg1.txt');
ttg2 = dlmread('./source/opencv_gpb/src/tg2.txt');
ttg3 = dlmread('./source/opencv_gpb/src/tg3.txt');


mPb_all = weights(1)*bbg1 + weights(2)*bbg2 + weights(1)*bbg3 + ...
          weights(4)*ccga1 + weights(5)*ccga2 + weights(6)*ccga3 + ...
          weights(7)*ccgb1 + weights(8)*ccgb2 + weights(9)*ccgb3 + ...
          weights(10)*ttg1 + weights(11)*ttg2 + weights(12)*ttg3;
 











bg_1 = changes(bg_r3);
bg_2 = changes(bg_r5);
bg_3 = changes(bg_r10);

cga_1 = changes(cga_r5);
cga_2 = changes(cga_r10);
cga_3 = changes(cga_r20);

cgb_1 = changes(cgb_r5);
cgb_2 = changes(cgb_r10);
cgb_3 = changes(cgb_r20);

tg_1 = changes(tg_r5);
tg_2 = changes(tg_r10);
tg_3 = changes(tg_r20);

% figure, display_img(bg_r3, 2, 4);
% figure, display_img(bg_r5, 1, 8);
% figure, display_img(bg_r10, 1, 8);
% figure, display_img(cga_r5, 1, 8);
% figure, display_img(cga_r10, 1, 8);
% figure, display_img(cga_r20, 1, 8);
% figure, display_img(cgb_r5, 1, 8);
% figure, display_img(cgb_r10, 1, 8);
% figure, display_img(cgb_r20, 1, 8);
% figure, display_img(tg_r5, 1, 8);
% figure, display_img(tg_r10, 1, 8);
% figure, display_img(tg_r20, 1, 8);


        
        
        
        
        



        