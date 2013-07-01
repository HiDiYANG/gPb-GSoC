function display_img(A, w, h)

for i = 1:8
    I = A(:,:,i);
    subplot(w,h,i), imshow(I,[]);
    disp(['max[', num2str(i),'] = ', num2str(max(I(:)))]);
end