function B = changes(A)

for i = 1:8
    I = A{i};
    B(:,:,i)=I;
end