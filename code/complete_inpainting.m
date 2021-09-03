clc;
clear;
close all;
img_names = ["caman.tif", "lena.tif", "peppers.tif"];
save_names = ["caman.png", "lena.png", "peppers.png"];
rec_names = ["caman.mat", "lena.mat", "peppers.mat"];
prob = [0.8, 0.5, 0.2];
RM = [];
PS = [];
for ext_counter = 1:3,
    [rec, rm, ps] = inpainting( img_names(ext_counter), prob(ext_counter), save_names(ext_counter) );
    save(rec_names(ext_counter), "rec");
    display(rm);
    display(ps);
    RM = [RM, rm];
    PS = [PS, ps];
end;

display(RM);
display(PS);