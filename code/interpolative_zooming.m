clc;
clear;
close all;
K = 19;
img_name = 'lena.tif';
save_name = 'lena.png';
img = im2gray(imread(img_name));
img = double(img);
img = imresize(img, [192, 192]);
imshow(img, []);
title = "zoom_original_" + save_name;
saveas(gcf, title);
figure;
[n, m] = size(img);
sN = 8;
N = sN * sN;
small_img = img(1:2:end, 1:2:end);
imshow(small_img, []);
title = "zoom_downsampled_" + save_name;
saveas(gcf, title);
figure;
sigma = 3;
y = (sigma * randn(n, m));
mask = zeros(n,m);
for i = 1:n,
    for j = 1:m,
        if(mod(i,2) == 1 && mod(j,2) == 1),
            y(i,j) = y(i,j) + img(i,j);
            mask(i,j) = 1;
        end
    end
end
imshow(y, []);
title = "zoom_uninterpolated_enlarged_" + save_name;
saveas(gcf, title);
figure;
f = reshape(img, [], 1);
reconstruction = zeros(n, m);
count = zeros(n, m);
mean = 255 * rand(K, N);
new_mean = zeros(K, N);
cov = 50 * rand(K, N, N);
for k = 1:K,
    cov(k, :, :) = directional_initialise(10 * (k - 1) * pi / 180, sN, 500);
end
Det = zeros([K, 1]);
C_inv = zeros(K, N, N);
cluster = zeros(n, m);
elem = zeros(K);
pred = zeros(n, m, N);
cost = Inf;
last_reconstruction = y;
new_cost = 0;
reconstruction = zeros(n, m);
    

for i = 1:K,
    C_inv(i,:,:) = pinv(reshape(cov(i,:,:), N, N));
    Det(i) = log(det(reshape(cov(i,:,:), N, N)));
    
end
for i = 1:n - sN + 1;
    for j = 1:m - sN + 1,
        count(i:i+sN-1, j:j+sN-1) = count(i:i+sN-1, j:j+sN-1) + 1;
    end
end


for loop = 1:8,
    new_cost = 0;
    reconstruction = zeros(n, m);
    %% E step
    
    for i = 1:n-sN+1;
        for j = 1:m-sN+1,
            U = zeros(N, N);
            for l = 1:sN,
                for q = 1:sN,
                    ind = l + ((q - 1) * sN);
                    U(ind, ind) = mask(i+l-1, j+q-1);
                end
            end
            r = reshape(y(i:i+sN-1, j:j+sN-1), [], 1);
            f_pred = zeros(K, N);
            Ut_U = U' * U;
            for k = 1:K,
                filter = (Ut_U + (sigma * sigma * reshape(C_inv(k, :, :), N, N)));
                f_pred(k,:) = reshape(filter \ ((U' * r) + (sigma * sigma * reshape(C_inv(k,:,:), N, N) * reshape(mean(k,:),[],1))), [], 1);
            end
            k = 1;
            vec = reshape(f_pred(k,:), [], 1) - reshape(mean(k, :), [], 1);
            J = (norm((U * reshape(f_pred(k,:), [], 1)) - r) ^ 2) + (sigma * sigma * vec' * reshape(C_inv(k,:,:), N, N) * vec) + (sigma * sigma * Det(k));
            ind = 1;
            for k = 2:K,
                vec = reshape(f_pred(k,:), [], 1) - reshape(mean(k, :), [], 1);
                J_ = (norm((U * reshape(f_pred(k,:), [], 1)) - r) ^ 2) + (vec' * reshape(C_inv(k,:,:), N, N) * vec) + (sigma * sigma * Det(k));
                if(~isnan(J_) && ~isinf(J_) && J_ < J),
                    ind = k;
                    J = J_;
                end
            end
            new_cost = new_cost + J;
            %f_pred(ind,:) = reshape(f_pred(ind,:),[],1) + reshape(mean(ind,:),[],1);
            reconstruction(i:i+sN-1, j:j+sN-1) = reconstruction(i:i+sN-1, j:j+sN-1) + reshape(f_pred(ind,:), sN, sN);
            cluster(i,j) = ind;
            pred(i,j,:) = f_pred(ind, :);
            new_mean(ind, :) = new_mean(ind, :) + f_pred(ind, :);
            elem(ind) = elem(ind) + 1;
        end
    end
    
    %% M step
    for i = 1:K,
        if(elem(i) > 0),
            mean(i, :) = new_mean(i, :) / elem(i);
        end
    end
    cov = zeros(K, N, N);
    for i = 1:n-sN+1;
        for j = 1:m-sN+1,
            vec = reshape(pred(i,j,:), [], 1) - reshape(mean(cluster(i,j), :), [], 1);
            cov(cluster(i,j), :, :) = reshape(cov(cluster(i,j), :, :), N, N) + (vec * vec');
        end
    end
    new_mean = zeros(K, N);
    cluster = zeros(n, m);
    elem = zeros(K);
    pred = zeros(n, m, N);
    
    for i = 1:K,
        if(elem(i) == 0),
            continue;
        end
        cov(i,:,:) = cov(i,:,:) / elem(i);
        if((sum(sum(sum(isnan(cov(i,:,:))))) > 0) || (sum(sum(sum(isinf(cov(i,:,:))))) > 0)),
            cov(i,:,:) = directional_initialise(10 * (i - 1) * pi / 180, sN, 2000);
        end
        while(rank(reshape(cov(i,:,:), N, N)) ~= N),
            cov(i,:,:) = reshape(cov(i,:,:), N, N) + (30 * eye(N) / 100);
            if((sum(sum(sum(isnan(cov(i,:,:))))) > 0) || (sum(sum(sum(isinf(cov(i,:,:))))) > 0)),
                cov(i,:,:) = directional_initialise(10 * (i - 1) * pi / 180, sN, 2000);
            end
        end
        C_inv(i,:,:) = pinv(reshape(cov(i,:,:), N, N));
        Det(i) = log(det(reshape(cov(i,:,:), N, N)));
    end
    epsilon = 0.1;
    if(isinf(new_cost) || (new_cost > cost) || ((cost - new_cost) / cost)  <= epsilon),
        break;
    end
        
    reconstruction = reconstruction ./ count;
    reconstruction = reconstruction - min(min(reconstruction));
    reconstruction = 255 * reconstruction / max(max(reconstruction));
    last_reconstruction = reconstruction;
    cost = new_cost;
end

imshow(last_reconstruction, []);
title = "zoom_interpolated_reconstructed_" + save_name;
saveas(gcf, title);

rmse = norm(last_reconstruction-img) / norm(img);
mse = norm(last_reconstruction-img) / (m*n);
psnr = 10 * log10(255*255 / mse);
fprintf("RMSE = %f\n", rmse);
fprintf("PSNR = %f\n", psnr);

%end