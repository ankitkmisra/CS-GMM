function [reconstructed_img, RMSE, PSNR] = inpainting(img_name, p, save_name)
mkdir 190070020_190050020_inpainting_results;
path = "190070020_190050020_inpainting_results/";
K = 19;
% Get image in grayscale
img = imread(img_name);
dim = size(size(img));
if (dim(2) == 3),
    img = rgb2gray(img);
end
img = double(img);
img = imresize(img, [128, 128]);
imshow(img, []);
% Store uncorrupted original image
title = path + "inpainting_original_p_" + sprintf("%f", p) + "_" + save_name;
saveas(gcf, title);
figure;
[n, m] = size(img);
sN = 8;
% If p is low, increase patch size
if(p < 0.5),
    sN = 12;
end
N = sN * sN;
% Get mask having about p fraction of pixels known
mask = rand(n, m);
mask(mask > p) = 0;
mask(mask <= p) = 1;
frac = sum(sum(mask)) / (n * m);
while(~((p - 0.1 <= frac) && (frac <= p + 0.1)))
    mask = rand(n,m);
    pos = mask <= p;
    mask(mask > p) = 0;
    mask(pos) = 1;
    frac = sum(sum(mask)) / (n * m);
end
sigma = 3;
% Get corrupted image
y = (mask .* img) + (sigma * randn(n, m));
imshow(y, []);
% Store corrupted image
title = path + "inpainting_corrupted_p_" + sprintf("%f", p) + "_" + save_name;
saveas(gcf, title);
figure;
f = reshape(img, [], 1);
% Initialise required variables
reconstruction = zeros(n, m);
count = zeros(n, m);
mean = 255 * zeros(K, N);
new_mean = zeros(K, N);
cov = 50 * rand(K, N, N);

%% Get directional basis
for k = 1:K,
    cov(k, :, :) = directional_initialise(10 * (k - 1) * pi / 180, sN, 5000);
end
Det = zeros([K, 1]);
C_inv = zeros(K, N, N);
cluster = zeros(n, m);
elem = zeros([K, 1]);
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
                filter = (pinv(Ut_U + (sigma * sigma * reshape(C_inv(k, :, :), N, N))));
                f_pred(k,:) = reshape(filter * ((U' * r) + (sigma * sigma * reshape(C_inv(k,:,:), N, N) * reshape(mean(k,:),[],1))), [], 1);
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
    elem = zeros([K, 1]);
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
    imshow(reconstruction, []);
    figure;
    last_reconstruction = reconstruction;
    cost = new_cost;
end
imshow(last_reconstruction, []);
figure;
title = path + "inpainting_reconstructed_p_" + sprintf("%f", p) + "_" + save_name;
saveas(gcf, title);
% Convert both images to 0-255 range
last_reconstruction = 255 * last_reconstruction / max(max(last_reconstruction));
img = 255 * img / max(max(img));
% Calculate MSE, RMSE, PSNR
mse = (norm(last_reconstruction(:) - img(:), 2) ^ 2) / (n * m);
rmse = (norm(last_reconstruction(:) - img(:), 2)) / norm(img(:), 2);
psnr = (20 * log(255)) - (10 * log(mse));
display(img_name);
display(p);
display(rmse);
display(psnr);
reconstructed_img = last_reconstruction;
save(save_name + ".mat", "last_reconstruction");
RMSE = rmse;
PSNR = psnr;
end