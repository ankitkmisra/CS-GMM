function [COV] = positional_initialise(theta, sN, blurs_per_sample, x0, y0, hi)
N = sN * sN;
slope = tan(theta); % x and y coordinate systems are reversed here
%P = max(1, samples);
n = blurs_per_sample;
C = zeros(N, N);
%PATCHES = zeros(sN * sN, P);
mean = zeros([sN * sN, 1]);

patch = 255 * ones([sN, sN]);
for y = 1:sN,
    x = ((y - y0) * slope) + x0;
    if(x <= sN),
        x = int64(max(1,x));
        patch(x:sN, y) = 0;
    end
end
    
for count = 1:n,
    blurred_patch = imgaussfilt(patch, hi * rand());
    PATCHES(:,count) = reshape(blurred_patch, [], 1);
    mean = mean + reshape(blurred_patch, [], 1);
end

mean = mean / n;
for count = 1:n,
    PATCHES(:, count) = PATCHES(:, count) - mean;
end
C = (PATCHES * PATCHES') / n;
while(rank(C) ~= N),
    C = C + (max(30,max(max(C)) / 1000) * eye(N));
end
M = PATCHES' * PATCHES / n;
[V, D] = eigs(C, sN);

for count = 1:min(sN,6),
    mat = reshape(V(:,count), sN, sN);
    mat = mat - min(min(mat));
    mat = 255 * mat / max(max(mat));
%     imshow(mat, []);
%     figure;
end

COV = C;
end