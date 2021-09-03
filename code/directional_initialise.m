function [COV] = directional_initialise(theta, sN, samples)
N = sN * sN;
slope = tan(theta); % x and y coordinate systems are reversed here
P = samples;
C = zeros(N, N);
PATCHES = zeros(sN * sN, P);
mean = zeros([sN * sN, 1]);

for count = 1:P,
    y0 = randi([1, sN]);
    x0 = randi([1, sN]);
    patch = 255 * ones([sN, sN]);
    % Store edge patch in variable patch
    for y = 1:sN,
        x = ((y - y0) * slope) + x0;
        if(x <= sN),
            x = int64(max(1,x));
            patch(x:sN, y) = 0;
        end
    end
    % Store patch
    PATCHES(:,count) = reshape(patch, [], 1);
    mean = mean + reshape(patch, [], 1);
end
mean = mean / P;
for count = 1:P,
    % Centre patch
    PATCHES(:, count) = PATCHES(:, count) - mean;
end
% Calculate covariance matrix
C = (PATCHES * PATCHES') / P;
% While not full rank, keep on epsilon I to C
while(rank(C) ~= N),
    C = C + (max(max(C)) / 1000 * eye(N));
end
M = PATCHES' * PATCHES / P;
[V, D] = eigs(C, sN);
%% Show some patches
% for count = 1:min(sN,8),
%     mat = reshape(V(:,count), sN, sN);
%     mat = mat - min(min(mat));
%     mat = 255 * mat / max(max(mat));
%     imshow(mat, []);
%     figure;
% end

COV = C;
end