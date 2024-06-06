%% Clear Variables
clearvars;
clc;
close all;
rng(1)

%% Oversegmentation: 
image_no = 1;
[labels, numlabels] = Oversegmentation(image_no,"slicmex",200,10);
img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
BW = boundarymask(labels);
overlaidImg = imoverlay(img, BW, 'red');
figure;
imagesc(overlaidImg); 

%% Extracting Gabor Features of All Superpixels in All Images: 
alg = "slicmex";
Ni = 200;
compactness = 10; 
resolutions = [2 4 8 16];
angles = [0 30 60 90];
[features_all,labels_all] = extractAll(alg,Ni,compactness,resolutions,angles);
%% K-Means for Gabor Features
k = 5;
newLabels = kmeansimg(k,features_all,labels_all);
%% Display All images generated with this algorithm 
figure('Position', [100, 300, 1500, 600]);
for i = 1:10
    subplot(2,5,i);
    m = i;
    bgImage = imread(sprintf('HW/CS448 HW3/data/%i.jpg',m));
    BW = boundarymask(newLabels{m});
    overlaidImg = imoverlay(bgImage, BW, 'red');
    imagesc(overlaidImg); 
    hold on; 
    h = imagesc(newLabels{m}); 
    transparency = 0.4; 
    alpha(h, transparency);  
    axis off;  
end
%% Display All boundary images generated with this algorithm 
figure('Position', [100, 300, 1500, 600]);
for i = 1:10
    subplot(2,5,i);
    m = i;
    img = imread(sprintf('HW/CS448 HW3/data/%i.jpg',m));
    BW = boundarymask(newLabels{i});
    overlaidImg = imoverlay(img, BW, 'red');
    imagesc(overlaidImg); 
    axis off;  
end
%% Extending the Features
threshold = 1000;
[extendedFeatures,labels_all] = extendFeatures(features_all,labels_all,threshold,Ni,2);
%% K-Means for Extended Features
k = 3;
newLabels = kmeansimg(k,extendedFeatures,labels_all);
%% Display All images generated with this algorithm 
figure('Position', [100, 300, 1500, 600]);
for i = 1:10
    subplot(2,5,i);
    m = i;
    bgImage = imread(sprintf('HW/CS448 HW3/data/%i.jpg',m));
    BW = boundarymask(newLabels{m});
    overlaidImg = imoverlay(bgImage, BW, 'red');
    imagesc(overlaidImg); 
    hold on; 
    h = imagesc(newLabels{m}); 
    transparency = 0.5; 
    alpha(h, transparency);  
    axis off;  
end
%% Display All boundary images generated with this algorithm 
figure('Position', [100, 300, 1500, 600]);
for i = 1:10
    subplot(2,5,i);
    m = i;
    img = imread(sprintf('HW/CS448 HW3/data/%i.jpg',m));
    BW = boundarymask(newLabels{i});
    overlaidImg = imoverlay(img, BW, 'red');
    imagesc(overlaidImg); 
    axis off;  
end
%% Generating plots: Intro 
N = 1001;
impulseImage = zeros(N, N);
centerIndex = (N + 1) / 2;
impulseImage(centerIndex, centerIndex) = 1;

wavelengths = [16, 32, 64, 128];  % Example wavelengths for different scales
orientations = [0, 45, 90, 135];  % Example orientations in degrees

gaborArray = gabor(wavelengths, orientations);
filtimages = imgaborfilt(impulseImage,gaborArray);
figure;

wavelengths = repmat([16, 32, 64, 128],[1 4]);  % Example wavelengths for different scales
orientations = repelem([0, 45, 90, 135],4); 
for j = 1:16
    subplot(4,4,j);
    imshow(filtimages(:,:,j));
    text = sprintf('$ \\lambda = %d, \\theta = %d$', wavelengths(j), orientations(j));
    title(text,'Interpreter','latex');
end 
%% Generating plots: Part 1
sp_params = 20*(5:19);
compactness_params = 3:17;
image_no = 8;
compact_trial = cell(1,15);
sp_trial = cell(1,15);
for i = 1:15
    [labels, numlabels] = Oversegmentation(image_no,"slicmex",200,compactness_params(i));
    img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
    BW = boundarymask(labels);
    overlaidImg = imoverlay(img, BW, 'red');
    compact_trial{i} = overlaidImg;
end

for i = 1:15
    [labels, numlabels] = Oversegmentation(image_no,"slicmex",sp_params(i),10); 
    img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
    BW = boundarymask(labels);
    overlaidImg = imoverlay(img, BW, 'red');
    sp_trial{i} = overlaidImg;
end

figure; 
for i = 1:15
    subplot(3,5,i);
    imagesc(sp_trial{i});
    text = sprintf('$ N_{SP}  = %i $', sp_params(i));
    title(text,'Interpreter','latex');
    axis off;  
end

figure; 
for i = 1:15
    subplot(3,5,i);
    imagesc(compact_trial{i});
    text = sprintf('$ C  = %i $', compactness_params(i));
    title(text,'Interpreter','latex');
    axis off;  
end

images = cell(10,2);
for i = 1:10
    [labels, numlabels] = Oversegmentation(i,"slicmex",200,10);
    img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",i));
    BW = boundarymask(labels);
    overlaidImg = imoverlay(img, BW, 'black');
    images{i,1} = overlaidImg;
    images{i,2} = labels;
end

figure; 
for i = 1:10
    subplot(2,5,i);
    imagesc(images{i,1});
    title('Image: ' + string(i) + '.jpg');
    hold on;
    h = imagesc(images{i,2}); 
    transparency = 0.3; 
    alpha(h, transparency);  
    axis off;  

end

%% Generating plots: Part 2
% Image 1
lambda = [2 4 6 8 16 32 64 128];
theta = (0:8)*15;
gaborArray = gabor(lambda,theta);
lambda_index = mod(1:72,8);
lambda_index(lambda_index == 0) = 8;
theta_index = ceil((1:72)/8);

%% Image 1
image_no = 1;
img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
grey_img = rgb2gray(img);
output = imgaborfilt(grey_img,gaborArray);
figure;
for i = 1:72
    subplot(9,8,i);
    imagesc(output(:,:,i));
    text = sprintf('$ \\lambda = %d, \\theta = %d$', lambda(lambda_index(i)), theta(theta_index(i)));
    title(text,'Interpreter','latex');
    axis off;
end
%% Image 4
image_no = 4;
img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
grey_img = rgb2gray(img);
output = imgaborfilt(grey_img,gaborArray);
figure;
for i = 1:72
    subplot(9,8,i);
    imagesc(output(:,:,i));
    text = sprintf('$ \\lambda = %d, \\theta = %d$', lambda(lambda_index(i)), theta(theta_index(i)));
    title(text,'Interpreter','latex');
    axis off;
end
%% Image 7
image_no = 7;
img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
grey_img = rgb2gray(img);
output = imgaborfilt(grey_img,gaborArray);
figure;
for i = 1:72
    subplot(9,8,i);
    imagesc(output(:,:,i));
    text = sprintf('$ \\lambda = %d, \\theta = %d$', lambda(lambda_index(i)), theta(theta_index(i)));
    title(text,'Interpreter','latex');
    axis off;
end
%% Image 10
image_no = 10;
img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
grey_img = rgb2gray(img);
output = imgaborfilt(grey_img,gaborArray);
figure;
for i = 1:72
    subplot(9,8,i);
    imagesc(output(:,:,i));
    text = sprintf('$ \\lambda = %d, \\theta = %d$', lambda(lambda_index(i)), theta(theta_index(i)));
    title(text,'Interpreter','latex');
    axis off;
end

%%

lambda = [2 4 8 16];
theta = [0 30 60 90];
gaborArray = gabor(lambda,theta);
lambda_index = mod(1:16,4);
lambda_index(lambda_index == 0) = 4;
theta_index = ceil((1:16)/4);

image_no = 2;
img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
grey_img = rgb2gray(img);
output = imgaborfilt(grey_img,gaborArray);
figure;
for i = 1:16
    subplot(4,4,i);
    imagesc(output(:,:,i));
    text = sprintf('$ \\lambda = %d, \\theta = %d$', lambda(lambda_index(i)), theta(theta_index(i)));
    title(text,'Interpreter','latex');
    axis off;
end

image_no = 3;
img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
grey_img = rgb2gray(img);
output = imgaborfilt(grey_img,gaborArray);
figure;
for i = 1:16
    subplot(4,4,i);
    imagesc(output(:,:,i));
    text = sprintf('$ \\lambda = %d, \\theta = %d$', lambda(lambda_index(i)), theta(theta_index(i)));
    title(text,'Interpreter','latex');
    axis off;
end

image_no = 5;
img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
grey_img = rgb2gray(img);
output = imgaborfilt(grey_img,gaborArray);
figure;
for i = 1:16
    subplot(4,4,i);
    imagesc(output(:,:,i));
    text = sprintf('$ \\lambda = %d, \\theta = %d$', lambda(lambda_index(i)), theta(theta_index(i)));
    title(text,'Interpreter','latex');
    axis off;
end
%% Generating Plots part 4
threshold = 1000;
mult = [2,3,4,5];
extendedFeatures = cell(1,4);
labels_all_n = cell(1,4);
k = 3;
for i = 1:4
    [extendedFeatures{i},labels_all] = extendFeatures(features_all,labels_all,threshold,Ni,mult(i));
    labels_all_n{i} = kmeansimg(k,extendedFeatures{i},labels_all);
end

%%
image_no = 3;
figure;
for i = 1:4
    img = imread(sprintf("HW/CS448 HW3/data/%i.jpg",image_no));
    BW = boundarymask(labels_all_n{i}{image_no});
    overlaidImg = imoverlay(img, BW, 'black');
    subplot(1,5,i);
    imagesc(overlaidImg);
    hold on;
    h = imagesc(labels_all_n{i}{image_no}); 
    transparency = 0.3; 
    alpha(h, transparency);  
    axis off;  
    text = sprintf('$r = %dr $', mult(i));
    title(text,'Interpreter','latex');
end
%%
%%==============================OVERSEGMENTATİON=======================================
function [labels, numlabels] = Oversegmentation(image_no,alg,Ni,compactness)
    path = sprintf("HW/CS448 HW3/data/%i.jpg",image_no);
    img = imread(path);
    if alg == "slicomex"
        [labels, numlabels] = slicomex(img,Ni);
    elseif alg == "slicmex"
        [labels, numlabels] = slicmex(img,Ni,compactness);
    else
        fprintf("Unidentified Algorithm !");
    end
end
%%==============================GABORFILTER=======================================
function [averages,labels] = GaborFilter(image_no,resolutions,angles,labels,num_labels)
    path = sprintf("HW/CS448 HW3/data/%i.jpg",image_no);
    img = imread(path);
    grey_img = rgb2gray(img);
    gaborArray = gabor(resolutions,angles);
    image_gabor = imgaborfilt(grey_img,gaborArray);
    num_features = length(resolutions)*length(angles);
    averages = zeros(num_labels, num_features);
    for label = 0:num_labels-1
        mask = labels == label;
        mask_expanded = repmat(mask, [1 1 num_features]);
        label_data = image_gabor .* mask_expanded;
        for feature = 1:num_features
            feature_data = label_data(:, :, feature);
            sum_feature = sum(feature_data(:));
            count_feature = sum(mask(:));
            if count_feature ~= 0
                averages(label+1, feature) = sum_feature / count_feature;
            end
        end
    end
end

%%==============================EXTRACTALL=======================================

function [averages_all, labels_all] = extractAll(alg, Ni, compactness, resolutions, angles)
    averagesFile = sprintf('HW/CS448 HW3/extracted_data/averages_all_Ni%i_comp%i_res[%i %i %i %i]_lambda[%i %i %i %i].mat',Ni,compactness,resolutions(1),resolutions(2),resolutions(3),resolutions(4),angles(1),angles(2),angles(3),angles(4));
    labelsFile = sprintf('HW/CS448 HW3/extracted_data/labels_all_Ni%i_comp%i_res[%i %i %i %i]_lambda[%i %i %i %i].mat',Ni,compactness,resolutions(1),resolutions(2),resolutions(3),resolutions(4),angles(1),angles(2),angles(3),angles(4));
    if exist(averagesFile, 'file') == 2 && exist(labelsFile, 'file') == 2
        disp('Loading existing data...');
        load(averagesFile, 'averages_all');
        load(labelsFile, 'labels_all');
    else
        disp('Creating new data...');
        averages_all = cell(1, 10);
        labels_all = cell(1, 10);
        cnt = 0;
        for i = 1:10
            [labels, numlabels] = Oversegmentation(i, alg, Ni, compactness);
            [averages_all{i}, labels_all{i}] = GaborFilter(i, resolutions, angles, labels, numlabels);
            cnt = cnt + 1;
            disp(string(cnt) + "/" + "10");
        end
        save(averagesFile, 'averages_all');
        save(labelsFile, 'labels_all');
    end
end
%%==============================FINDCENTERS=========================================
function centers = findCenters(labels)
    max_label = max(labels(:));
    centers = zeros(max_label, 2);
    for lbl = 0:max_label
        [row, col] = find(labels == lbl);
        centers(lbl+1, :) = [mean(row), mean(col)];
    end
end
%%==============================NBFEATURES=========================================

function features = neighbourfeatures(book_no,radii,min_thresh,labels,features_all)
    max_label = max(labels(:));
    features = zeros(max_label, 48);
    centers = findCenters(labels);
    for main_sp = 0:max_label
            neighbourhood1 = [];
            neighbourhood2 = [];
            for dum_sp = 0:max_label
                if dum_sp ~= main_sp
                    [row, col] = find(labels == dum_sp);
                    distances = sqrt((centers(main_sp+1, 1) - row).^2 + (centers(main_sp+1, 2) - col).^2);
                    area1 = sum(distances < 1.5*radii);
                    area2 = sum((distances > 1.5*radii)&(distances < 2.5*radii));
                    minPixels = min_thresh;
                    if area1 >= minPixels
                        neighbourhood1 = [neighbourhood1, dum_sp];
                    end
                    if area2 >= minPixels
                        neighbourhood2 = [neighbourhood2, dum_sp];
                    end
                end
            end
            
            gabor_features = features_all{book_no}(main_sp+1, :);
            
            N1Features = mean(features_all{book_no}(neighbourhood1+1, :), 1);
            N2Features = mean(features_all{book_no}(neighbourhood2+1, :), 1);
            features(main_sp + 1, :) = [gabor_features, N1Features, N2Features];
    end
end


%%==============================EXTENDFEATURES=======================================
function [extendedFeatures,labels_all] = extendFeatures(features_all,labels_all,min_thresh,Ni,mult)
    radii = sqrt(502*756/(Ni*pi))*mult;
    extendedFeatures = cell(1,10);
    cnt = 0;
    for i = 1:10
        disp(string(cnt) + "/" + "10")
        labels = labels_all{i};
        extendedFeatures{i} = neighbourfeatures(i,radii,min_thresh,labels,features_all);
        cnt = cnt + 1;
    end
    disp("10/10");
end

%%==============================KMEANSIMG=======================================

function newLabels = kmeansimg(k,averages_all,labels_all)
    allFeatures = [];
    for i = 1:length(averages_all)
        allFeatures = [allFeatures; averages_all{i}];
    end 
    [clusterIdx, ~] = kmeans(allFeatures, k);
    newLabels = cell(size(labels_all));
    startIndex = 1;
    
    for i = 1:length(averages_all)
        max_label = size(averages_all{i}, 1);
        currentIndices = clusterIdx(startIndex:startIndex + max_label - 1);
        uniqueLabels = unique(labels_all{i});
        labelMap = containers.Map(uniqueLabels, currentIndices);
        newlabels = labels_all{i};
        for j = 1:length(uniqueLabels)
            newlabels(labels_all{i} == uniqueLabels(j)) = labelMap(uniqueLabels(j));
        end 
        newLabels{i} = newlabels;   
        startIndex = startIndex + max_label;
    end
end
