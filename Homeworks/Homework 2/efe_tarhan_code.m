close all;
clearvars;
clc;
%% Plotting Canny Edge Detection Results 
book_names = {'algorithms.png','bitmemisoykuler.png','chess.png','cinali.png','cpp.png','datamining.png','harrypotter.png','heidi.png','kpss.png','lordofrings.png','patternrecognition.png', 'sefiller.png', 'shawshank.png','stephenking.png','ters.png'};
rbook_names = {'algorithmsR.png','bitmemisoykulerR.png','chessR.png','cinaliR.png','cppR.png','dataminingR.png','harrypotterR.png','heidiR.png','kpssR.png','lordofringsR.png','patternrecognitionR.png', 'sefillerR.png', 'shawshankR.png','stephenkingR.png','tersR.png'};

path = "HW/CS448 HW2/HW2_images/template images/";
pathR = "HW/CS448 HW2/HW2_images/rotated images/";
%% Canny Threshold Low 
canny_h = 0.4;
sigma = 2;

canny_l = linspace(0.01,0.3,16);
data = cell(1,16);
book_no = 10;
for i = 1:length(data)
    full_path = path + book_names{book_no};
    image = imread(full_path);
    greyimg = rgb2gray(image);
    data{i} = edge(greyimg,"canny",[canny_l(i) canny_h],sigma);
end

figure;
for m = 1:length(data)
    subplot(2,8,m)
    imshow(data{m})
    titleText = sprintf('T_L = %.2f', canny_l(m));
    title(titleText);
end

%% Canny Threshold High
canny_l = 0.1;
sigma = 2;

canny_h = linspace(0.12,0.8,16);
data = cell(1,16);
book_no = 10;
for i = 1:length(data)
    full_path = path + book_names{book_no};
    image = imread(full_path);
    greyimg = rgb2gray(image);
    data{i} = edge(greyimg,"canny",[canny_l canny_h(i)],sigma);
end

figure;
for m = 1:length(data)
    subplot(2,8,m)
    imshow(data{m})
    titleText = sprintf('T_H = %.2f', canny_h(m));
    title(titleText);
end

%% Canny Sigma 
canny_h = 0.2;
canny_l = 0.1;

sigma = linspace(1,4,16);
data = cell(1,16);
book_no = 10;
for i = 1:length(data)
    full_path = path + book_names{book_no};
    image = imread(full_path);
    greyimg = rgb2gray(image);
    data{i} = edge(greyimg,"canny",[canny_l canny_h],sigma(i));
end

figure;
for m = 1:length(data)
    subplot(2,8,m)
    imshow(data{m})
    titleText = sprintf('sigma = %.2f', sigma(m));
    title(titleText);
end

%% Canny of All Images
canny_h = 0.2;
canny_l = 0.1;
sigma = 2;

images = cell(1,15);
for i = 1:length(book_names)
    full_path = path + book_names{i};
    image = imread(full_path);
    greyimg = rgb2gray(image);
    images{i} = edge(greyimg,"canny",[canny_l canny_h],sigma);
end

figure;
for j = 1:length(rbook_names)
    subplot(3, 5, j);
    imshow(images{j});
    titleText = sprintf('%s', rbook_names{j});
    title(titleText);
end


images = cell(1,15);
for i = 1:length(rbook_names)
    full_path = pathR + rbook_names{i};
    image = imread(full_path);
    greyimg = rgb2gray(image);
    images{i} = edge(greyimg,"canny",[canny_l canny_h],sigma);
end

figure;
for j = 1:length(book_names)
    subplot(3, 5, j);
    imshow(images{j});
    titleText = sprintf('%s', book_names{j});
    title(titleText);
end

%% Plotting Hough Results 
canny_h = 0.2;
canny_l = 0.1;
sigma = 2;

%% Hough Maximum Number of Peaks
peak_ratio = 0.02;
fillgap = 6;
minlength = 8;

max_peak = floor(linspace(100,900,16));
data = cell(1,16);
book_no = 10;
for i = 1:length(data)
    [data{i}.line,data{i}.image] = extractLines(book_no,false,canny_l,canny_h,sigma,max_peak(i),peak_ratio,fillgap,minlength);
end

figure;
for m = 1:length(data)
    lines = data{m}.line;
    image = data{m}.image;
    subplot(2,8,m)
    imshow(image);
    hold on;
    titleText = sprintf('Max # of Peaks = %.2i', max_peak(m));
    title(titleText, 'Interpreter', 'latex');
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
    end
    hold off;

end
%% Hough Peak Ratio 
max_peak = 600;
fillgap = 6;
minlength = 8;

peak_ratio = linspace(0.01,0.2,16);
data = cell(1,16);
book_no = 10;

for i = 1:length(data)
    [data{i}.line,data{i}.image] = extractLines(book_no,false,canny_l,canny_h,sigma,max_peak,peak_ratio(i),fillgap,minlength);
end

figure;
for m = 1:length(data)
    lines = data{m}.line;
    image = data{m}.image;
    subplot(2,8,m)
    imshow(image);
    hold on;
    titleText = sprintf('Peak Ratio = %.2f', peak_ratio(m));
    title(titleText, 'Interpreter', 'latex');
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
    end
    hold off;
end
%% Hough FillGap 

max_peak = 600;
peak_ratio = 0.02;
minlength = 8;

fillgap = 1:16;
data = cell(1,16);
book_no = 10;

for i = 1:length(data)
    [data{i}.line,data{i}.image] = extractLines(book_no,false,canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap(i),minlength);
end

figure;
for m = 1:length(data)
    lines = data{m}.line;
    image = data{m}.image;
    subplot(2,8,m)
    imshow(image);
    hold on;
    titleText = sprintf('FillGap = %i', fillgap(m));
    title(titleText, 'Interpreter', 'latex');
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
    end
    hold off;

end

%% Hough MinLength

max_peak = 600;
peak_ratio = 0.02;
fillgap = 6;

minlength = 1:16;
data = cell(1,16);
book_no = 10;

for i = 1:length(data)
    [data{i}.line,data{i}.image] = extractLines(book_no,false,canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength(i));
end

figure;
for m = 1:length(data)
    lines = data{m}.line;
    image = data{m}.image;
    subplot(2,8,m)
    imshow(image);
    hold on;
    titleText = sprintf('MinLength = %i', minlength(m));
    title(titleText, 'Interpreter', 'latex');
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
    end
    hold off;
end
%% Hough of All Images 
max_peak = 600;
peak_ratio = 0.02;
fillgap = 6;
minlength = 8;

data = cell(1,15);
for i = 1:length(data)
    [lines,imgedge] = extractLines(i,false,canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength);
    data{i}.lines = lines;
    data{i}.image = imgedge;
end
figure;
for m = 1:length(data)
    lines = data{m}.lines;
    image = data{m}.image;
    subplot(3,5,m)
    imshow(image);
    hold on;
    title(book_names{m});
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
    end
    hold off;
end

data = cell(1,15);
for i = 1:length(data)
    [lines,imgedge] = extractLines(i,true,canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength);
    data{i}.lines = lines;
    data{i}.image = imgedge;
end
figure;
for m = 1:length(data)
    lines = data{m}.lines;
    image = data{m}.image;
    subplot(3,5,m)
    imshow(image);
    hold on;
    title(rbook_names{m});
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'green');
    end
    hold off;
end
%% Bin Count 
bin_count = ceil(linspace(20,180,16));
data = cell(1,16);
book_no = 3;
for i = 1:length(data)
    [data{i}.line,data{i}.image] = extractLines(book_no,false,canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength);
    [data{i}.binVals,data{i}.binEdges] = createHistogram(data{i}.line,bin_count(i));
end

figure;
for m = 1:length(data)
    subplot(2,8,m)
    bar(data{m}.binEdges(1:end-1),data{m}.binVals);
    titleText = sprintf('Bin Count = %i', bin_count(m));
    title(titleText, 'Interpreter', 'latex');
end

for i = 1:length(data)
    [data{i}.line,data{i}.image] = extractLines(book_no,true,canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength);
    [data{i}.binVals,data{i}.binEdges] = createHistogram(data{i}.line,bin_count(i));
end

figure;
for m = 1:length(data)
    subplot(2,8,m)
    bar(data{m}.binEdges(1:end-1),data{m}.binVals);
    titleText = sprintf('Bin Count = %i', bin_count(m));
    title(titleText, 'Interpreter', 'latex');
end
%% Bins of All Books
canny_l = 0.1;
canny_h = 0.2;
sigma = 2;
max_peak = 600;
peak_ratio = 0.02;
fillgap = 6;
minlength = 8;
bin_count = 60;

[templatehist,rotatehist,binEdges] = createAllHists(canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength,bin_count);

figure;
for j = 1:length(book_names)
    subplot(3, 5, j);
    bar(binEdges(2:end),templatehist{j});
    titleText = sprintf('%s', book_names{j});
    title(titleText);
end


figure;
for j = 1:length(rbook_names)
    subplot(3, 5, j);
    bar(binEdges(2:end),rotatehist{j});
    titleText = sprintf('%s', rbook_names{j});
    title(titleText);
end

%% HYPERPARAMETER DEFINITIONS
canny_l = 0.1;
canny_h = 0.2;
sigma = 2;
max_peak = 600;
peak_ratio = 0.02;
fillgap = 6;
minlength = 8;
bin_count = 60;

%% MATCHING OPERATIONS - Fully Correct

[templatehist,rotatehist,binEdges] = createAllHists(canny_l, canny_h, sigma, max_peak, peak_ratio, fillgap, minlength,bin_count);
[matches,truecount] = findMatches(templatehist,rotatehist,binEdges);
disp(matches)
fprintf("Number of correct matches is %i\n",truecount);
%% MATCHING OPERATIONS - 180 bins  
bin_count = 180;
[templatehist,rotatehist,binEdges] = createAllHists(canny_l, canny_h, sigma, max_peak, peak_ratio, fillgap, minlength,bin_count);
[matches,truecount] = findMatches(templatehist,rotatehist,binEdges);
disp(matches)
fprintf("Number of correct matches is %i\n",truecount);
bin_count = 60;
%% MATCHING OPERATIONS - 200 peaks
max_peak = 200;
[templatehist,rotatehist,binEdges] = createAllHists(canny_l, canny_h, sigma, max_peak, peak_ratio, fillgap, minlength,bin_count);
[matches,truecount] = findMatches(templatehist,rotatehist,binEdges);
disp(matches)
fprintf("Number of correct matches is %i\n",truecount);

%% FUNCTION DEFINITIONS

function [lines,imgedge] = extractLines(book_no,rotated,canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength)
    book_names = {'algorithms.png','bitmemisoykuler.png','chess.png','cinali.png','cpp.png','datamining.png','harrypotter.png','heidi.png','kpss.png','lordofrings.png','patternrecognition.png', 'sefiller.png', 'shawshank.png','stephenking.png','ters.png'};
    rbook_names = {'algorithmsR.png','bitmemisoykulerR.png','chessR.png','cinaliR.png','cppR.png','dataminingR.png','harrypotterR.png','heidiR.png','kpssR.png','lordofringsR.png','patternrecognitionR.png', 'sefillerR.png', 'shawshankR.png','stephenkingR.png','tersR.png'};
    template_path = "HW2_images/template images/";
    rotate_path = "HW2_images/rotated images/";
    if rotated
        full_path = rotate_path + rbook_names{book_no};
    else
        full_path = template_path + book_names{book_no};
    end
    image = imread(full_path);
    greyimg = rgb2gray(image);
    imgedge = edge(greyimg,"canny",[canny_l canny_h],sigma);
    [H,theta,rho] = hough(imgedge);
    peaks = houghpeaks(H, max_peak, 'threshold', peak_ratio*max(H(:)));
    lines = houghlines(imgedge, theta, rho, peaks, 'FillGap', fillgap, 'MinLength', minlength);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [binVals,binEdges] = createHistogram(lines,bin_count)
    binVals = zeros(1, bin_count);
    binEdges = linspace(0,180,bin_count+1);
    for i = 1:length(lines)
        linelen = norm(lines(i).point2-lines(i).point1);
        linang = (lines(i).theta)+90;
        %linang = normalizeTheta(lines(i).rho,lines(i).theta);
        binIndex = find(linang >= binEdges, 1, 'last');
        if linang == binEdges(end)
            binIndex = length(binEdges) - 1;
        end
        binVals(binIndex) = binVals(binIndex) + linelen;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [templatehist,rotatehist,binEdges] = createAllHists(canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength,bin_count)
    book_count = 15;
    templatehist = cell(1,book_count);
    rotatehist = cell(1,book_count);
    for i = 1:book_count
        [lines,~] = extractLines(i,false,canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength);
        [binVals,~] = createHistogram(lines,bin_count);
        templatehist{i} = binVals;

        [lines,~] = extractLines(i,true,canny_l,canny_h,sigma,max_peak,peak_ratio,fillgap,minlength);
        [binVals,binEdges] = createHistogram(lines,bin_count);
        rotatehist{i} = binVals;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rotatedArray = circularRotateRight(array, k)
    n = numel(array); 
    k = mod(k, n);
    rotatedArray = array(mod(-k:n-k-1, n) + 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [matches,truecount] = findMatches(templatehist,rotatehist,binEdges)
    binCounts = length(binEdges)-1;
    matches = zeros(15,3);
    for i = 1:length(rotatehist)
        minval = inf;
        minimg = 1;
        minindx = 1;
        for j = 1:length(templatehist)
            rotatevals = rotatehist{i};
            templatevals = templatehist{j};

            for rotate = 0:(binCounts-1)
                dist = sum((circularRotateRight(rotatevals,rotate)-templatevals).^2);
                if dist < minval
                    minindx = rotate +1;
                    minval = dist;
                    minimg = j;
                end
            end
        end
        minindx = minindx * 180/binCounts;
        matches(i,:) = [i,minimg,minindx];
    end
    truecount = sum(matches(:,1) == matches(:,2));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
