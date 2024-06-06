clearvars
close all

%% Q1 
src_img = imread("HW1/Images/Q1.png");
src_img = imbinarize(src_img(:,:,1));
struct_et = [0 1 0; 1 1 1; 0 1 0];
struct_et = ones(3);
dilated_img = dilation(src_img,struct_et);
eroded_img = erosion(src_img,struct_et);
closed_img = erosion(dilated_img,struct_et);


figure(1);
subplot(1,2,1);
imshow(src_img);
title("Original Binary Image")
subplot(1,2,2);
imshow(dilated_img);
title("Dilated Binary Image")

figure(2);
subplot(1,2,1);
imshow(src_img);
title("Original Binary Image")
subplot(1,2,2);
imshow(eroded_img);
title("Eroded Binary Image")

figure(3);
subplot(1,2,1);
imshow(src_img);
title("Original Binary Image")
subplot(1,2,2);
imshow(closed_img);
title("Closed Binary Image")


%% Q2
clearvars
src_img_a = imread("HW1/Images/Q2_a.jpg");
src_img_b = imread("HW1/Images/Q2_b.png");

figure;
imshow(src_img_a);

src_imga = src_img_a(:,:,1);
vals_a = custom_histogram(src_imga);

figure;
imshow(src_img_b);

src_imgb = src_img_b(:,:,1);
vals_b = custom_histogram(src_imgb);

%% Q3

[out_img_a,vals_new_a] = customHistEq(src_img_a(:,:,1),vals_a);
[out_img_b,vals_new_b] = customHistEq(src_img_b(:,:,1),vals_b);

figure;
subplot(2,2,1)
imshow(src_img_a)
title("Original Image")
subplot(2,2,2)
imshow(out_img_a)
title("Histogram-Equalized Image")
subplot(2,2,3)
bar(vals_a)
title("Original Histogram")
subplot(2,2,4)
bar(vals_new_a)
title("Equalized Histogram")


figure;
subplot(2,2,1)
imshow(src_img_b)
title("Original Image")
subplot(2,2,2)
imshow(out_img_b)
title("Histogram-Equalized Image")
subplot(2,2,3)
bar(vals_b)
title("Original Histogram")
subplot(2,2,4)
bar(vals_new_b)
title("Equalized Histogram")

%% Q4
clearvars
src_img_a = imread("HW1/Images/Q4_a.png");
src_img_b = imread("HW1/Images/Q4_b.png");

out_img_a = otsu_threshold(src_img_a);
out_img_b = otsu_threshold(src_img_b);

figure();
subplot(1,2,1);
imshow(src_img_a);
subplot(1,2,2);
imshow(out_img_a);

figure();
subplot(1,2,1);
imshow(src_img_b);
subplot(1,2,2);
imshow(out_img_b);

%% Q5
raw_img = imread("HW/CS448 HW1/HW1/Images/Q5.png");
thresh_img = otsu_threshold(raw_img);
erose_img1 = erosion(thresh_img,ones(3));
erose_img2 = erosion(erose_img1,ones(3));
erose_img3 = erosion(erose_img2,ones(3));
erose_img4 = erosion(erose_img3,ones(3));
[labeledImage, numComponents] = CCA(erose_img4);

close all;

figure();
subplot(2,2,1);
imshow(erose_img1);
title("After 1st Erosion")
subplot(2,2,2);
imshow(erose_img2);
title("After 2nd Erosion")
subplot(2,2,3);
imshow(erose_img3);
title("After 3rd Erosion")
subplot(2,2,4);
imshow(erose_img4);
title("After 4th Erosion")

figure();
subplot(2,2,1);
imshow(raw_img);
title("Raw Greyscale Image")

subplot(2,2,2)
imshow(thresh_img);
title("Otsu Thresholded Binary Image")


subplot(2,2,3)
imshow(erose_img1);
title("Eroded Image")


subplot(2,2,4)
imagesc(labeledImage);
colorbar;
title("Connected Components of the Image")
disp("Number of items in the image is: " + num2str(numComponents));

%% Q6 
x = imread("HW/CS448 HW1/HW1/Images/Q6.png");
sobel_x = [-1 0 1;-2 0 2;-1 0 1];
sobel_y = [1 2 1; 0 0 0; -1 -2 -1];

prewitt_x = [-1 0 1; -1 0 1; -1 0 1];
prewitt_y = [1 1 1; 0 0 0; -1 -1 -1];

y_sx = conv2dim(x,sobel_x);
y_sy = conv2dim(x,sobel_y);
y_px = conv2dim(x,prewitt_x);
y_py = conv2dim(x,prewitt_y);

y_smag = sqrt(y_sx.^2 + y_sy.^2); 
y_pmag = sqrt(y_px.^2 + y_py.^2); 

holds = reshape(y_smag.',1,[]);
holdp = reshape(y_pmag.',1,[]);
thresh_s = mean(holds) + 0.5*std(holds);
thresh_p = mean(holdp) + 0.5*std(holdp);

y_smag = y_smag > thresh_s; 
y_pmag = y_pmag > thresh_p; 

figure; 
imshow(y_smag);
title("Edges Detected Using Sobel Operator")

figure;
imshow(y_pmag);
title("Edges Detected Using Prewitt Operator")

%% FUNCTIONS         

% Q1) Dilation Function
function dilated_img = dilation(src_img,struct_et)
src_img = src_img(:,:,1) > 0;
struct_et = struct_et == 1;
dum = size(struct_et);
ones_dum = [0 0];
for y = 1:dum(1)
    for x = 1:dum(2)
        if struct_et(y,x) > 0
            ones_dum = vertcat(ones_dum,[y x]);
        end
    end
end
ones_dum = ones_dum(2:end,:);

size_img = size(src_img);
size_struct = size(struct_et)-1;
len_x = size_img(2) + size_struct(2);
len_y = size_img(1) + size_struct(1);

padded_img = uint8(zeros([len_y , len_x])) == 0;
padded_img(size_struct(1)/2+1:end-size_struct(1)/2,size_struct(2)/2+1:end-size_struct(2)/2) = src_img;

for x = size_struct(2)/2+1:len_x-size_struct(2)/2
    for y = size_struct(1)/2+1:len_y-size_struct(1)/2
        boolval = 0;
        for a = ones_dum'
            ny = a(1); nx = a(2);
            boolval = boolval | (struct_et(ny,nx) & padded_img(y + (ny-(size_struct(1)/2+1)) , x + nx-(size_struct(2)/2 +1)));
        end
        dilated_img(y-size_struct(1)/2,x-size_struct(2)/2) = boolval;
    end
end
dilated_img = uint8(dilated_img*255);
end


% Q1) Erosion Function 

function eroded_img = erosion(src_img,struct_et)
src_img = src_img(:,:,1) > 0;
struct_et = struct_et == 1;
dum = size(struct_et);
ones_dum = [0 0];
for y = 1:dum(1)
    for x = 1:dum(2)
        if struct_et(y,x) == 1
            ones_dum = vertcat(ones_dum,[y x]);
        end
    end
end
ones_dum = ones_dum(2:end,:);

size_img = size(src_img);
size_struct = size(struct_et)-1;
len_x = size_img(2) + size_struct(2);
len_y = size_img(1) + size_struct(1);

padded_img = uint8(zeros([len_y , len_x])) == 0;
padded_img(size_struct(1)/2+1:end-size_struct(1)/2,size_struct(2)/2+1:end-size_struct(2)/2) = src_img;

for y = size_struct(1)/2+1:len_y-size_struct(1)/2
    for x = size_struct(2)/2+1:len_x-size_struct(2)/2
        boolval = 1;
        for a = ones_dum'
            ny = a(1); nx = a(2);
            boolval = boolval & (struct_et(ny,nx) & padded_img(y + (ny-(size_struct(1)/2+1)) , x + nx-(size_struct(2)/2 +1)));
        end
        eroded_img(y-size_struct(1)/2,x-size_struct(2)/2) = boolval;
    end
eroded_img = eroded_img * 255;
end

end


% Q2) Histogram Function

function vals = custom_histogram(src_img)
src_img = reshape(src_img.',1,[]);
vals = zeros([1 256]);
for i = 0:255
    vals(i+1) = sum(src_img == i);
end

figure;
bar(vals);
title("Greyscale Image Histogram")

end

% Q3) Equalizer Function
function [img_eq,vals_new] = customHistEq(img, vals)
    cdf = cumsum(vals) / sum(vals);
    cdf_norm = round(cdf * 255);
    img_eq = zeros(size(img), 'uint8');
    for i = 1:256
        img_eq(img == i-1) = cdf_norm(i);
    end
    vals_new = custom_histogram(img_eq);
end

% Q4) OTSU Thresholding

function binary_image = otsu_threshold(source_image)
    counts = custom_histogram(source_image);
    total = sum(counts);
    sum0 = 0;
    w0 = 0;
    maximum = 0.0;
    sum1 = dot((0:255), counts);
   
    for px = 1:256
        w0 = w0 + counts(px);
        if w0 == 0
            continue;
        end
        w1 = total - w0;
        if w1 == 0
            break;
        end

        sum0 = sum0 +  (px-1) * counts(px);
        m0 = sum0 / w0;
        m1 = (sum1 - sum0) / w1;
        intervar = w0 * w1 * (m0 - m1)^2;
        if ( intervar > maximum )
            maximum = intervar;
            level = px;
        end
    end
    
    binary_image = source_image > level;
    disp(level);

end

% Q5) 


function [labeledImage, numComponents] = CCA(rawImage)
    rawImage = rawImage > 0;
    
    labeledImage = zeros(size(rawImage));
    numComponents = 0;
    [rows, cols] = size(rawImage);
    
    directions = [-1, 0; 1, 0; 0, -1; 0, 1];
    
    function search(startRow, startCol, label)
        stack = [startRow, startCol];
        while ~isempty(stack)
            pixel = stack(1, :);
            stack(1, :) = [];
            
            for i = 1:4
                row_ = pixel(1) + directions(i, 1);
                col_ = pixel(2) + directions(i, 2);
                
                if (row_ > 0) ...
                && (row_ <= rows)...
                && (col_ > 0) ... 
                && (col_ <= cols) ...
                && (rawImage(row_, col_) == 1) ...
                && labeledImage(row_, col_) == 0
                    
                    labeledImage(row_, col_) = label;
                    stack(end+1, :) = [row_, col_]; 
                end
            end
        end
    end
    for r = 1:rows
        for c = 1:cols
            if rawImage(r, c) == 1 && labeledImage(r, c) == 0
                numComponents = numComponents + 1;
                labeledImage(r, c) = numComponents;
                search(r, c, numComponents);
            end
        end
    end
end


function y = conv2dim(x,h)
    y = zeros(size(x) + (size(h)+1)/2);
    sizeh = size(h);
    sizex = size(x);

    imax = sizeh(1);
    jmax = sizeh(2);

    mmax = sizex(1);
    nmax = sizex(2);
    
    beg1 = sizeh(1) - 1;
    beg2 = sizeh(2) - 1;
    end1 = sizeh(1)-1 + sizex(1);
    end2 = sizeh(2)-1 + sizex(2);

    paddedx = zeros(size(x) + 2 * size(h) - 2);
    paddedx(beg1+1:end1,beg2+1:end2) = x;
    
    for m = 1:mmax+(sizeh(1)+1)/2
        for n = 1:nmax+(sizeh(2)+1)/2
            subMatrix = paddedx(m + beg1:-1:m-imax + beg1 + 1,n + beg2:-1:n-jmax + beg2 + 1);
            y(m,n) = y(m,n) + sum(sum(subMatrix .* h));
        end
    end

end


