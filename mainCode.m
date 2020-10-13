function[]=CLAHE();
clc
close all 
clear all

[filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
he = imread([pathname,filename]);
%disp(filename);
t=filename(1:1);
%disp(t);
he = imresize(he,[256,256]);
he = imadjust(he,stretchlim(he));
figure, imshow(he);title('Contrast Enhanced');
%text(size(he,2),size(he,1)+15,...
%     'Image courtesy of Alan Partin, Johns Hopkins University', ...
 %    'FontSize',7,'HorizontalAlignment','right');
 cform = makecform('srgb2lab');
lab_he = applycform(he,cform);
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = 3;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
pixel_labels = reshape(cluster_idx,nrows,ncols);
%imshow(pixel_labels,[]), title('image labeled by cluster index');
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = he;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

imshow(segmented_images{1});title('Cluster 1');imshow(segmented_images{2});title('Cluster 2');
imshow(segmented_images{3});title('Cluster 3');
%set(gcf, 'Position', get(0,'Screensize'));
mean_cluster_value = mean(cluster_center,2);
[tmp, idx] = sort(mean_cluster_value);
blue_cluster_num = idx(1);

L = lab_he(:,:,1);
blue_idx = find(pixel_labels == blue_cluster_num);
L_blue = L(blue_idx);
is_light_blue = imbinarize(L_blue);
nuclei_labels = repmat(uint8(0),[nrows ncols]);
nuclei_labels(blue_idx(is_light_blue==false)) = 1;
nuclei_labels = repmat(nuclei_labels,[1 1 3]);
blue_nuclei = he;
blue_nuclei(nuclei_labels ~= 1) = 0;
imshow(blue_nuclei), title('blue nuclei');
% Convert to grayscale if image is RGB
if ndims(blue_nuclei) == 3
   img = rgb2gray(blue_nuclei);
end

d = imdistline;
[centers, radii] = imfindcircles(img,[1 17],'ObjectPolarity','dark')
[centers, radii] = imfindcircles(img,[1 17],'ObjectPolarity','dark', ...
    'Sensitivity',0.9)
%imshow(img);

%h = viscircles(centers,radii);
%[centers, radii] = imfindcircles(img,[1 26], 'ObjectPolarity','dark', ...
 %         'Sensitivity',0.92,'Method','twostage');

%delete(h);

%h = viscircles(centers,radii);
%[centers, radii] = imfindcircles(img,[1 26], 'ObjectPolarity','dark', ...
 %         'Sensitivity',0.95);

%delete(h);

viscircles(centers,radii);
radius=max(radii);
%radius=radius*1000;
%disp(radius);
aream=radii*3.14;
aream=aream.*radii;
area1=max(aream);
%area1=area1*1000;
%disp(area1);
perim=radii*2*3.14;
peri=max(perim);
%peri=peri*1000;
%disp(peri);
Variance = mean2(var(double(blue_nuclei)));
smoothness=Variance/10000;
%smoothness=smoothness*1000;
%disp(smoothness);
texture = std2(blue_nuclei);
%texture=texture*1000;
%disp(texture);
feat_disease = [radius,texture,peri,area1,smoothness];
%%
% Load All The Features
load('fdata.mat')
%disp(radius);
% Put the test features into variable 'test'
%disp(feat_disease);
test = feat_disease;
%disp(test);
disp(radius);
%test=test*1000;
%disp(test);
%disp(t);
result = multisvm(data,grp,test,t);
%disp(result);
if result == 0
    helpdlg(' Benign ');
    disp(' Benign ');
elseif result == 1
    helpdlg(' Malignant ');
    disp('Malignant');
end