clear all;

numberOfFolders = 40;
tillWhichImage = 9;

imageDims = [112, 92]; % mxn;
image_dims = imageDims;
images = zeros(prod(imageDims), tillWhichImage*numberOfFolders);
normalizedImages = zeros(prod(imageDims), tillWhichImage*numberOfFolders);
imageIndex = 1;
numImages = tillWhichImage*numberOfFolders;
num_images = numImages;

for i = 0:(numberOfFolders - 1)
    index = numberOfFolders - i;
    input_dir = strcat('/atandtcambridge/s',num2str(index),'/');
    disp(input_dir);
    for n = 1:tillWhichImage
        file = strcat(num2str(n),'.pgm');
        fullPath = strcat(input_dir,file);
        disp(fullPath);
        img = imread(fullPath);
        img = im2double(img);
        %imshow(img);
        images(:, imageIndex) = reshape(img(:),[],1);
        imageIndex = imageIndex + 1;
    end
end

% imagePrint = reshape(images(:,1),imageDims);
% figure, imshow(imagePrint,[]);

% steps 1 and 2: find the mean image and the mean-shifted input images
mean_face = mean(images, 2);
repmat = repmat(mean_face, 1, num_images);
shifted_images = images - repmat;
 
% steps 3 and 4: calculate the ordered eigenvectors and eigenvalues
[evectors, score, evalues] = princomp(images');
 
% step 5: only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
num_eigenfaces = 40;
evectors = evectors(:, 1:num_eigenfaces);
 
% step 6: project the images into the subspace to generate the feature vectors
features = evectors' * shifted_images;

%get input image and convert to type double
input_image = imread('/atandtcambridge/s2/10.pgm');
input_image = im2double(input_image);

% calculate the similarity of the input to each training image
feature_vec = evectors' * (input_image(:) - mean_face);
similarity_score = arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)), 1:num_images);
 
% find the image with the highest similarity
[match_score, match_ix] = max(similarity_score);
 
% display the result
figure(), imshow(input_image,[]);
figure(),  imshow(reshape(images(:,match_ix), image_dims),[]);

% title(sprintf('matches %s, score %f', filenames(match_ix).name, match_score));



