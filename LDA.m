clear all;

numberOfFolders = 40;
tillWhichImage = 9;

imageDims = [112, 92]; % mxn;
image_dims = imageDims;
sizeImg = prod(imageDims);
images = zeros(prod(imageDims), tillWhichImage*numberOfFolders);
normalizedImages = zeros(prod(imageDims), tillWhichImage*numberOfFolders);
imageIndex = 1;
numImages = tillWhichImage*numberOfFolders;
num_images = numImages;

for index = 1:numberOfFolders
    input_dir = strcat('atandtcambridge/s',num2str(index),'/');
    disp(input_dir);
    for n = 1:tillWhichImage
        file = strcat(num2str(n),'.pgm');
        fullPath = strcat(input_dir,file);
        disp(fullPath);
        img = imread(fullPath);
        img = histeq(img);
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

%Calculate the average of each class (person)
imageIndex = 1;
meanClass = zeros(sizeImg, numberOfFolders);
for index = 1:numberOfFolders
    meanClass(:,index) = meanClass(:,index) + mean(images(:,((9*(index-1)+1):((9*(index-1)+1)+(tillWhichImage - 1)))),2);
end

XB = zeros(sizeImg, numberOfFolders);
for index = 1:numberOfFolders
    XB(:,index) = meanClass(:,index) - mean_face(:);
end

imageIndex = 1;
XW = zeros(sizeImg, numImages);
for index = 1:numberOfFolders
    for n = 1:tillWhichImage
        XW(:,imageIndex) = images(:,imageIndex) - meanClass(:,index);
        imageIndex = imageIndex + 1;
    end
end

% Yu and Yang's method to avoid SB being singular 
 
[vecB,valB] = eig(XB'*XB); 
D = diag(valB); 
[temp, sortind] = sort(-1.*D); 
vecB = vecB(:,sortind); 
D = D(sortind); 
V = XB*vecB; 
 
maxnoFeature = 30; 
Y = V(:,1:maxnoFeature); 
DB = Y'*XB*XB'*Y; 
Z = Y*DB^(-0.5); 
H = Z'*XW;%M*630

[vecU,valU]=eig(H*H'); 
diagvalU=diag(valU); 

maxnoEig = 4;

accuracyLDA = zeros(numImages,maxnoEig-1);

%for L = 1:maxnoEig - 1
L = 20;
    vecUPart = vecU(:,1:L);
    W = Z*vecUPart; 
    for i=1:L 
        W(:,i) = W(:,i)/norm(W(:,i)); 
    end 
    
    ProjectedImages = zeros(L,numImages); 
    ProjectedTestImageOriginal = imread('atandtcambridge/s20/10.pgm');
    ProjectedTestImageOriginal = histeq(ProjectedTestImageOriginal);
    ProjectedTestImageOriginal = im2double(ProjectedTestImageOriginal);
    ProjectedTestImageOriginal = reshape(ProjectedTestImageOriginal(:),[],1);
    ProjectedTestImage = W'*(ProjectedTestImageOriginal-mean_face);
    
    for i = 1:numImages 
        ProjectedImages(:,i) = W'*(images(:,i)-mean_face); 
    end
    
    errDistance = zeros(numImages,1); 
    
    for trainInd = 1 : numImages 
        errDistance(trainInd) = (norm(ProjectedTestImage(:)- ProjectedImages(:,trainInd)))^2; 
    end
    [minDist, recogInd] = min(errDistance);

    figure(), imshow(reshape(ProjectedTestImageOriginal,image_dims),[]);
    figure(),  imshow(reshape(images(:,recogInd), image_dims),[]);
%end


