%clear memory completely
clear all;

%number of people and the number of photos they each have
numberOfPeople = 40;
imagePerPerson = 8;
numberOfImages = numberOfPeople*imagePerPerson;

%size of image post face detection and cropping
imageDimension = [100, 100]; 
pixelValue = prod(imageDimension);

%array to hold training database
trainingImages = zeros(prod(imageDimension), numberOfImages);

%Face detection using Viola-Jones Algorithm
FaceDetect = vision.CascadeObjectDetector;

%test database variables of images
testColour = 0; %set to 1 if images are in colour
testFaceRecognitionEnable = 0; %set to 1 if face recognition is required
testDirectory = 'atandtcambridge'; %select database to load from
testFileType = '.pgm';

%load database of images
colour = 0; %set to 1 if images are in colour
faceRecognitionEnable = 0; %set to 1 if face recognition is required
directory = 'atandtcambridge'; %select database to load from
fileType = '.pgm';
imageIndex = 1; %temporary variable required in reading loop

for index = 1:numberOfPeople
    inputPathFull = strcat('/',directory,'/s',num2str(index),'/');
    for number = 1:imagePerPerson
        file = strcat(num2str(number),fileType);
        inputFilePathFull = strcat(inputPathFull,file);
        disp(inputFilePathFull);
        img = imread(inputFilePathFull); %read image
        img = im2double(img); %change image class to double
        if(faceRecognitionEnable == 1)
            rectangle = step(FaceDetect, img); %detect face in the image
            img = imcrop(img, rectangle); %crop image to only contain face
        end
        if(colour ==  1)
            img = rgb2gray(img); %change to gray scale if required
        end
        img = imresize(img, [100 100]); %resize to consistent size
        %img = histeq(img); %increase contrast of images
        trainingImages(:, imageIndex) = reshape(img(:),[],1); %store image as 1-d array
        imageIndex = imageIndex + 1;
    end
end
disp('Images loaded');

%
%
%
% Images LOADED!

%Find mean face of training database
meanFaceTraining = mean(trainingImages, 2); 
numberEigenFaces = 30;

%
% PCA Functions
% Principle Component Analysis
%

%Find mean-shifted input images
repmat = repmat(meanFaceTraining, 1, numberOfImages); % returns an n-by-n tiling of A. The size of B is size(A) * n
shiftedImages = trainingImages - repmat;

%Calculate the ordered eigenvectors and eigenvalues
[evectors, score, evalues] = princomp(trainingImages');

disp('PCA Training Complete');
%
% PCA Training Complete
%
%

%
% LDA Functions (Magic)
% Linear Discriminant Analysis
%

%Calculate the average image of each person
meanFacePersonTraining = zeros(pixelValue, numberOfPeople);
for index = 1:numberOfPeople
    meanFacePersonTraining(:,index) =  meanFacePersonTraining(:,index) + mean(trainingImages(:,((imagePerPerson*(index-1)+1):((imagePerPerson*(index-1)+1)+(imagePerPerson - 1)))),2);
end

%Subtract mean face from mean face of each person in training database
ScatterBetweenClass = zeros(pixelValue, numberOfPeople);
for index = 1:numberOfPeople
    ScatterBetweenClass(:,index) = meanFacePersonTraining(:,index) - meanFaceTraining(:);
end

%Subtract mean face of each person from their individual photos
ScatterWithinClass = zeros(pixelValue, numberOfImages);
imageIndex = 1;
for index = 1:numberOfPeople
    for number = 1:imagePerPerson
        ScatterWithinClass(:,imageIndex) = trainingImages(:,imageIndex) - meanFacePersonTraining(:,index);
        imageIndex = imageIndex + 1;
    end
end

%This is the magical maths behind LDA. Arnav please analyze
[vectors, eigenValuesMatrix] = eig(ScatterBetweenClass'*ScatterBetweenClass);
diagonal = diag(eigenValuesMatrix);
[temporary, sortedEigen] = sort(-1.*diagonal);
vectors = vectors(:,sortedEigen);
diagonal = diagonal(sortedEigen);
vectors = ScatterBetweenClass*vectors;

maxNoFeature = 30; %Features we want to use in LDA
selectedFeatures = vectors(:,1:maxNoFeature);
DB = selectedFeatures'*ScatterBetweenClass*ScatterBetweenClass'*selectedFeatures;
Z = selectedFeatures*DB^(-0.5);
H = Z'*ScatterWithinClass;

[vector, value] = eig(H*H');
diagonal = diag(value);


disp('LDA training complete');
%
% LDA Training Complete
%
%

%
% Computing Accuracy Values for different eigenface 
%
%
disp('Studying accuracy now');
%Variables to store results
eigenSamePersonPCA = zeros(numberEigenFaces,1);
eigenSamePersonLDA = zeros(numberEigenFaces,1);

for eigenFaces = 1:numberEigenFaces
    imageIndex = 1;
    %Generate PCA features using Eigen number
    %Retaining top eigen faces
    clearvars evectorsNew; %remove from memory
    evectorsNew = evectors(:, 1:eigenFaces);

    %Project all images into the subspace to generate the feature vectors
    features = evectorsNew' * shiftedImages;
    
    %Generate LDA features using Eigen number
    vectorPart = vector(:,1:eigenFaces);
    weight = Z*vectorPart;
    for tempIndex = 1:eigenFaces
        weight(:,tempIndex) = weight(:,tempIndex)/norm(weight(:,tempIndex));
    end
    
    projectedImages = zeros(eigenFaces,numberOfImages);
    
    for tempIndex = 1:numberOfImages
        projectedImages(:,tempIndex) = weight'*(trainingImages(:,tempIndex) - meanFaceTraining);
    end
    
    for index = 1:numberOfPeople
        inputPathFull = strcat('/',testDirectory,'/s',num2str(index),'/');
        for number = 9:10
            file = strcat(num2str(number),testFileType);
            inputFilePathFull = strcat(inputPathFull,file);
            disp(eigenFaces);
            disp(inputFilePathFull);
            img = imread(inputFilePathFull); %read image
            img = im2double(img); %change image class to double
            if(testFaceRecognitionEnable == 1)
                rectangle = step(FaceDetect, img); %detect face in the image
                img = imcrop(img, rectangle); %crop image to only contain face
            end
            if(testColour ==  1)
                img = rgb2gray(img); %change to gray scale if required
            end
            img = imresize(img, [100 100]); %resize to consistent size
            %img = histeq(img);
            testImage = img;
            
            %PCA detection
            % calculate the similarity of the input to each training image
            featureVec = evectorsNew' * (testImage(:) - meanFaceTraining(:));
            similarityScore = arrayfun(@(n) 1 / (1 + norm(features(:,n) - featureVec)), 1:numberOfImages);
            
            % find the image with the highest similarity
            [match_score, pcaMatchID] = max(similarityScore);
            
            %LDA detection
            testImageLDA = reshape(img(:),[],1);
            testImageLDA = weight'*(testImageLDA - meanFaceTraining);
            
            errDistance = zeros(numberOfImages,1);
            
            for trainIndex = 1:numberOfImages
                errDistance(trainIndex) = (norm(testImageLDA(:) - projectedImages(:,trainIndex)))^2; 
            end
            [minDist, ldaMatchID] = min(errDistance);
            
            %Store results
            if(index == ceil(pcaMatchID/imagePerPerson))
                eigenSamePersonPCA(eigenFaces) = eigenSamePersonPCA(eigenFaces) + 1;
            end
            
            if(index == ceil(ldaMatchID/imagePerPerson))
                eigenSamePersonLDA(eigenFaces) = eigenSamePersonLDA(eigenFaces) + 1;
            end
                
            imageIndex = imageIndex + 1;
        end
    end
end

figure(); 
plot(eigenSamePersonLDA, 'b'), xlabel('Eigenvalue'), ylabel('Correctly Recognized Face'); 
hold on;  
plot(eigenSamePersonPCA, 'r'); 
legend('LDA','PCA');
legend('Location','SouthEast');

