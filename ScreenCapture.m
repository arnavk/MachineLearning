info = imaqhwinfo('winvideo', 1);
vid = videoinput('winvideo', 1, info.DefaultFormat);
preview(vid);
start(vid);
set(vid, 'ReturnedColorSpace', 'rgb');
webcamUpsideDown = 0;
if(webcamUpsideDown == 1)
    inputImage = im2double(imrotate(getsnapshot(vid), 180));
else
    inputImage = im2double(getsnapshot(vid));
end
stop(vid);
faceDetect = vision.CascadeObjectDetector;
boundingBox = step(faceDetect, inputImage);
[row, column] = size(boundingBox);
imageArray = zeroes(10000,row);
hold on
for i = 1:size(boundingBox,1)
    rectangle('Position', boundingBox(i,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
    imageArray(i,:) = reshape(imresize(imcrop(inputImage, boundingBox(i,:)), [100 100]),[],1);
end
title('Face Detection');
hold off;