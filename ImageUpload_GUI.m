classdef ImageUpload_GUI < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                    matlab.ui.Figure
        SelectVehicleTypePanel      matlab.ui.container.Panel
        TaxiButton                  matlab.ui.control.Button
        TruckButton                 matlab.ui.control.Button
        MinibusVanButton            matlab.ui.control.Button
        BusButton                   matlab.ui.control.Button
        Motorcycle500ccButton       matlab.ui.control.Button
        Motorcycle500ccButton_2     matlab.ui.control.Button
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Common function to handle button press and file upload
        function imagePath = handleVehicleSelection(app, vehicleType)
            imagePath = "";
            [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp)'}, ...
                ['Select an image for ', vehicleType]);
            if isequal(file, 0)
                disp('User selected Cancel');
            else
                imagePath = fullfile(path, file);
                disp(['User selected ', fullfile(path, file)]);
                fprintf('Selected vehicle type: %s\n', vehicleType);
                % Here you can add the code to handle the selected file
            end
        end

        % Button pushed function: TaxiButton
        function TaxiButtonPushed(app, event)
            path = handleVehicleSelection(app, 'Taxi');
            img = imread(path); % read image
            %% RGB to YCbCr
            
            % % Convert the image from RGB to YCbCr
            ycbcrImage = rgb2ycbcr(img);
            
            % Extract the Y channel
            yChannel = ycbcrImage(:, :, 1);
            %% Edge Detection
            img_edged = edge(yChannel, 'Canny');
            %% ROI Extraction
            contours = bwboundaries(img_edged);
            
            figure; 
            subplot(221);imshow(img);
            
            max_bounding_box = [];
            max_area = -1;
            
            % Loop through each contour
            for k = 1:length(contours)
                % Iterate over contours to find the largest rectangular contour
                boundary = contours{k};
                area = polyarea(boundary(:, 2), boundary(:, 1)); % Calculates the area of the boundary
                if area > max_area % Finds the largest boundary
                    max_area = area;
                    bounding_box = regionprops('table', img_edged, 'BoundingBox');
                    max_bounding_box = bounding_box.BoundingBox(k,:);
                end
            end
            
            rectangle('Position', max_bounding_box, 'EdgeColor', 'g', 'LineWidth', 2);
            roi = imcrop(img, max_bounding_box); % Extract Region Of Interest based on max_area 
            subplot(222);imshow(roi);
            
            %% Thresholding to separate image from background
            threshold = graythresh(roi); % Using Otsu's method of binarization
            x = rgb2ycbcr(roi);
            yChannelROI = x(:, :, 1);
            binaryImage = imbinarize(yChannelROI, threshold);
            %% Template Matching Character Recognition
            template_firstLetter = readFirstLetter(binaryImage);
            subplot(223), imshow(binaryImage);
            title(sprintf('Template Detection:\nFirst Character Detected: %s\nState: %s', template_firstLetter, getState(template_firstLetter)));
            %% OCR Library Character Recognition
            OCR_results = ocr(binaryImage, 'TextLayout', 'Word', 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890');
            OCR_firstLetter = OCR_results.Text(1);  % Get the first letter of the recognized text
            OCR_plate = OCR_results.Text; % Get the entire plate number
            
            subplot(224);imshow(binaryImage); % Display the results
            
            title(sprintf('OCR Detection:\nFirst Character Detected: %s\nState: %s', OCR_firstLetter, getState(OCR_firstLetter)));
            
            hold off;
            
            function letter = readFirstLetter(snap)
                % Define the template images for specific characters
                templates = {
                    'A', imread('Template/A.bmp');
                    'M', imread('Template/M.bmp');
                    'B', imread('Template/B.bmp');
                    'N', imread('Template/N.bmp');
                    'C', imread('Template/C.bmp');
                    'P', imread('Template/P.bmp');
                    'R', imread('Template/R.bmp');
                    'F', imread('Template/F.bmp');
                    'T', imread('Template/T.bmp');
                    'W', imread('Template/W.bmp');
                    'V', imread('Template/V.bmp');
                    'J', imread('Template/J.bmp');
                    'K', imread('Template/K.bmp');
                };
                snap = imresize(snap, [42 24]);  % Resize snap to match the template size
                rec = [];
                % Perform correlation and find the best match
                for n = 1:size(templates, 1)
                    cor = corr2(templates{n, 2}, snap);
                    rec = [rec cor];
                end
                [~, ind] = max(rec);
                letter = templates{ind, 1};  % Get the letter corresponding to the highest correlation
            end
            
            function result = getState(letter)
                if strcmp(letter, 'A')
                    result = 'Perak';
                elseif strcmp(letter, 'M')
                    result = 'Malacca';
                elseif strcmp(letter, 'B')
                    result = 'Selangor';
                elseif strcmp(letter, 'N')
                    result = 'Negeri Sembilan';
                elseif strcmp(letter, 'C')
                    result = 'Pahang';
                elseif strcmp(letter, 'P')
                    result = 'Penang';
                elseif strcmp(letter, 'R')
                    result = 'Perlis';
                elseif strcmp(letter, 'F')
                    result = 'Putrajaya';
                elseif strcmp(letter, 'T')
                    result = 'Terengganu';
                elseif strcmp(letter, 'W') || strcmp(letter, 'V')
                    result = 'Kuala Lumpur';
                elseif strcmp(letter, 'J')
                    result = 'Johor';
                elseif strcmp(letter, 'K')
                    result = 'Kedah';
                elseif strcmp(letter, 'H')
                    result = 'Taxi';
                elseif strcmp(letter, 'Z')
                    result = 'Military';
                else
                    result = 'Unknown prefix';
                end
            end

        end

        % Button pushed function: TruckButton
        function TruckButtonPushed(app, event)
            path = handleVehicleSelection(app, 'Truck');
            % Load the image
            image = imread(path);
            
            % Convert to YCbCr color space
            ycbcrImage = rgb2ycbcr(image);
            
            % Extract the Y channel
            yChannel = ycbcrImage(:,:,1);
            
            % Extract the V channel
            vChannel = ycbcrImage(:,:,3);
            
            % Edge detection
            img_edged = edge(yChannel, 'Canny');
            
            % ROI Extraction
            contours = bwboundaries(img_edged);
            
            hold on;
            
            max_area = -1;
            max_bounding_box = [];
            
            % Loop through each contour
            for k = 1:length(contours)
                boundary = contours{k};
                x = boundary(:, 2);
                y = boundary(:, 1);
                min_x = min(x);
                max_x = max(x);
                min_y = min(y);
                max_y = max(y);
                width = max_x - min_x + 1;
                height = max_y - min_y + 1;
                
                aspect_ratio = width / height;
                if aspect_ratio < 2 || aspect_ratio > 6
                    continue; % Skip contours that do not match the aspect ratio of truck plates
                end
            
                % Extract the ROI
                roi = vChannel(min_y:max_y, min_x:max_x);
            
                % Check for white color (mean pixel intensity in grayscale should be high)
                mean_intensity = mean(roi(:));
                if mean_intensity < 0.3  % Adjust this threshold as per truck license plate color characteristics
                    continue; % Skip non-white regions
                end
            
                % Area calculation and selection
                area = polyarea(boundary(:, 2), boundary(:, 1));
                if area > max_area
                    max_area = area;
                    % Add a margin around the bounding box
                    margin = 4.5; % Adjust this value as needed
                    max_bounding_box = [min_x - margin, min_y - margin, width + 2*margin, height + 2*margin];
                end
            end
            
            % Draw the bounding box on the original image
            if ~isempty(max_bounding_box)
                rectangle('Position', max_bounding_box, 'EdgeColor', 'g', 'LineWidth', 2);
            end
            
            plate = imcrop(yChannel, max_bounding_box);
            hold off;
            
            % Extract the region of interest
            plate = imcrop(yChannel, max_bounding_box);
            
            % Resize the image
            resized_plate = imresize(plate, [200, NaN]); % Resize to a height of 100 pixels while maintaining aspect ratio
            
            % Contrast Enhancement
            enhanced_plate = adapthisteq(resized_plate); % Adaptive Histogram Equalization
            
            % Noise Reduction using Gaussian Filter
            filtered_plate = imgaussfilt(enhanced_plate, 2); % Gaussian filter with standard deviation of 1 pixel
            
            % Binarization
            binary_plate = filtered_plate > 185;
            
            % Dilation and Erosion
            se = strel('square', 1); % Define structuring element
            dilated_plate = imdilate(binary_plate, se);
            eroded_plate = imerode(dilated_plate, se);
            
            % Additional Noise Reduction
            cleaned_plate = bwareaopen(eroded_plate, 250); % Remove small connected components (noise)
            
            % Use OCR to read the characters
            ocrResults = ocr(cleaned_plate, 'TextLayout', 'Word', 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ');
            
            % Get the recognized text
            recognizedText = ocrResults.Text;
            
            % Extract the first character
            if ~isempty(recognizedText) && isletter(recognizedText(1))
                letter = recognizedText(1);
            else
                letter = 'N/A';
            end
            
            % Determine the state name
            state_name = getState(letter);
            
            % Display the images as subplots
            subplot(1, 3, 1);
            imshow(image);
            title('Original Image');
            hold on;
            if ~isempty(max_bounding_box)
                rectangle('Position', max_bounding_box, 'EdgeColor', 'g', 'LineWidth', 2);
            end
            hold off;
            
            subplot(1, 3, 2);
            imshow(plate);
            title('Extracted Plate');
            
            subplot(1, 3, 3);
            imshow(cleaned_plate);
            title(sprintf('OCR Detection\nFirst Character Detected: %s\nState: %s', letter, state_name));
            
            % Function to get the state name from the letter
            function result = getState(letter)
                if strcmp(letter, 'A')
                    result = 'Perak';
                elseif strcmp(letter, 'M')
                    result = 'Malacca';
                elseif strcmp(letter, 'B')
                    result = 'Selangor';
                elseif strcmp(letter, 'N')
                    result = 'Negeri Sembilan';
                elseif strcmp(letter, 'C')
                    result = 'Pahang';
                elseif strcmp(letter, 'P')
                    result = 'Penang';
                elseif strcmp(letter, 'R')
                    result = 'Perlis';
                elseif strcmp(letter, 'F')
                    result = 'Putrajaya';
                elseif strcmp(letter, 'T')
                    result = 'Terengganu';
                elseif strcmp(letter, 'W') || strcmp(letter, 'V')
                    result = 'Kuala Lumpur';
                elseif strcmp(letter, 'J')
                    result = 'Johor';
                elseif strcmp(letter, 'S')
                    result = 'Sabah';
                elseif strcmp(letter, 'Q')
                    result = 'Sarawak';
                elseif strcmp(letter, 'K')
                    result = 'Kedah';
                elseif strcmp(letter, 'H')
                    result = 'Taxi';
                elseif strcmp(letter, 'Z')
                    result = 'Military';
                else
                    result = 'Unknown prefix';
                end
            end
        end

        % Button pushed function: MinibusVanButton
        function MinibusVanButtonPushed(app, event)
            path = handleVehicleSelection(app, 'Minibus/Van');
            % Read the input image
            img = imread(path);
            
            % Convert the image to grayscale
            grayImg = rgb2gray(img);
            
            % Perform global thresholding to obtain the binary threshold value
            imgThreshold = graythresh(grayImg);
            
            % Binarize the grayscale image using the obtained threshold value
            segmentedImg = imbinarize(grayImg, imgThreshold);
            
            % Find connected components in the binarized image
            cc = bwconncomp(segmentedImg);
            
            % Get properties of the connected regions (area and bounding box)
            stats = regionprops("table", cc, "Area", "BoundingBox");
            area = stats.Area;
            bbox = stats.BoundingBox;
            
            % Calculate the center of each bounding box
            center_x = bbox(:, 1) + bbox(:, 3) / 2;
            center_y = bbox(:, 2) + bbox(:, 4) / 2;
            
            % Define the region of interest (bottom center region of the image)
            bottomCenterRegion = (center_y <= size(img, 1) * 0.9) & ...
                                 (center_y >= size(img, 1) * 0.6) & ...
                                 (center_x >= size(img, 2) * 0.2) & ...
                                 (center_x <= size(img, 2) * 0.8);
            
            % Initialize variable to store the selected bounding boxes
            selectedBoundingBoxes = [];
            % Iterate through each bounding box found
            for i = 1:size(bbox, 1)
                % Filter bounding boxes based on criteria
                if (area(i) > 30) && ...
                   (bbox(i, 3) < 35) && ...
                   (bbox(i, 4) >= 15 && bbox(i, 4) <= 85) && ...
                   bottomCenterRegion(i)
               
                    % If no bounding box is selected yet, add the first one
                    if isempty(selectedBoundingBoxes)
                        selectedBoundingBoxes = [selectedBoundingBoxes; bbox(i, :)];
                        continue;
                    end
            
                    % Get threshold coordinates for proximity to previous bounding box
                    threshold_x = selectedBoundingBoxes(end, 1) + selectedBoundingBoxes(end, 3) + 40;
                    threshold_y = selectedBoundingBoxes(end, 2) + selectedBoundingBoxes(end, 4) + 40;
            
                    % Add bounding boxes that are near the previous bounding box
                    if (bbox(i, 1) < threshold_x) && (bbox(i, 2) < threshold_y)
                        selectedBoundingBoxes = [selectedBoundingBoxes; bbox(i, :)];
                    end  
                end
            end
            
            % Check if no bounding boxes match the criteria
            if isempty(selectedBoundingBoxes)
                error('No bounding boxes found that match the criteria.');
            end
            
            % Calculate the coordinates for the final bounding box encompassing all selected bounding boxes
            min_x = min(selectedBoundingBoxes(:, 1));
            max_x = max(selectedBoundingBoxes(:, 1) + selectedBoundingBoxes(:, 3));
            min_y = min(selectedBoundingBoxes(:, 2));
            max_y = max(selectedBoundingBoxes(:, 2) + selectedBoundingBoxes(:, 4));
            
            % Calculate width and height of the bounding box
            w = max_x - min_x;
            h = max_y - min_y;
            
            % Define the region of interest (ROI) for OCR based on the first bounding box
            roi_min_x = min(selectedBoundingBoxes(1, 1));
            roi_max_x = max(roi_min_x + selectedBoundingBoxes(1, 3));
            roi_min_y = min(selectedBoundingBoxes(1, 2));
            roi_max_y = max(roi_min_y + selectedBoundingBoxes(1, 4));
            roi = [roi_min_x - 5,  roi_min_y - 5, roi_max_x - roi_min_x + 10, roi_max_y - roi_min_y + 10];
            
            % Get the state/prefix from the image using the defined region
            [state, prefix] = getState(img, [min_x - 20, min_y - 10, w + 50, h + 15]);
            
            
            % Create a figure for the subplots
            figure('Name', 'License Plate Detection', 'NumberTitle', 'off');
            
            % Subplot 1: Display the original image with highlighted bounding boxes
            subplot(2, 2, 1);
            imshow(img);
            hold on;
            % Red rectangle for the license plate
            rectangle('Position', [min_x - 20, min_y - 10, w + 50, h + 15], 'EdgeColor', 'r', 'LineWidth', 2);
            % Yellow rounded rectangle for the ROI
            rectangle('Position', roi, 'EdgeColor', 'y', 'LineWidth', 2, 'Curvature', [1 1]); 
            hold off;
            title('Original Image with Bounding Boxes');
            
            % Subplot 2: Display the cropped license plate image
            croppedLicense = imcrop(img, [min_x - 20, min_y - 10, w + 50, h + 15]);
            subplot(2, 2, 2);
            imshow(croppedLicense);
            title('Cropped License Plate');
            
            % Subplot 3: Display the recognized text information
            subplot(2, 2, 3);
            axis off;
            text(0.1, 0.6, ['Prefix: ' prefix], 'FontSize', 12);
            text(0.1, 0.4, ['State: ' state], 'FontSize', 12);
            title('Recognized Text Information');
            % Perform OCR on the cropped image and find the first alphabet for state identification
            testing = ocr(croppedLicense);
            recognizedText = testing.Text;
            disp(['Recognized Text: ', recognizedText]);
            
            % Iterate through the recognized text to find the first alphabet
            stateFound = false;
            for i = 1:length(recognizedText)
                if isletter(recognizedText(i))
                    [state, prefix] = getStateFromPrefix(recognizedText(i));
                    stateFound = true;
                    break;
                end
            end
            
            % If no valid state prefix is found, set state to 'State not found'
            if ~stateFound
                state = 'State not found';
                prefix = '';
            end
            
            % Subplot 4: Display the state information based on the recognized text
            subplot(2, 2, 4);
            axis off;
            if stateFound
                text(0.1, 0.7, ['Recognized Text: ' recognizedText], 'FontSize', 12);
                text(0.1, 0.5, ['Prefix: ' prefix], 'FontSize', 12); % Display the prefix
                text(0.1, 0.3, ['State: ' state], 'FontSize', 12);
            else
                text(0.1, 0.5, 'State not found', 'FontSize', 12);
            end
            title('Raw OCR Result');
            
            % Display the recognized text and state in the console
            disp(['Recognized Text: ', recognizedText]);
            disp(['Prefix: ', prefix]);
            disp(['State: ', state]);
            
            % Function to get the state/prefix from the license plate region
            function [result, prefix] = getState(image, roi)
                ocrResults = ocr(image, roi);
               disp(ocrResults);
                if ocrResults.Text == ""
                    result = 'Unknown prefix';
                    prefix = 'Unrecognizable';
                    return;
                end
            
                prefix = ocrResults.Text(1);
            
                % Determine the state based on the prefix
                [result, prefix] = getStateFromPrefix(prefix);
            end
            
            % Function to get state from a prefix character
            function [result, prefix] = getStateFromPrefix(prefix)
                switch prefix
                    case 'A'
                        result = 'Perak';
                    case 'M'
                        result = 'Malacca';
                    case 'B'
                        result = 'Selangor';
                    case 'N'
                        result = 'Negeri Sembilan';
                    case 'C'
                        result = 'Pahang';
                    case 'P'
                        result = 'Penang';
                    case 'R'
                        result = 'Perlis';
                    case 'F'
                        result = 'Putrajaya';
                    case 'T'
                        result = 'Terengganu';
                    case {'W', 'V', 'w'}
                        result = 'Kuala Lumpur';
                    case 'J'
                        result = 'Johor';
                    case 'K'
                        result = 'Kedah';
                    case 'H'
                        result = 'Taxi';
                    case 'Z'
                        result = 'Military';
                    case 'S'
                        result = 'Private and Commercial Vehicle';
                    otherwise
                        result = 'Unknown prefix';
                end
            end
        end

        % Button pushed function: BusButton
        function BusButtonPushed(app, event)
            path = handleVehicleSelection(app, 'Bus');
            img = imread(path); % read image
            %% RGB to YCbCr
            
            % % Convert the image from RGB to YCbCr
            ycbcrImage = rgb2ycbcr(img);
            
            % Extract the Y channel
            yChannel = ycbcrImage(:, :, 1);
            %% Edge Detection
            img_edged = edge(yChannel, 'Canny');
            %% ROI Extraction
            contours = bwboundaries(img_edged);
            
            figure; 
            subplot(221);imshow(img);
            
            max_bounding_box = [];
            max_area = -1;
            
            % Loop through each contour
            for k = 1:length(contours)
                % Iterate over contours to find the largest rectangular contour
                boundary = contours{k};
                area = polyarea(boundary(:, 2), boundary(:, 1)); % Calculates the area of the boundary
                if area > max_area % Finds the largest boundary
                    max_area = area;
                    bounding_box = regionprops('table', img_edged, 'BoundingBox');
                    max_bounding_box = bounding_box.BoundingBox(k,:);
                end
            end
            
            rectangle('Position', max_bounding_box, 'EdgeColor', 'g', 'LineWidth', 2);
            roi = imcrop(img, max_bounding_box); % Extract Region Of Interest based on max_area 
            subplot(222);imshow(roi);
            
            %% Thresholding to separate image from background
            threshold = graythresh(roi); % Using Otsu's method of binarization
            x = rgb2ycbcr(roi);
            yChannelROI = x(:, :, 1);
            binaryImage = imbinarize(yChannelROI, threshold);
            %% Template Matching Character Recognition
            template_firstLetter = readFirstLetter(binaryImage);
            subplot(223), imshow(binaryImage);
            title(sprintf('Template Detection:\nFirst Character Detected: %s\nState: %s', template_firstLetter, getState(template_firstLetter)));
            %% OCR Library Character Recognition
            OCR_results = ocr(binaryImage, 'TextLayout', 'Word', 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890');
            OCR_firstLetter = OCR_results.Text(1);  % Get the first letter of the recognized text
            OCR_plate = OCR_results.Text; % Get the entire plate number
            
            subplot(224);imshow(binaryImage); % Display the results
            
            title(sprintf('OCR Detection:\nFirst Character Detected: %s\nState: %s', OCR_firstLetter, getState(OCR_firstLetter)));
            
            hold off;
            
            function letter = readFirstLetter(snap)
                % Define the template images for specific characters
                templates = {
                    'A', imread('Template/A.bmp');
                    'M', imread('Template/M.bmp');
                    'B', imread('Template/B.bmp');
                    'N', imread('Template/N.bmp');
                    'C', imread('Template/C.bmp');
                    'P', imread('Template/P.bmp');
                    'R', imread('Template/R.bmp');
                    'F', imread('Template/F.bmp');
                    'T', imread('Template/T.bmp');
                    'W', imread('Template/W.bmp');
                    'V', imread('Template/V.bmp');
                    'J', imread('Template/J.bmp');
                    'K', imread('Template/K.bmp');
                };
                snap = imresize(snap, [42 24]);  % Resize snap to match the template size
                rec = [];
                % Perform correlation and find the best match
                for n = 1:size(templates, 1)
                    cor = corr2(templates{n, 2}, snap);
                    rec = [rec cor];
                end
                [~, ind] = max(rec);
                letter = templates{ind, 1};  % Get the letter corresponding to the highest correlation
            end
            
            function result = getState(letter)
                if strcmp(letter, 'A')
                    result = 'Perak';
                elseif strcmp(letter, 'M')
                    result = 'Malacca';
                elseif strcmp(letter, 'B')
                    result = 'Selangor';
                elseif strcmp(letter, 'N')
                    result = 'Negeri Sembilan';
                elseif strcmp(letter, 'C')
                    result = 'Pahang';
                elseif strcmp(letter, 'P')
                    result = 'Penang';
                elseif strcmp(letter, 'R')
                    result = 'Perlis';
                elseif strcmp(letter, 'F')
                    result = 'Putrajaya';
                elseif strcmp(letter, 'T')
                    result = 'Terengganu';
                elseif strcmp(letter, 'W') || strcmp(letter, 'V')
                    result = 'Kuala Lumpur';
                elseif strcmp(letter, 'J')
                    result = 'Johor';
                elseif strcmp(letter, 'K')
                    result = 'Kedah';
                elseif strcmp(letter, 'H')
                    result = 'Taxi';
                elseif strcmp(letter, 'Z')
                    result = 'Military';
                else
                    result = 'Unknown prefix';
                end
            end
        end

        % Button pushed function: MotorcycleButton
        function MotorcycleButtonPushed(app, event)
            path = handleVehicleSelection(app, 'Motorcycle');
            % Load the image
            image = imread(path);
            
            % Convert to YCbCr color space
            ycbcrImage = rgb2ycbcr(image);
            
            % Extract the Y channel
            yChannel = ycbcrImage(:,:,1);
            
            % Extract the V channel
            vChannel = ycbcrImage(:,:,3);
            
            % Edge detection using Canny
            img_edged = edge(yChannel, 'Canny');
            
            % ROI Extraction
            contours = bwboundaries(img_edged);
            
            hold on;
            
            max_area = -1;
            max_bounding_box = [];
            
            % Loop through each contour
            for k = 1:length(contours)
                boundary = contours{k};
                x = boundary(:, 2);
                y = boundary(:, 1);
                min_x = min(x);
                max_x = max(x);
                min_y = min(y);
                max_y = max(y);
                width = max_x - min_x + 1;
                height = max_y - min_y + 1;
            
                % Adjust aspect ratio range if needed
                aspect_ratio = width / height;
                if aspect_ratio < 1 || aspect_ratio > 9
                    continue;
                end
                % Extract the ROI
                roi = vChannel(min_y:max_y, min_x:max_x);
                mean_intensity = mean(roi(:));
                if mean_intensity < 0.3  
                    continue; 
                end
                % Area calculation and selection
                area = polyarea(boundary(:, 2), boundary(:, 1));
                if area > max_area
                    max_area = area;
                    margin = 5;
                    max_bounding_box = [min_x - margin, min_y - margin, width + 2*margin, height + 2*margin];
                end
            end
            
            % Draw the bounding box on the original image
            if ~isempty(max_bounding_box)
                subplot(1, 3, 1);
                imshow(image);
                title('Original Image');
                rectangle('Position', max_bounding_box, 'EdgeColor', 'g', 'LineWidth', 2);
            end
            hold off;
            
            % Extract the region of interest
            plate = imcrop(yChannel, max_bounding_box);
            
            % Resize the image
            resized_plate = imresize(plate, [200, NaN]); % Resize to a height of 100 pixels while maintaining aspect ratio
            
            % Contrast Enhancement
            enhanced_plate = adapthisteq(resized_plate); % Adaptive Histogram Equalization
            
            % Noise Reduction using Gaussian Filter
            filtered_plate = imgaussfilt(enhanced_plate, 3); % Gaussian filter with standard deviation of 1 pixel
            
            % Binarization
            binary_plate = filtered_plate > 175;
            
            % Dilation and Erosion
            se = strel('square', 1); % Define structuring element
            dilated_plate = imdilate(binary_plate, se);
            eroded_plate = imerode(dilated_plate, se);
            
            % Additional Noise Reduction
            cleaned_plate = bwareaopen(eroded_plate, 240); 
            
            % Label connected components in the binary image
            labeledImage = bwlabel(cleaned_plate);
            
            % Get properties of each labeled region
            props = regionprops(labeledImage, 'BoundingBox');
            
            % Initialize the state name variable
            state_name = 'Unknown';
            
            % Loop through each detected region and extract characters
            for k = 1 : min(1, length(props))
                thisBB = props(k).BoundingBox;
                % Extract the region of interest (character)
                characterImage = imcrop(cleaned_plate, thisBB);
                
                % Read the first letter
                letter = readFirstLetter(characterImage);
                
                % Check if the letter is an alphabet
                if isletter(letter)
                    state_name = getState(letter);
                    break; 
                end
            end
            
            % Display the images as subplots
            subplot(1, 3, 2);
            imshow(plate);
            title('Extracted Plate');
            
            subplot(1, 3, 3);
            imshow(cleaned_plate);
            title(sprintf('Template Detection\nFirst Character Detected: %s\nState: %s', letter, state_name));
            
            % Function to read the first letter
            function letter = readFirstLetter(characterImage)
                % Define the template images for specific characters
                templates = {
                    'A', imread('Template/A.bmp');
                    'M', imread('Template/M.bmp');
                    'B', imread('Template/B.bmp');
                    'N', imread('Template/N.bmp');
                    'C', imread('Template/C.bmp');
                    'P', imread('Template/P.bmp');
                    'R', imread('Template/R.bmp');
                    'F', imread('Template/F.bmp');
                    'T', imread('Template/T.bmp');
                    'W', imread('Template/W.bmp');
                    'V', imread('Template/V.bmp');
                    'J', imread('Template/J.bmp');
                    'K', imread('Template/K.bmp');
                };
                
                % Resize the character image to match the template size
                snap = imresize(characterImage, [42 24]);  
                rec = [];
                
                % Perform correlation and find the best match
                for n = 1:size(templates, 1)
                    cor = corr2(templates{n, 2}, snap);
                    rec = [rec cor];
                end
                [~, ind] = max(rec);
                letter = templates{ind, 1};  % Get the letter corresponding to the highest correlation
            end
            
            % Function to get the state name from the letter
            function result = getState(letter)
                if strcmp(letter, 'A')
                    result = 'Perak';
                elseif strcmp(letter, 'M')
                    result = 'Malacca';
                elseif strcmp(letter, 'B')
                    result = 'Selangor';
                elseif strcmp(letter, 'N')
                    result = 'Negeri Sembilan';
                elseif strcmp(letter, 'C')
                    result = 'Pahang';
                elseif strcmp(letter, 'P')
                    result = 'Penang';
                elseif strcmp(letter, 'R')
                    result = 'Perlis';
                elseif strcmp(letter, 'F')
                    result = 'Putrajaya';
                elseif strcmp(letter, 'T')
                    result = 'Terengganu';
                elseif strcmp(letter, 'W') || strcmp(letter, 'V')
                    result = 'Kuala Lumpur';
                elseif strcmp(letter, 'J')
                    result = 'Johor';
                elseif strcmp(letter, 'S')
                    result = 'Sabah';
                elseif strcmp(letter, 'Q')
                    result = 'Sarawak';
                elseif strcmp(letter, 'K')
                    result = 'Kedah';
                elseif strcmp(letter, 'H')
                    result = 'Taxi';
                elseif strcmp(letter, 'Z')
                    result = 'Military';
                else
                    result = 'Unknown prefix';
                end
            end
        end

        % Button pushed function: Motorcycle500ccButton_2
        function CarButton_2Pushed(app, event)
            path = handleVehicleSelection(app, 'Cars');
            % Read the input image
            img = imread(path);
            
            % Convert the image to grayscale
            grayImg = rgb2gray(img);
            
            % Perform global thresholding to obtain the binary threshold value
            imgThreshold = graythresh(grayImg);
            
            % Binarize the grayscale image using the obtained threshold value
            segmentedImg = imbinarize(grayImg, imgThreshold);
            
            % Find connected components in the binarized image
            cc = bwconncomp(segmentedImg);
            
            % Get properties of the connected regions (area and bounding box)
            stats = regionprops("table", cc, "Area", "BoundingBox");
            area = stats.Area;
            bbox = stats.BoundingBox;
            
            % Calculate the center of each bounding box
            center_x = bbox(:, 1) + bbox(:, 3) / 2;
            center_y = bbox(:, 2) + bbox(:, 4) / 2;
            
            % Define the region of interest (bottom center region of the image)
            bottomCenterRegion = (center_y <= size(img, 1) * 0.9) & ...
                                 (center_y >= size(img, 1) * 0.6) & ...
                                 (center_x >= size(img, 2) * 0.2) & ...
                                 (center_x <= size(img, 2) * 0.8);
            
            % Initialize variable to store the selected bounding boxes
            selectedBoundingBoxes = [];
            % Iterate through each bounding box found
            for i = 1:size(bbox, 1)
                % Filter bounding boxes based on criteria
                if (area(i) > 30) && ...
                   (bbox(i, 3) < 35) && ...
                   (bbox(i, 4) >= 15 && bbox(i, 4) <= 85) && ...
                   bottomCenterRegion(i)
               
                    % If no bounding box is selected yet, add the first one
                    if isempty(selectedBoundingBoxes)
                        selectedBoundingBoxes = [selectedBoundingBoxes; bbox(i, :)];
                        continue;
                    end
            
                    % Get threshold coordinates for proximity to previous bounding box
                    threshold_x = selectedBoundingBoxes(end, 1) + selectedBoundingBoxes(end, 3) + 40;
                    threshold_y = selectedBoundingBoxes(end, 2) + selectedBoundingBoxes(end, 4) + 40;
            
                    % Add bounding boxes that are near the previous bounding box
                    if (bbox(i, 1) < threshold_x) && (bbox(i, 2) < threshold_y)
                        selectedBoundingBoxes = [selectedBoundingBoxes; bbox(i, :)];
                    end  
                end
            end
            
            % Check if no bounding boxes match the criteria
            if isempty(selectedBoundingBoxes)
                error('No bounding boxes found that match the criteria.');
            end
            
            % Calculate the coordinates for the final bounding box encompassing all selected bounding boxes
            min_x = min(selectedBoundingBoxes(:, 1));
            max_x = max(selectedBoundingBoxes(:, 1) + selectedBoundingBoxes(:, 3));
            min_y = min(selectedBoundingBoxes(:, 2));
            max_y = max(selectedBoundingBoxes(:, 2) + selectedBoundingBoxes(:, 4));
            
            % Calculate width and height of the bounding box
            w = max_x - min_x;
            h = max_y - min_y;
            
            % Define the region of interest (ROI) for OCR based on the first bounding box
            roi_min_x = min(selectedBoundingBoxes(1, 1));
            roi_max_x = max(roi_min_x + selectedBoundingBoxes(1, 3));
            roi_min_y = min(selectedBoundingBoxes(1, 2));
            roi_max_y = max(roi_min_y + selectedBoundingBoxes(1, 4));
            roi = [roi_min_x - 5,  roi_min_y - 5, roi_max_x - roi_min_x + 10, roi_max_y - roi_min_y + 10];
            
            % Get the state/prefix from the image using the defined region
            [state, prefix] = getState(img, [min_x - 20, min_y - 10, w + 50, h + 15]);
            
            
            % Create a figure for the subplots
            figure('Name', 'License Plate Detection', 'NumberTitle', 'off');
            
            % Subplot 1: Display the original image with highlighted bounding boxes
            subplot(2, 2, 1);
            imshow(img);
            hold on;
            % Red rectangle for the license plate
            rectangle('Position', [min_x - 20, min_y - 10, w + 50, h + 15], 'EdgeColor', 'r', 'LineWidth', 2);
            % Yellow rounded rectangle for the ROI
            rectangle('Position', roi, 'EdgeColor', 'y', 'LineWidth', 2, 'Curvature', [1 1]); 
            hold off;
            title('Original Image with Bounding Boxes');
            
            % Subplot 2: Display the cropped license plate image
            croppedLicense = imcrop(img, [min_x - 20, min_y - 10, w + 50, h + 15]);
            subplot(2, 2, 2);
            imshow(croppedLicense);
            title('Cropped License Plate');
            
            % Subplot 3: Display the recognized text information
            subplot(2, 2, 3);
            axis off;
            text(0.1, 0.6, ['Prefix: ' prefix], 'FontSize', 12);
            text(0.1, 0.4, ['State: ' state], 'FontSize', 12);
            title('Recognized Text Information');
            % Perform OCR on the cropped image and find the first alphabet for state identification
            testing = ocr(croppedLicense);
            recognizedText = testing.Text;
            disp(['Recognized Text: ', recognizedText]);
            
            % Iterate through the recognized text to find the first alphabet
            stateFound = false;
            for i = 1:length(recognizedText)
                if isletter(recognizedText(i))
                    [state, prefix] = getStateFromPrefix(recognizedText(i));
                    stateFound = true;
                    break;
                end
            end
            
            % If no valid state prefix is found, set state to 'State not found'
            if ~stateFound
                state = 'State not found';
                prefix = '';
            end
            
            % Subplot 4: Display the state information based on the recognized text
            subplot(2, 2, 4);
            axis off;
            if stateFound
                text(0.1, 0.7, ['Recognized Text: ' recognizedText], 'FontSize', 12);
                text(0.1, 0.5, ['Prefix: ' prefix], 'FontSize', 12); % Display the prefix
                text(0.1, 0.3, ['State: ' state], 'FontSize', 12);
            else
                text(0.1, 0.5, 'State not found', 'FontSize', 12);
            end
            title('Raw OCR Result');
            
            % Display the recognized text and state in the console
            disp(['Recognized Text: ', recognizedText]);
            disp(['Prefix: ', prefix]);
            disp(['State: ', state]);
            
            % Function to get the state/prefix from the license plate region
            function [result, prefix] = getState(image, roi)
                ocrResults = ocr(image, roi);
               disp(ocrResults);
                if ocrResults.Text == ""
                    result = 'Unknown prefix';
                    prefix = 'Unrecognizable';
                    return;
                end
            
                prefix = ocrResults.Text(1);
            
                % Determine the state based on the prefix
                [result, prefix] = getStateFromPrefix(prefix);
            end
            
            % Function to get state from a prefix character
            function [result, prefix] = getStateFromPrefix(prefix)
                switch prefix
                    case 'A'
                        result = 'Perak';
                    case 'M'
                        result = 'Malacca';
                    case 'B'
                        result = 'Selangor';
                    case 'N'
                        result = 'Negeri Sembilan';
                    case 'C'
                        result = 'Pahang';
                    case 'P'
                        result = 'Penang';
                    case 'R'
                        result = 'Perlis';
                    case 'F'
                        result = 'Putrajaya';
                    case 'T'
                        result = 'Terengganu';
                    case {'W', 'V', 'w'}
                        result = 'Kuala Lumpur';
                    case 'J'
                        result = 'Johor';
                    case 'K'
                        result = 'Kedah';
                    case 'H'
                        result = 'Taxi';
                    case 'Z'
                        result = 'Military';
                    case 'S'
                        result = 'Private and Commercial Vehicle';
                    otherwise
                        result = 'Unknown prefix';
                end
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 400 300];
            app.UIFigure.Name = 'Select Vehicle Type';

            % Create SelectVehicleTypePanel
            app.SelectVehicleTypePanel = uipanel(app.UIFigure);
            app.SelectVehicleTypePanel.Title = 'Select Vehicle Type';
            app.SelectVehicleTypePanel.Position = [20 20 360 260];

            % Create TaxiButton
            app.TaxiButton = uibutton(app.SelectVehicleTypePanel, 'push');
            app.TaxiButton.ButtonPushedFcn = createCallbackFcn(app, @TaxiButtonPushed, true);
            app.TaxiButton.Position = [20 200 150 30];
            app.TaxiButton.Text = 'Taxi';

            % Create TruckButton
            app.TruckButton = uibutton(app.SelectVehicleTypePanel, 'push');
            app.TruckButton.ButtonPushedFcn = createCallbackFcn(app, @TruckButtonPushed, true);
            app.TruckButton.Position = [190 200 150 30];
            app.TruckButton.Text = 'Truck';

            % Create MinibusVanButton
            app.MinibusVanButton = uibutton(app.SelectVehicleTypePanel, 'push');
            app.MinibusVanButton.ButtonPushedFcn = createCallbackFcn(app, @MinibusVanButtonPushed, true);
            app.MinibusVanButton.Position = [20 140 150 30];
            app.MinibusVanButton.Text = 'Minibus/Van';

            % Create BusButton
            app.BusButton = uibutton(app.SelectVehicleTypePanel, 'push');
            app.BusButton.ButtonPushedFcn = createCallbackFcn(app, @BusButtonPushed, true);
            app.BusButton.Position = [190 140 150 30];
            app.BusButton.Text = 'Bus';

            % Create Motorcycle500ccButton
            app.Motorcycle500ccButton = uibutton(app.SelectVehicleTypePanel, 'push');
            app.Motorcycle500ccButton.ButtonPushedFcn = createCallbackFcn(app, @MotorcycleButtonPushed, true);
            app.Motorcycle500ccButton.Position = [20 80 150 30];
            app.Motorcycle500ccButton.Text = 'Motorcycle';

            % Create Motorcycle500ccButton_2
            app.Motorcycle500ccButton_2 = uibutton(app.SelectVehicleTypePanel, 'push');
            app.Motorcycle500ccButton_2.ButtonPushedFcn = createCallbackFcn(app, @CarButton_2Pushed, true);
            app.Motorcycle500ccButton_2.Position = [190 80 150 30];
            app.Motorcycle500ccButton_2.Text = 'Car';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = ImageUpload_GUI

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end
