function Traffic_sign_detection_and_recognition
	clc; % Clear the command window.
	close all; % Close all figures (except those of imtool.)
    clear all;
    shape_recognition();

    
function shape_recognition
try
	workspace; % Make sure the workspace panel is showing.
	fontSize = 12;

	% For reference, compute the theoretical circularity of a bunch of regular polygons
	% with different number of sides starting with 3 (triangle).
	dividingValues = PlotTheoreticalCircularity;
	
	% Make the last dividing value infinity because any circularity from .99999 up to inifinity should be a circle.
	% and sometimes you have a circularity more than 1 due to quantization errors.
	dividingValues(end) = inf;
	
	% Now create a demo image.
	[originalImage, binaryImage] = ImportImage;

	% Count the number of shapes
	[~, numShapes] = bwlabel(binaryImage);
	
	% Display the image.
    figure;
	subplot(1, 2, 1);
	imshow(binaryImage);
	caption = sprintf('Image with %d Shapes', numShapes);
	title(caption, 'FontSize', fontSize);
	hold on; % So that text labels won't blow away the image.
	
	% Set up figure properties:
	% Enlarge figure to full screen.
	%set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.5 1]);
	% Get rid of tool bar and pulldown menus that are along top of figure.
	%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
	% Give a name to the title bar.
	set(gcf, 'Name', 'Shapes', 'NumberTitle', 'Off')
	drawnow; % Make it display immediately.
	
	[labeledImage, numberOfObjects] = bwlabel(binaryImage);

    % labeledImage is an integer-valued image where all pixels in the blobs have values of 1, or 2, or 3, or ... etc.
    subplot(1, 2, 1);
    imshow(labeledImage, []);  % Show the labeled binary image.
    title('Labeled Image, from bwlabel()', 'FontSize', fontSize);
    drawnow;
    % Assign each blob a different color to visually show the user the distinct blobs.
    coloredLabels = label2rgb (labeledImage, 'hsv', 'k', 'shuffle'); % pseudo random color labels
    % coloredLabels is an RGB image.  We could have applied a colormap instead (but only with R2014b and later)
    subplot(1, 2, 1);
    imshow(coloredLabels);
    axis image; % Make sure image is not artificially stretched because of screen's aspect ratio.
    caption = sprintf('Pseudo colored labels, from label2rgb().');
    title(caption, 'FontSize', fontSize);

	blobMeasurements = regionprops(labeledImage, 'Perimeter', 'Area', 'Centroid', 'Image', 'BoundingBox');

	
	% Compute the number of vertices by looking at the number of peaks in a plot of distance from centroid.
	numSidesDistance = FindNumberOfVertices(blobMeasurements, labeledImage);
	
	% Get all the measurements into single arrays for convenience.
	allAreas = [blobMeasurements.Area];
	allPerimeters = [blobMeasurements.Perimeter];	
    circularities = 4 * pi * allAreas./allPerimeters.^2;
   


% 	% Sort in order of increasing circularity
% 	[sortedCircularities, sortOrder] = sort(circularities, 'Ascend');
% 	% Sort all the measurements in the same way.
% 	blobMeasurements = blobMeasurements(sortOrder);
% 	allAreas = allAreas(sortOrder);
% 	allPerimeters = allPerimeters(sortOrder);
% 	numSidesDistance = numSidesDistance(sortOrder);

%     %------------------------------------------------------------------------------------------------------------------------------------------------------
%     % Get centroids into an N -by-2 array directly from props,
%     % rather than accessing them as a field of the props strcuture array.
%     % We can get the centroids of ALL the blobs into 2 arrays,
%     % one for the centroid x values and one for the centroid y values.
%     allBlobCentroids = vertcat(blobMeasurements.Centroid);		% A 10 row by 2 column array of (x,y) centroid coordinates.
%     centroidsX = allBlobCentroids(:, 1);			% Extract out the centroid x values into their own vector.
%     centroidsY = allBlobCentroids(:, 2);			% Extract out the centroid y values into their own vector.
%     % Put the labels on the rgb labeled image also.
%     subplot(1, 2, 1);
%     for n = 1 : numberOfObjects           % Loop through all blobs.
% 	    % Place the blob label number at the centroid of the blob.
% 	    text(centroidsX(n), centroidsY(n)+10, num2str(n), 'FontSize', 14, 'FontWeight', 'Bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
%     end
%     %------------------------------------------------------------------------------------------------------------------------------------------------------

	
	% Plot a bar chart of the circularities.
	subplot(1, 2, 2);
	bar(circularities);
	ylim([0.55, 1.1]);
	grid on;
	title('Actual Measured Circularities', 'FontSize', fontSize);
	
% 	% Let's compute areas a different way.  The "Area" returned by regionprops is a count of the number of pixels.
% 	% This sometimes overestimates the area.  Let's use bwarea, which computes the area on a
% 	% pixel-center to pixel center basis.
% 	for k = 1 : numberOfObjects
% 		thisBlob = blobMeasurements(k).Image;
% 		allBwAreas(k) = bwarea(thisBlob);
% 	end
% 	bwCircularities = (4 * pi *  allBwAreas) ./ allPerimeters.^2
% 	sortedCircularities = bwCircularities
	
% 	% Put up red horizontal lines at the dividing values
% 	hold on;
% 	xl = xlim();
% 	for k = 1 : length(numSidesCircularity)-1
% 		thisSideLength = numSidesCircularity(k);
% 		thisDividingValue = dividingValues(thisSideLength);
% 		line(xl, [thisDividingValue, thisDividingValue], 'Color', 'r');
% 		% For the first 6, print the dividing value at the left just above the line.
% 		% After 6 it would get too crowded
% 		if k <= 6
% 			theLabel = sprintf('Dividing value = %.4f', thisDividingValue);
% 			text(xl(1)+0.1, thisDividingValue + 0.005, theLabel, 'Color', 'r');
% 		end
% 	end
	
	% Explain why the labels may not be accurate.
	message = sprintf('\nBefore we start classifying the shapes,\nnote that the circularity may deviate from the theoretical circularity\ndepending on the size, rotation, and the algorithm\nused to compute area and perimeter.');
	fprintf('%s\n', message);
	uiwait(helpdlg(message));
	
	% Say the possible shape they are, one by one.
	subplot(1, 2, 1);
	for blobNumber = 1 : numberOfObjects
		%==============================================================
		% Determine the number of sizes according to the circularity
		% Get the circularity of this specific blob.
		thisCircularity = circularities(blobNumber);
		% See which theoretical dividing value it's less than.
		% This will determine the number of sides it has.
		numSidesCircularity = find(thisCircularity < dividingValues, 1, 'first');
		% Assign a string naming the shape according to the distance algorithm.
		if numSidesCircularity == 3
			% Blob has 3 sides.
			theShapeCirc = 'triangle';
		elseif numSidesCircularity == 4
			% Blob has 4 sides.
			theShapeCirc = 'square';
% 		elseif numSidesCircularity == 5
% 			% Blob has 5 sides.
% 			theShapeCirc = 'pentagon';
% 		elseif numSidesCircularity == 6
% 			% Blob has 6 sides.
% 			theShapeCirc = 'hexagon';
        elseif numSidesCircularity == 8
            %Blob has 8 sides.
			theShapeCirc = 'octagon';
        else% numSidesCircularity > 8
			% Blob has 9 or more sides.
			theShapeCirc = 'nearly circular';
%         else
%             % Blob is not a specific shape.
% 			theShapeCirc = 'not a shape';
		end		
		
		%==============================================================
		% Determine the number of sizes according to the centroid-to-perimeter algorithm
		% Classify the shape by the centroid-to-perimeter algorithm which seems to be more accurate than the circularity algorithm.
		numSidesDist = numSidesDistance(blobNumber);
		% Assign a string naming the shape according to the distance algorithm.
		if numSidesDist == 3
			% Blob has 3 sides.
			theShapeDistance = 'triangle';
		elseif numSidesDist == 4
			% Blob has 4 sides.
			theShapeDistance = 'square';
% 		elseif numSidesDist == 5
% 			% Blob has 5 sides.
% 			theShapeDistance = 'pentagon';
% 		elseif numSidesDist == 6
% 			% Blob has 6 sides.
% 			theShapeDistance = 'hexagon';
        elseif numSidesDist == 8
            %Blob has 8 sides.
			theShapeDistance = 'octagon';
        else %numSidesDist > 8
			% Blob has 9 or more sides.
			theShapeDistance = 'nearly circular';
%         else
%             % Blob is not a specific shape.
% 			theShapeDistance = 'not a shape';
		end		
		
		% Place a label on the shape
		xCentroid = blobMeasurements(blobNumber).Centroid(1);
		yCentroid = blobMeasurements(blobNumber).Centroid(2);
		blobLabel = sprintf('#%d = %s', blobNumber, theShapeDistance);
		plot(xCentroid, yCentroid, 'w+', 'LineWidth', 2, 'MarkerSize', 15);
		text(xCentroid+20, yCentroid, blobLabel, 'FontSize', fontSize, 'Color', 'w', 'FontWeight', 'Bold');

		% Inform the user what the circularity and shape are.
		distanceMessage = sprintf('The centroid-to-perimeter algorithm predicts shape #%d has %d sides, so it predicts the shape is a %s', blobNumber, numSidesDistance(blobNumber), theShapeDistance);
		circMessage = sprintf('The circularity of object #%d is %.3f, so the circularity algorithm predicts the object is a %s shape.\nIt is estimated to have %d sides.\n(Range for %s is [%.4f - %.4f].)',...
			blobNumber, thisCircularity, theShapeCirc, numSidesCircularity, ...
			theShapeDistance, dividingValues(numSidesCircularity - 1), dividingValues(numSidesCircularity));
		% See if the number of sides determined each way agrees with each other.
		if numSidesDistance(blobNumber) == numSidesCircularity
			agreementMessage = sprintf('For blob #%d, the two algorithms agree on %d sides.', blobNumber, numSidesCircularity);
		else
			agreementMessage = sprintf('For blob #%d there is disagreement.', blobNumber);
		end
		% Combine all messages into one.
		promptMessage = sprintf('%s\n\n%s\n\n%s', distanceMessage, circMessage, agreementMessage);
		fprintf('%s\n', promptMessage);
		
		% Give user an opportunity to bail out if they want to.
		titleBarCaption = 'Continue?';
		button = questdlg(promptMessage, titleBarCaption, 'Continue', 'Quit', 'Continue');
		if strcmpi(button, 'Quit')
			return;
		end
    end

    % Crop each ROI out to a separate sub-image on a new figure.
    message = sprintf('Would you like to crop out each ROI to individual images?');
    reply = questdlg(message, 'Extract Individual Images?', 'Yes', 'No', 'Yes');
    % Note: reply will = '' for Upper right X, 'Yes' for Yes, and 'No' for No.
    if strcmpi(reply, 'Yes')

        figure;
	    caption = sprintf('Image with %d Shapes', numShapes);
	    title(caption, 'FontSize', fontSize);
	    hold on; % So that text labels won't blow away the image.
	    
	    % Set up figure properties:
	    % Enlarge figure to full screen.
	    %set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.5 1]);
	    % Get rid of tool bar and pulldown menus that are along top of figure.
	    %set(gcf, 'Toolbar', 'none', 'Menu', 'none');
	    % Give a name to the title bar.
	    set(gcf, 'Name', 'Possible Traffic Signs', 'NumberTitle', 'Off')
	    %drawnow; % Make it display immediately.
    
    
    % 	    % Maximize the figure window.
    % 	    hFig2 = figure;	% Create a new figure window.
    % 	    hFig2.Units = 'normalized';
    % 	    %hFig2.WindowState = 'maximized'; % Go to full screen.
    % 	    hFig2.NumberTitle = 'off'; % Get rid of "Figure 1"
    % 	    hFig2.Name = 'Possible Traffic Signs'; % Put this into title bar.
	    
        for k = 1 : numberOfObjects		% Loop through all blobs.
		    % Find the bounding box of each blob.
		    thisBlobsBoundingBox = blobMeasurements(k).BoundingBox;  % Get list of pixels in current blob.
		    % Extract out this coin into it's own image.
		    subImage = imcrop(originalImage, thisBlobsBoundingBox);
            target_images{k}=subImage;
		    subplot(3, 4, k);
		    imshow(target_images{k});
            title(k);
        end

        template_images_struct = load('Images/template_images.mat');
        template_images = template_images_struct.template_images;
        figure;
	    caption = sprintf('Image with %d Shapes', numShapes);
	    title(caption, 'FontSize', fontSize);
	    hold on; % So that text labels won't blow away the image.
	    
	    % Set up figure properties:
	    % Enlarge figure to full screen.
	    %set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.5 1]);
	    % Get rid of tool bar and pulldown menus that are along top of figure.
	    %set(gcf, 'Toolbar', 'none', 'Menu', 'none');
	    % Give a name to the title bar.
	    set(gcf, 'Name', 'Recognized Traffic Signs', 'NumberTitle', 'Off');
	    %drawnow; % Make it display immediately.

        for j = 1 : length(target_images)
            % Load the target image and the template image
            target_image = target_images{j};
            target_image = imresize(target_image, [120 120]);

            % Split the target image into its color channels
            target_image_red = target_image(:,:,1);
            target_image_green = target_image(:,:,2);
            target_image_blue = target_image(:,:,3);
            
            A = cell(numel(template_images), 2);
            
            
            % Loop through each image file
            for i = 1:numel(template_images)
            
                % Read in the image file
                filename = template_images(i).name;
                % Use fullfile to create the full file path
                filepath = fullfile('./Traffic_signs', filename);
                template_image = imread(filepath);
                A{i, 1} = string(filename);
                
                
                % Split the template image into its color channels
                template_image_red = template_image(:,:,1);
                template_image_green = template_image(:,:,2);
                template_image_blue = template_image(:,:,3);
                
                % Compute the cross-correlation between the template and the target image for each channel
                c_red = normxcorr2(template_image_red, target_image_red);
                c_green = normxcorr2(template_image_green, target_image_green);
                c_blue = normxcorr2(template_image_blue, target_image_blue);
                
                % Combine the cross-correlation maps from each channel
                c = c_red + c_green + c_blue;
                c = c/3;
                
                % Reshape the images into column vectors
                template_vec = reshape(template_image, [], 1);
                target_vec = reshape(target_image, [], 1);
                
                % Concatenate the column vectors into a matrix
                X = [template_vec, target_vec];
                
                % Convert the matrix to double precision
                X = double(X);
                
                % Normalize the matrix to have zero mean and unit variance
                X = zscore(X);
                
                % Compute the correlation coefficient matrix between the template and the target image
                R = corrcoef(X);
                
                % Extract the correlation coefficient from the matrix
                A{i,2} = R(1,2);
                
%                 figure;
%                 surf(c);
%                 shading flat;
%                 title(filepath);
                
                % Look for locations where the cross-correlation is high
                [max_c, imax] = max(abs(c(:)));
                [ypeak, xpeak] = ind2sub(size(c),imax(1));
                yoffset = ypeak-size(template_image,1);
                xoffset = xpeak-size(template_image,2);
                
                % % Display the cross-correlation map
                % figure;
                % imshow(c, []);
                % title('Cross-Correlation Map');
            end
            
            % Find the maximum value of correlation coefficient matrix
            [max_val, max_idx] = max(cell2mat(A(:,2)));
            if max_val < 0.1
                fprintf('----------------------------------------------------------------\n')
                fprintf('No Sign detected\n');
                filePath = fullfile('./Images/', 'notSign.png');
                imshow(filePath);
                title('No Sign detected');
            else
            
                % Get the position of the maximum value
                [row, col] = ind2sub(size(A), max_idx);
                
                % % Print the maximum value and its position
                fprintf('----------------------------------------------------------------\n')
                fprintf('Maximum value: %d (at position %d)\n', max_val, max_idx);
                fprintf('Possible Traffic Sign: ' + A{row,1} + '\n');
                template_extracted{j} = A{row,1};
    
                % Read in the image file
                fileName = A{row,1};
                % Use fullfile to create the full file path
                filePath = fullfile('./Traffic_signs', fileName);
                subplot(3, 4, j);
                imshow(filePath);
                title(fileName);
            end



        end
    
    end

	uiwait(helpdlg('Done with demo!'));
catch ME
	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
		ME.stack(1).name, ME.stack(1).line, ME.message);
	fprintf(1, '%s\n', errorMessage);
	uiwait(warndlg(errorMessage));
end

% %----------------------------------------------------------------------------------------------------------------------------------
% % Creates an image with a specified number of circles, triangles, rectangles, and pentagons
% function [binaryImage, numSides] = CreateDemoImage()
% try
% 	rows = 800;
% 	columns = round(rows * 3/4); % 4/3 aspect ratio.
% 	figure;
% 	
% 	% Create an image and add in some triangles at various angles and various sizes.
% 	binaryImage = false(rows, columns);
% 	numShapesToPlace = 3;
% 	for numSides = 3 : 6
% 		shapesPlacedSoFar = 0;
% 		centroidToVertexDistance = [30, 75];
% 		% Define fail-safe parameters.
% 		maxNumberOfAttempts = 50;
% 		numberOfAttempts = 0;
% 		while shapesPlacedSoFar < numShapesToPlace && numberOfAttempts < maxNumberOfAttempts
% 			thisBinaryImage = CreatePolygon(numSides, centroidToVertexDistance, rows, columns);
% 			% Sometimes two polygons will be next to each other but not overlapping.
% 			% However bwlabel() and bwconncomp() would consider those two regions as being the same region.
% 			% To check for and prevent that kind of situation (which happened to me once),
% 			% we need to dilate the binary image by one layer before checking for overlap.
% 			dilatedImage = imdilate(thisBinaryImage, true(9));
% 			% See if any pixels in this binary image overlap any existing pixels.
% 			overlapImage = binaryImage & dilatedImage;
% 			if ~any(overlapImage(:))
% 				% No pixels overlap, so OR in this image.
% 				binaryImage = binaryImage | thisBinaryImage;
% 				shapesPlacedSoFar = shapesPlacedSoFar + 1;
% 			else
% 				fprintf('Skipping attempt %d because of overlap.\n', numberOfAttempts);
% 			end
% 			numberOfAttempts = numberOfAttempts + 1;
% 		end
% 	end
% 	
% 	% Create an image and add in some circles at various angles and various sizes.
% 	numShapesToPlace = 3;
% 	shapesPlacedSoFar = 0;
% 	numSides = 30; % Pretty round
% 	centroidToVertexDistance = [30, 75];
% 	while shapesPlacedSoFar < numShapesToPlace && numberOfAttempts < maxNumberOfAttempts
% 		thisBinaryImage = CreatePolygon(numSides, centroidToVertexDistance, rows, columns);
% 		% See if any pixels in this binary image overlap any existing pixels.
% 		overlapImage = binaryImage & thisBinaryImage;
% 		if ~any(overlapImage(:))
% 			% No pixels overlap, so OR in this image.
% 			binaryImage = binaryImage | thisBinaryImage;
% 			shapesPlacedSoFar = shapesPlacedSoFar + 1;
% 		end
% 		numberOfAttempts = numberOfAttempts + 1;
% 	end
% 	
% 	% Pass back the number of sides we decided to use.
% 	numSides = [3:6, 30];
% catch ME
% 	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
% 		ME.stack(1).name, ME.stack(1).line, ME.message);
% 	fprintf(1, '%s\n', errorMessage);
% 	uiwait(warndlg(errorMessage));
% end
% 
% %----------------------------------------------------------------------------------------------------------------------------------
% % Create a single polygon with the specified number of sides in a binary image of the specified number of rows and columns.
% % centroidToVertexDistance is the distance from the centroid to each vertex.
% % If centroidToVertexDistance is a length 2 vector, then this indicated the minimum and maximum size range and
% % it will create a random size polygon between the min and max distance.
% function binaryImage = CreatePolygon(numSides, centroidToVertexDistance, rows, columns)
% try
% 	% Get the range for the size from the center to the vertices.
% 	if length(centroidToVertexDistance) > 1
% 		% Random size between a min and max distance.
% 		minDistance = centroidToVertexDistance(1);
% 		maxDistance = centroidToVertexDistance(2);
% 	else
% 		% All the same size.
% 		minDistance = centroidToVertexDistance;
% 		maxDistance = centroidToVertexDistance;
% 	end
% 	thisDistance = (maxDistance - minDistance) * rand(1) + minDistance;
% 	
% 	% Create a polygon around the origin
% 	for v = 1 : numSides
% 		angle = v * 360 / numSides;
% 		x(v) = thisDistance * cosd(angle);
% 		y(v) = thisDistance * sind(angle);
% 	end
% 	% Make last point the same as the first
% 	x(end+1) = x(1);
% 	y(end+1) = y(1);
% 	% 	plot(x, y, 'b*-', 'LineWidth', 2);
% 	% 	grid on;
% 	% 	axis image;
% 	
% 	% Rotate the coordinates by a random angle between 0 and 360
% 	angleToRotate = 360 * rand(1);
% 	rotationMatrix = [cosd(angleToRotate), sind(angleToRotate); -sind(angleToRotate), cosd(angleToRotate)];
% 	% Do the actual rotation
% 	xy = [x', y']; % Make a numSides*2 matrix;
% 	xyRotated = xy * rotationMatrix; % A numSides*2 matrix times a 2*2 = a numSides*2 matrix.
% 	x = xyRotated(:, 1); % Extract out the x as a numSides*2 matrix.
% 	y = xyRotated(:, 2); % Extract out the y as a numSides*2 matrix.
% 	
% 	% Get a random center location between centroidToVertexDistance and (columns - centroidToVertexDistance).
% 	% This will ensure it's always in the image.
% 	xCenter = thisDistance + (columns - 2 * thisDistance) * rand(1);
% 	% Get a random center location between centroidToVertexDistance and (rows - centroidToVertexDistance).
% 	% This will ensure it's always in the image.
% 	yCenter = thisDistance + (rows - 2 * thisDistance) * rand(1);
% 	% Translate the image so that the center is at (xCenter, yCenter) rather than at (0,0).
% 	x = x + xCenter;
% 	y = y + yCenter;
% 	binaryImage = poly2mask(x, y, rows, columns);
% catch ME
% 	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
% 		ME.stack(1).name, ME.stack(1).line, ME.message);
% 	fprintf(1, '%s\n', errorMessage);
% 	uiwait(warndlg(errorMessage));
% end

%----------------------------------------------------------------------------------------------------------------------------------
% https://en.wikipedia.org/wiki/Regular_polygon
% Which says A = (1/4) * n * s^2 * cot(pi/n)
function circularity = ComputeTheoreticalCircularity(numSides)
try
	sideLength = 1;
	perimeter = numSides * sideLength;
	area = (1/4) * numSides * sideLength^2 / tan(pi / numSides);
	circularity = (4 * pi * area) / perimeter ^2;
catch ME
	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
		ME.stack(1).name, ME.stack(1).line, ME.message);
	fprintf(1, '%s\n', errorMessage);
	uiwait(warndlg(errorMessage));
end

%----------------------------------------------------------------------------------------------------------------------------------
% Makes a figure with the theoretical circularity for a bunch of different number of sides plotted.
function dividingValues = PlotTheoreticalCircularity()
try
    figure;
	dividingValues = []; % Initialize
	fontSize = 12;
	% For reference, compute the theoretical circularity of a bunch of regular polygons with different number of sides.
	fprintf('Number of Sides     Theoretical Circularity\n');
	% Define an array with the number of sides we want to compute the circularity for.
	numSides = 3 : 16;
	for k = 1 : length(numSides)
		thisSideLength = numSides(k);
		% Compute the theoretically perfect circularity, if the polygons were perfect instead of digitized.
		circularity(k) = ComputeTheoreticalCircularity(thisSideLength);
	end
	% Plot the theoretical circularities on the curve with a cross.
	plot(numSides, circularity, 'b+-', 'LineWidth', 2, 'MarkerSize', 20);
	grid on;
	hold on;
	xl = xlim(); % Get left and right x coordinates of the graph.
	% Plot theoretical lines in dark red.
	darkRed = [0.85, 0, 0];
	for k = 1 : length(numSides)
		% Make theoretical line on the plot in a magenta color.
		line(xl, [circularity(k), circularity(k)], 'Color', darkRed, 'LineWidth', 2);
		fprintf('     %d                  %f\n', thisSideLength, circularity(k));
		if k < 7 % Only print text if it's not too crowded and close together.
			% Make text with the true value
			message = sprintf('Theoretical value for %d sides = %.4f', thisSideLength, circularity(k));
			text(xl(1)+0.1, circularity(k) + 0.005, message, 'Color', darkRed);
		end
	end
	
	% Set up figure properties:
	% Enlarge figure to full screen.
	%set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
	% Get rid of tool bar and pulldown menus that are along top of figure.
	%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
	% Give a name to the title bar.
	set(gcf, 'Name', 'Theoretical Circularities', 'NumberTitle', 'Off')
	drawnow; % Make it display immediately.
	
	title('Theoretical Circularities', 'FontSize', fontSize, 'Interpreter', 'None');
	xlabel('Number of Sides', 'FontSize', fontSize);
	ylabel('True Circularity', 'FontSize', fontSize);
	
	% Get the midpoint between one circularity and the one for the next higher number of sides.
	dividingValues = conv(circularity, [1, 1]/2, 'valid');
	% Prepend two zeros so that we can use this array as a lookup table where we pass in
	% the number of sides as an index and it tells us the dividing value between
	% that number of sides and one more than that.
	% For example, right now dividingValues(1) gives us the dividing value between 3 and 4
	% and dividingValues(3) gives us the dividing value between 5 and 6 (instead of between 3 and 4).
	dividingValues = [0, 0, dividingValues];
	% Now dividingValues(3) will give us the dividing value between 3 and 4.
	
	% Put up red horizontal lines at the dividing values
	hold on;
	xl = xlim();
	darkGreen = [0, 0.5, 0];
	for k = 1 : length(numSides)-1
		thisSideLength = numSides(k);
		thisDividingValue = dividingValues(thisSideLength);
		h = line(xl, [thisDividingValue, thisDividingValue], 'Color', darkGreen, 'LineWidth', 2, 'LineStyle', '--');
		% 		h.LineStyle = '--';
		% For the first 6, print the dividing value at the left just above the line.
		% After 6 it would get too crowded
		if k <= 6
			theLabel = sprintf('Dividing value = %.4f', thisDividingValue);
			text(xl(1)+0.1, thisDividingValue + 0.005, theLabel, 'Color', darkGreen);
		end
	end
	
catch ME
	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
		ME.stack(1).name, ME.stack(1).line, ME.message);
	fprintf(1, '%s\n', errorMessage);
	uiwait(warndlg(errorMessage));
end

% Now compute the number of vertices by looking at the number of peaks in a plot of distance from centroid.
function numVertices = FindNumberOfVertices(blobMeasurements, labeledImage)
try

    % load the normalized signatures from a file
    loaded_signatures = load('Shape_signatures.mat');
    loaded_signatures = loaded_signatures.Shape_signatures;

	numVertices = 0; % Initialize.
	% Get the number of blobs in the image.
	numRegions = length(blobMeasurements);
	hFig = figure;
	promptUser = true; % Let user see the curves.
	
	% For each blob, get its boundaries and find the distance from the centroid to each boundary point.
	for k = 1 : numRegions
		% Extract just this blob alone.
		thisBlob = ismember(labeledImage, k) > 0;

		% Find the boundaries
		thisBoundary = bwboundaries(thisBlob);
		thisBoundary = cell2mat(thisBoundary); % Convert from cell to double.
		% Get x and y
		x = thisBoundary(:, 2);
		y = thisBoundary(:, 1);
		% Get the centroid
		xCenter = blobMeasurements(k).Centroid(1);
		yCenter = blobMeasurements(k).Centroid(2);
        
        %Find the index of closest point to the upper right corner of the region
        [~,indx] = min(sqrt((y - size(thisBlob,2)).^2 + (x - size(thisBlob,1)).^2));
		% Compute distances

		distances = sqrt((x - xCenter).^2 + (y - yCenter).^2);
        distances = distances / max(distances);

        %resize to a vector of N=100 elements
        distances = imresize(distances, [100 1]);
        %distances = smoothdata(distances, 'movmean',2);

        D(4) = zeros;
        for c = 1 : 4
            for z = 1 : 100
                %D(z ,c) = distances(z) - loaded_signatures(z,c); 
                D(c) = D(c) + abs(distances(z) - loaded_signatures(z,c));
            end
        end

        if promptUser % Let user see the image and the curves.

			cla;
            subplot(1, 2, 1);
			imshow(thisBlob);
            hold on;
            plot(blobMeasurements(k).Centroid(1),blobMeasurements(k).Centroid(2),'bx');
            plot(x(indx),y(indx), 'ro');

			% Plot the distances.
            subplot(1, 2, 2);
			plot(distances, 'b-', 'LineWidth', 3);
			grid on;
			message = sprintf('Centroid to perimeter distances for shape #%d', k);
			title(message, 'FontSize', 12);
			% Scale y axis
			yl = ylim();
			ylim([0, yl(2)]); % Set lower limit to 0.
		end
		
		% Find the range of the peaks
		peakRange = max(distances) - min(distances);
		minPeakHeight = 0.1 * peakRange;
		% Find the peaks
		[peakValues, peakIndexes] = findpeaks(distances, 'MinPeakProminence', minPeakHeight);
		% Find the valleys.
		[valleyValues, valleyIndexes] = findpeaks(-distances, 'MinPeakProminence', minPeakHeight);
		numVertices(k) = max([length(peakValues), length(valleyValues)]);
		% Circles seem to have a ton of peaks due to the very small range and quanitization of the image.
		% If the number of peaks is more than 10, make it zero to indicate a circle.
		if numVertices(k) > 10
			numVertices(k) = 0;
		end
		
		if promptUser % Let user see the curves.
			% Plot the peaks.
			hold on;
			plot(peakIndexes, distances(peakIndexes), 'r^', 'MarkerSize', 5, 'LineWidth', 2);
			
			% Plot the valleys.
			hold on;
			plot(valleyIndexes, distances(valleyIndexes), 'rv', 'MarkerSize', 5, 'LineWidth', 2);
			
			message = sprintf('Centroid to perimeter distances for shape #%d.  Found %d peaks.', k, numVertices(k));
			title(message, 'FontSize', 10);
			
			% The figure un-maximizes each time when we call cla, so let's maximize it again.
			% Set up figure properties:
			% Enlarge figure to full screen.
			set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.5 1]);
			% Get rid of tool bar and pulldown menus that are along top of figure.
			%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
			% Give a name to the title bar.
			set(gcf, 'Name', 'Shape Signature', 'NumberTitle', 'Off')
			
			% Let user see this shape's distances plotted before continuing.
			promptMessage = sprintf('Do you want to Continue processing,\nor Cancel processing?');
			titleBarCaption = 'Continue?';
			button = questdlg(promptMessage, titleBarCaption, 'Continue', 'Cancel', 'Continue');
			if strcmpi(button, 'Cancel')
				promptUser = false;
			end
		end
	end
	close(hFig);
catch ME
	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
		ME.stack(1).name, ME.stack(1).line, ME.message);
	fprintf(1, '%s\n', errorMessage);
	uiwait(warndlg(errorMessage));
end




function [originalImage, binaryImage] = ImportImage()
try
    img = imread('./Images/img218.jpg');
    %img = imsharpen(img,'Radius',2,'Amount',0.8);
    
    % Get the size of the initial image in pixels
    [height, width, ~] = size(img);
    fprintf('\nOriginal Height: %d\t and Width: %d\t\n', height, width);


    % Preserve the aspect ratio and set the height to 400 pixels
    img = imresize(img, [400 NaN]);

    % Get the size of the image in pixels after resize
    [height, width, ~] = size(img);
    fprintf('\nResized Height: %d\t and Width: %d\t\n', height, width);

    originalImage = img;

    figure;
    subplot(3, 3, 1);
    imshow(img);
    title('Original Image');
    
    img_gray=rgb2gray(img);
    
    edging=edge(img_gray, 'log', 0.001);
    edgeFiltered = bwareaopen(edging, 40);
    edgeFilled = imfill(edgeFiltered,"holes");
    
    subplot(3, 3, 2);
    imshow(edgeFiltered);
    title('Edge Detection (cleared image)');
    
    subplot(3, 3, 3);
    imshow(edgeFilled);
    title('Edge Detection (with imfill)');
    
    red = createMaskRed(img);
    blue = createMaskBlue(img);
    % yellow = createMaskYellow(img);
    %imshow(yellow);
    
    %imshow( yellow | red | blue);
    
    BWimg = (red | blue);
    
    BWcleanedbuttom = bwareaopen(BWimg, 150); 
    BWcleanedtop = bwareaopen(BWimg, 60000);
    BWcleaned = imsubtract(BWcleanedbuttom, BWcleanedtop);
    BWFilled = imfill(BWcleaned,"holes");
    

    subplot(3, 3, 5);
    imshow(BWcleaned);
    title('Color Mask (cleared image)');
    
    subplot(3, 3, 6);
    imshow(BWFilled);
    title('Color Mask (with imfill)');
    
    % Union and Intersection of filled images
    union = (BWFilled | edgeFilled);
    intersection = (BWFilled & edgeFilled);
    intersectionFiltered = bwareaopen(double(intersection), 200);
    
    
    %union = imgaussfilt(union, 0.7);
    
    subplot(3, 3, 8);
    imshow(union);
    title('Union');
    
    subplot(3, 3, 9);
    imshow(intersection);
    title('Intersection');
    
    
    smoothed = imgaussfilt(double(intersectionFiltered),1.5);
    
    subplot(3, 3, 7);
    imshow(smoothed);
    title('Smoothed')

    binaryImage = smoothed;
    
    
    % bin = im2bw(img_gray);
    % figure;imshow(bin);
    
    [center,radii] = imfindcircles(smoothed,[30,150],'ObjectPolarity','bright','Sensitivity',0.85,'Method','PhaseCode');
    h = viscircles(center,radii);
    
    delete(h);
    viscircles(center,radii);
    
    %  s = regionprops(bin_fill,'centroid');
    %  centroids = cat(1,s.Centroid);
    %  imshow(bincleaned)
    %  hold on
    %  plot(centroids(:,1),centroids(:,2),'b*')
    %  hold off
    
    
    [B,L] = bwboundaries(smoothed,'noholes'); % computes the exterior boundaries of objects in img L


    subplot(3, 3, 4);
    imshow(label2rgb(L, @jet, [.5 .5 .5])); % display the boundaries in image
    title('Boundaries')
    
    hold on
    for k = 1:length(B)
        boundary = B{k}; 
        plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
    end
    
    % the roundness of an object is evaluated with metric=4*pi*area/perimeter^2
    stats = regionprops(L,'Area','Centroid');  %computes the areas (in pixels) and centroids of objects
    threshold=0.87;
    
    % a threshold is defined for roundness of objects
    for k = 1:length(B) 
        % for every boundary
        boundary=B{k};
        per=diff(boundary).^2; perimeter= sum(sqrt(sum(per,2))); %estimate of obj perimeter
        area=stats(k).Area; 
        % compute the area of the object
        metric= 4*pi*area/perimeter^2; % the metric is computed
        metric_str=sprintf('%2.2f',metric); % display the metric
    
        if metric > threshold
            centroid = stats(k).Centroid;
            plot(centroid(1),centroid(2),'ko'); 
        end
    
        text(boundary(1,2)-35,boundary(1,1)+13,metric_str,'Color', 'g',...
        'FontSize',14,'FontWeight','bold');    % label the objects with corresponding metric
    end
catch ME
	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
		ME.stack(1).name, ME.stack(1).line, ME.message);
	fprintf(1, '%s\n', errorMessage);
	uiwait(warndlg(errorMessage));
end


function [BW,maskedRGBImage] = createMaskRed(RGB)
    %createMask  Threshold RGB image using auto-generated code from colorThresholder app.
    %  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
    %  auto-generated code from the colorThresholder app. The colorspace and
    %  range for each channel of the colorspace were set within the app. The
    %  segmentation mask is returned in BW, and a composite of the mask and
    %  original RGB images is returned in maskedRGBImage.
    
    % Auto-generated by colorThresholder app on 02-Jan-2023
    %------------------------------------------------------
    
    
    % Convert RGB image to chosen color space
    I = rgb2hsv(RGB);
    
    % Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.836;
    channel1Max = 0.065;
    
    % Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.300;
    channel2Max = 1.000;
    
    % Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.200;
    channel3Max = 1.000;
    
    % Create mask based on chosen histogram thresholds
    sliderBW = ( (I(:,:,1) >= channel1Min) | (I(:,:,1) <= channel1Max) ) & ...
        (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
        (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
    BW = sliderBW;
    
    % Initialize output masked image based on input image.
    maskedRGBImage = RGB;
    
    % Set background pixels where BW is false to zero.
    maskedRGBImage(repmat(~BW,[1 1 3])) = 0;


function [BW,maskedRGBImage] = createMaskBlue(RGB)
    %createMask  Threshold RGB image using auto-generated code from colorThresholder app.
    %  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
    %  auto-generated code from the colorThresholder app. The colorspace and
    %  range for each channel of the colorspace were set within the app. The
    %  segmentation mask is returned in BW, and a composite of the mask and
    %  original RGB images is returned in maskedRGBImage.
    
    % Auto-generated by colorThresholder app on 02-Jan-2023
    %------------------------------------------------------
    
    
    % Convert RGB image to chosen color space
    I = rgb2hsv(RGB);
    
    % Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.543;
    channel1Max = 0.743;
    
    % Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.555;
    channel2Max = 1.000;
    
    % Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.350;
    channel3Max = 1.000;
    
    % Create mask based on chosen histogram thresholds
    sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
        (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
        (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
    BW = sliderBW;
    
    % Initialize output masked image based on input image.
    maskedRGBImage = RGB;
    
    % Set background pixels where BW is false to zero.
    maskedRGBImage(repmat(~BW,[1 1 3])) = 0;


