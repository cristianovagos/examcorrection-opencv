// Visual Studio ONLY - Allowing for pre-compiled header files

// This has to be the first #include

// Remove it, if you are not using Windows and Visual Studio

//#include "stdafx.h"


// The remaining #includes

// Basic OpenCV functionalities


#include <iostream>


#include "opencv2/core/core.hpp"

#include "opencv2/highgui/highgui.hpp"

#include "opencv2/imgproc/imgproc.hpp"

//#include "opencv2/text.hpp"

//GLOBAL VARIABLES
using namespace cv;
using namespace std;

#define X 1;
#define LETTER 2;
#define QUESTIONN 3;

struct Grid
{
	int width;
	int length;
	int type;
};

Mat templateX;
Mat image;
Mat finalImg;
int m;
Mat frame;
Point2f src[4];
char Option[] = {' ', 'A', 'B', 'C', 'D' };

void extractGrid();
bool detectExamManual(Mat exam);
void detectExamAuto(Mat exam);
int MatchingMethod(Mat rect);
void drawLine(Vec2f line, Mat &image, Scalar rgb);
void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center);
void onMouse(int event, int x, int y, int, void* param);

int main(int argc, char** argv) {
	templateX = imread("template/X.jpg", 0);

	// False
	bool useCamera = false;
	bool detectionAuto = true;
	
	if(useCamera) {
		// Open the default camera
		VideoCapture cap(0);

		// Check if the capture is working
		if (!cap.isOpened())
			return -1;
		
		// Get a frame from the camera
		cap >> frame;
	} else {
		frame = imread("exam7.jpg", 0);
	}
	
	// Check if we have a frame
	if(frame.empty())
	   return -1;
	
	// Detection Mode
	if(detectionAuto) {
		// Auto
		detectExamAuto(frame);
	} else {
		// Manual
		if (!detectExamManual(frame))
			return 0;
	}

	extractGrid();

	// Show the frame captured
	//imshow("capture", frame);
	//imshow("final grid", finalImg);
	//waitKey(0);
	destroyAllWindows();

	return 0;
}

//-------------------------------------------------------------------------//


//-------------------------------------------------------------------------//

void extractGrid()
{
	Mat gray;

	if (finalImg.channels() == 3)
	{
		cvtColor(finalImg, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = finalImg;
	}
	
	// Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
	Mat bw;
	adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	
	// Create the images that will use to extract the horizonta and vertical lines
	Mat horizontal = bw.clone();
	Mat vertical = bw.clone();

	int scale = 15; // play with this variable in order to increase/decrease the amount of lines to be detected

	// Specify size on horizontal axis
	int horizontalsize = horizontal.cols / scale;

	// Create structure element for extracting horizontal lines through morphology operations
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));

	// Apply morphology operations
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	//    dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1)); // expand horizontal lines

	// Show extracted horizontal lines
	imshow("horizontal", horizontal);

	// Specify size on vertical axis
	int verticalsize = vertical.rows / scale;

	// Create structure element for extracting vertical lines through morphology operations
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));

	// Apply morphology operations
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));

	//dilate(vertical, vertical, verticalStructure, Point(-1, -1)); // expand vertical lines

	vector<Vec4i> linesH;

	vector<vector<Point> > contoursH;

	Mat element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * 1 + 1, 2 * 1 + 1),
		Point(1, 1));
	//erode(joints, joints, element, Point(-1, -1));

	findContours(horizontal, contoursH, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/*sorting contours*/
	int i, j;
	bool swapped = false;
	int n = contoursH.size();
	for (i = 0; i < n - 1; i++)
	{
		swapped = false;
		for (j = 0; j < n - i - 1; j++)
		{
			if (contoursH[j][0].y > contoursH[j + 1][0].y)//if (contoursH[j] > contoursH[j + 1])
			{
				vector<Point> temp = contoursH[j];
				contoursH[j] = contoursH[j + 1];
				contoursH[j + 1] = temp;

				swapped = true;
			}
		}
		// IF no two elements were swapped by inner loop, then break
		if (swapped == false)
			break;
	}
	
	vector<vector<Point> > contoursV;
	findContours(vertical, contoursV, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	/*sorting contours*/
	n = contoursV.size();
	swapped=false;
	for (i = 0; i < n - 1; i++)
	{
		swapped = false;
		for (j = 0; j < n - i - 1; j++)
		{
			if (contoursV[j][0].x > contoursV[j+1][0].x)//if (contoursH[j] > contoursH[j + 1])
			{
				vector<Point> temp = contoursV[j];
				contoursV[j] = contoursV[j + 1];
				contoursV[j + 1] = temp;

				swapped = true;
			}
		}

		// IF no two elements were swapped by inner loop, then break
		if (swapped == false)
			break;
	}

	//------------Create Grid----------------------------------//
	
	int nLinhas = contoursV.size() - 1;
	int nColunas = contoursV.size() - 1;
	Mat crop;
	int nTables = nColunas / 5; //cada tabela tem 5 colunas: numero | A | B | C | D | 
	Mat temp;
	Rect recAnswer;
	int max = 0;
	finalImg.copyTo(temp);
	for (int t = 0; t < nTables; t++)//percorrer todas as tabelas
	{
		int tableN = t * 5;
		for (int l = 1; l < nLinhas; l++) //a primeira linha contem as opcoes de escolha, ignora-se
		{

			int index = 0;
			max = 0;

			int width = contoursV[tableN + 5][0].x - contoursV[tableN + 1][0].x;
			int height = contoursH[l + 1][0].y - contoursH[l][0].y;

			int sqaureWidth = width / 4;
			for (int c = 1; c < 5; c++)
			{
				Rect rec = Rect(contoursV[tableN + c][0].x, contoursH[l][0].y,
					sqaureWidth,
					height);
				crop = temp(rec);
				threshold(crop, crop, 125, 255, THRESH_BINARY_INV);
				double p = 0;

				p = (double)countNonZero(crop);
				p = p / (crop.size().width * crop.size().height);

				if ( p>=max && p>=0.3) {
					max = p;
					index = c;
					recAnswer = rec;
				}
				else {
					index = -1;
				}
				
			}
			//crop image
			printf("\nResposta %u: %c ", l + tableN, Option[index+1]);
			if (index != 0)
			{
				rectangle(finalImg, recAnswer, (255, 255, 255));
			}
			imshow("SquaresTest", finalImg);
		}
	}

	waitKey();
}

//-------------------------------------------------------------------------//

bool detectExamManual(Mat exam)
{
	
	imshow("Click 4 points - LT - RT - LB - RB", exam);
	m = 0;
	setMouseCallback("Click 4 points - LT - RT - LB - RB", onMouse);
	int width;
	int height;
	waitKey();

	if (m >= 4)
	{
		width = (src[1].x - src[0].x >= src[3].x - src[2].x) ? src[1].x - src[0].x : src[3].x - src[2].x;
		height = (src[2].y - src[0].y >= src[3].y - src[1].y) ? src[2].y - src[0].y : src[3].y - src[1].y;

		Point2f dst[4];
		dst[0].x = 0; dst[0].y = 0;
		dst[1].x = width; dst[1].y = 0;
		dst[2].x = 0; dst[2].y = height;
		dst[3].x = width; dst[3].y = height;

 		Mat result, perspective;
		frame.copyTo(result);
		perspective = getPerspectiveTransform(src, dst);
		warpPerspective(exam, result, perspective, Size(width, height));

		
		imshow("Perspective", result);
		waitKey();
		result.copyTo(finalImg);
		return true;
	}
	return false;
	
}


//----------------------------------------------------------------------------//

int MatchingMethod( Mat rect) {
	/// Source image to display
	Mat img_display;

	Mat img;
	Mat result;
	rect.copyTo(img);
	
	/// Prepare the image for perspective
		int width, height;
		width = templateX.cols * 4 + 1;
		height = templateX.rows + 1;
		Point2f src[4];
		src[0].x = 0; src[0].y = 0;
		src[1].x = img.cols; src[1].y = 0;
		src[2].x = 0; src[2].y = img.rows;
		src[3].x = img.cols; src[3].y = img.rows;

		Point2f dst2[4];
		dst2[0].x = 0; dst2[0].y = 0;
		dst2[1].x = width; dst2[1].y = 0;
		dst2[2].x = 0; dst2[2].y = height;
		dst2[3].x = width; dst2[3].y = height;

		Mat resultP, perspective;
		frame.copyTo(result);
		perspective = getPerspectiveTransform(src, dst2);
		warpPerspective(img, resultP, perspective, Size(width, height));
		imshow("Perspective", resultP);
		imshow("original", rect);
		waitKey();

		int result_cols = resultP.cols - templateX.cols + 1;
		int result_rows = resultP.rows - templateX.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching and Normalize
	int match_method = CV_TM_SQDIFF_NORMED;
	matchTemplate(resultP, templateX, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// Show me what you got
	rectangle(resultP, matchLoc, Point(matchLoc.x + templateX.cols, matchLoc.y + templateX.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templateX.cols, matchLoc.y + templateX.rows), Scalar::all(0), 2, 8, 0);

	imshow("X", resultP);
	imshow("result", result);

	return matchLoc.x;
}

//-------------------------------------------------------------------------//

void detectExamAuto(Mat exam) {
	// Creating empty grid with same size of the "exam"
	Mat grid = Mat(exam.size(), CV_8UC1);

	// Remove some noise with Gaussian Blur (5x5), smoothing the image
	GaussianBlur(exam, exam, Size(5, 5), 0);

	// Threshold the image according to local neighborhood
	// Normal threshold takes the image as a whole instead, that's why we don't use it
	adaptiveThreshold(exam, grid, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);

	// Invert grid image, setting the contours white
	bitwise_not(grid, grid);

	// Creating cross structuring element (3x3)
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));

	// Dilate image with the cross structuring element to increase grid borders 
	dilate(grid, grid, kernel);

	//imshow("exam", exam);
	//imshow("exam gray", grid);
	//waitKey(0);
	//destroyAllWindows();

	// Now we will find the maximum filled area possible (also called blob) 
	// in the captured image (that should be the answer grid)
	int count = 0;
	int max = -1;
	Point maxPoint;

	// Iterate through all grid
	for (int y = 0; y < grid.size().height; y++) {
		// Get current line
		uchar *row = grid.ptr(y);

		for (int x = 0; x < grid.size().width; x++) {
			// We only want the "whiter" areas (the contours)! (>=128)
			if (row[x] >= 128) {
				// Cover the found area darker (gray color) for better distinguish from others
				int area = floodFill(grid, Point(x, y), CV_RGB(0, 0, 64));
				if (area > max) {
					// Update the maximum area found
					maxPoint = Point(x, y);
					max = area;
				}
			}
		}
	}

	// We fill the max blob found with white color
	floodFill(grid, maxPoint, CV_RGB(255, 255, 255));

	// Iterate through all grid
	for (int y = 0; y < grid.size().height; y++) {
		// Get current line
		uchar *row = grid.ptr(y);

		for (int x = 0; x < grid.size().width; x++) {
			// If the row is equal to 64 (the gray color covered before) or it's the
			// maximum point we don't want it there, so we fill it with black
			if (row[x] == 64 && x != maxPoint.x && y != maxPoint.y) {
				int area = floodFill(grid, Point(x, y), CV_RGB(0, 0, 0));
			}
		}
	}

	// Erode back the grid with the cross structuring element
	erode(grid, grid, kernel);

	// Now that we have the area we want, we need to extract only the grid
	// The grid has horizontal and vertical lines, so we will need to find those lines

	// Find lines in the grid using Hough Transform
	vector<Vec2f> lines;
	HoughLines(grid, lines, 1, CV_PI / 180, 270);

	// Draw the lines obtained from the Hough transform in the grid
	for (int i = 0; i < lines.size(); i++) {
		drawLine(lines[i], grid, CV_RGB(0, 0, 128));
	}

	//imshow("exam", exam);
	//imshow("exam gray lines", grid);
	//waitKey(0);
	//destroyAllWindows();


	// We now have lots of lines so we need to minimize the lines obtained
	// so we can extract only the grid. So we will have only the boundary lines

	// Initialize with dummy values (x, y) to compare next
	Vec2f top = Vec2f(1000, 1000);
	double topYInterception = 100000, topXInterception = 0;

	Vec2f bottom = Vec2f(-1000, -1000);
	double bottomYInterception = 0, bottomXInterception = 0;

	Vec2f left = Vec2f(1000, 1000);
	double leftYInterception = 0, leftXInterception = 100000;

	Vec2f right = Vec2f(-1000, -1000);
	double rightYInterception = 0, rightXInterception = 0;

	// Iterate through all lines
	for (int i = 0; i < lines.size(); i++) {
		// Get current line
		Vec2f currentLine = lines[i];

		// Get the values (point, angle)
		float p = currentLine[0];
		float theta = currentLine[1];

		// Check for impossible values on lines
		if (p == 0 && theta == -100)
			continue;

		double xInterception, yInterception;
		xInterception = p / cos(theta);
		yInterception = p / (cos(theta)*sin(theta));

		// Check if line is almost vertical (between 80 to 100 degrees angle)
		if (theta > CV_PI * 80 / 180 && theta < CV_PI * 100 / 180) {
			if (p < top[0])
				top = currentLine;
			if (p > bottom[0])
				bottom = currentLine;
		}
		// Check if line is almost horizontal (around 170 or almost no degrees angle)
		else if (theta < CV_PI * 10 / 180 || theta > CV_PI * 170 / 180) {
			if (xInterception > rightXInterception) {
				right = currentLine;
				rightXInterception = xInterception;
			}
			else if (xInterception <= leftXInterception) {
				left = currentLine;
				leftXInterception = xInterception;
			}
		}
	}

	// Draw black lines with the final lines detected
	drawLine(top, exam, CV_RGB(0, 0, 0));
	drawLine(bottom, exam, CV_RGB(0, 0, 0));
	drawLine(left, exam, CV_RGB(0, 0, 0));
	drawLine(right, exam, CV_RGB(0, 0, 0));

	//imshow("exam", exam);
	//waitKey(0);
	//destroyAllWindows();

	// Calculate intersections of the four lines obtained
	// as seen in https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
	// and https://stackoverflow.com/questions/16524096/how-to-calculate-the-point-of-intersection-between-two-lines

	Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;
	Point topLeft, topRight, bottomLeft, bottomRight;

	// Top intersections
	top1.x = 0;
	top1.y = top[0] / sin(top[1]);
	top2.x = grid.size().width;
	top2.y = -top2.x / tan(top[1]) + top1.y;

	// Bottom intersections
	bottom1.x = 0;
	bottom1.y = bottom[0] / sin(bottom[1]);
	bottom2.x = grid.size().width;
	bottom2.y = -bottom2.x / tan(bottom[1]) + bottom1.y;

	// Left intersections
	// (they are vertical so we have to check for infinite angle)
	if (left[1] != 0)
	{
		// Calculate intersection points
		left1.x = 0;
		left1.y = left[0] / sin(left[1]);
		left2.x = grid.size().width;
		left2.y = -left2.x / tan(left[1]) + left1.y;
	}
	else
	{
		// Infinite angle! Other way of calculate
		left1.y = 0;
		left1.x = left[0] / cos(left[1]);
		left2.y = grid.size().height;
		left2.x = left1.x - grid.size().height*tan(left[1]);

	}

	// Right intersections
	// (they are vertical so we have to check for infinite angle)
	if (right[1] != 0)
	{
		// Calculate intersection points
		right1.x = 0;
		right1.y = right[0] / sin(right[1]);
		right2.x = grid.size().width;
		right2.y = -right2.x / tan(right[1]) + right1.y;
	}
	else
	{
		// Infinite angle! Other way of calculate
		right1.y = 0;
		right1.x = right[0] / cos(right[1]);
		right2.y = grid.size().height;
		right2.x = right1.x - grid.size().height*tan(right[1]);

	}

	// Get top points
	double topA = top2.y - top1.y;
	double topB = top1.x - top2.x;
	double topC = topA*top1.x + topB*top1.y;

	// Get bottom points
	double bottomA = bottom2.y - bottom1.y;
	double bottomB = bottom1.x - bottom2.x;
	double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;

	// Get left points
	double leftA = left2.y - left1.y;
	double leftB = left1.x - left2.x;
	double leftC = leftA*left1.x + leftB*left1.y;

	// Get right points
	double rightA = right2.y - right1.y;
	double rightB = right1.x - right2.x;
	double rightC = rightA*right1.x + rightB*right1.y;

	// Intersection of left and top
	double detTopLeft = leftA*topB - leftB*topA;
	topLeft = Point((topB*leftC - leftB*topC) / detTopLeft, (leftA*topC - topA*leftC) / detTopLeft);

	// Intersection of top and right
	double detTopRight = rightA*topB - rightB*topA;
	topRight = Point((topB*rightC - rightB*topC) / detTopRight, (rightA*topC - topA*rightC) / detTopRight);

	// Intersection of right and bottom
	double detBottomRight = rightA*bottomB - rightB*bottomA;
	bottomRight = Point((bottomB*rightC - rightB*bottomC) / detBottomRight, (rightA*bottomC - bottomA*rightC) / detBottomRight);// Intersection of bottom and left

																																// Intersection of left and bottom
	double detBottomLeft = leftA*bottomB - leftB*bottomA;
	bottomLeft = Point((bottomB*leftC - leftB*bottomC) / detBottomLeft, (leftA*bottomC - bottomA*leftC) / detBottomLeft);


	// Find the max length of the final grid (for building the final grid we need the size)
	// As it's a rectangle, we need to find the max length on the vertical/horizontal lines

	// Initially we say that the biggest length on the horizontal is the bottom line, and the vertical is the right line
	int maxLengthHori = (bottomLeft.x - bottomRight.x)*(bottomLeft.x - bottomRight.x) + (bottomLeft.y - bottomRight.y)*(bottomLeft.y - bottomRight.y);
	int maxLengthVert = (topRight.x - bottomRight.x)*(topRight.x - bottomRight.x) + (topRight.y - bottomRight.y)*(topRight.y - bottomRight.y);

	// Check with the other lines if they are the biggest
	int temp = (topRight.x - topLeft.x)*(topRight.x - topLeft.x) + (topRight.y - topLeft.y)*(topRight.y - topLeft.y);

	if (temp > maxLengthHori)
		maxLengthHori = temp;

	temp = (bottomLeft.x - topLeft.x)*(bottomLeft.x - topLeft.x) + (bottomLeft.y - topLeft.y)*(bottomLeft.y - topLeft.y);

	if (temp > maxLengthVert)
		maxLengthVert = temp;

	maxLengthVert = sqrt((double)maxLengthVert);
	maxLengthHori = sqrt((double)maxLengthHori);

	// Assign coordinates to the final grid
	Point2f src[4], dst[4];

	// Top Left (0, 0)
	src[0] = topLeft;
	dst[0] = Point2f(0, 0);

	// Top Right (maxLengthHori-1, 0)
	src[1] = topRight;
	dst[1] = Point2f(maxLengthHori - 1, 0);

	// Bottom Right (maxLengthHori-1, 0)
	src[2] = bottomRight;
	dst[2] = Point2f(maxLengthHori - 1, maxLengthVert - 1);

	// Bottom Left (0, maxLengthVert-1)
	src[3] = bottomLeft;
	dst[3] = Point2f(0, maxLengthVert - 1);

	// Build the final grid, with the coordinates assigned, correcting the perspective
	Mat finalGrid = Mat(Size(maxLengthHori, maxLengthVert), CV_8UC1);
	cv::warpPerspective(exam, finalGrid, cv::getPerspectiveTransform(src, dst), Size(maxLengthHori, maxLengthVert));

	finalGrid.copyTo(finalImg);
	//imshow("image", exam);
	//imshow("final grid", finalGrid);
	//waitKey(0);
	//destroyAllWindows();
}

//-------------------------------------------------------------------------//

void drawLine(Vec2f line, Mat &image, Scalar rgb) {
	if (line[1] != 0) {
		float m = -1 / tan(line[1]);
		float c = line[0] / sin(line[1]);

		cv::line(image, Point(0, c), Point(image.size().width, m*image.size().width + c), rgb);
	}
	else {
		cv::line(image, Point(line[0], 0), Point(line[0], image.size().height), rgb);
	}
}

//-------------------------------------------------------------------------//

void onMouse(int event, int x, int y, int, void* param) {
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		//Point2f * mp = (Point2f*)param;
		src[m].x = x;
		src[m].y = y;
		m++;

		printf("\n %d click on %u, %u ", m, x, y);
	}
}

