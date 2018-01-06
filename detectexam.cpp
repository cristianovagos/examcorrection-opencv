/* 
    Exam Correction - CV 2017/18

    Cristiano Vagos - 65169
    Miguel Brás - 49977
*/


// Visual Studio ONLY - Allowing for pre-compiled header files
// This has to be the first #include
// Remove it, if you are not using Windows and Visual Studio

//#include "stdafx.h"

#include <iostream>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

void detectExam(Mat exam);
void drawLine(Vec2f line, Mat &image, Scalar rgb);

int main(int argc, char** argv) {
    // Open the default camera
    //VideoCapture cap(0);
	
    //namedWindow("Exam Correction - CV 2017/18", CV_WINDOW_KEEPRATIO);

    // Check if the capture is working
    //if (!cap.isOpened())
    //	return -1;
	//while(true) {
        // Get a frame from the camera
		Mat frame;
		//cap >> frame; // get a new frame from camera

        frame = imread("exam.jpg", 0);

        // Check if we have a frame
        //if(frame.empty())
        //    break;

        // Detect exam
        detectExam(frame);

        // Show the frame captured
        imshow("capture", frame);
        //if(waitKey(30) >= 0) break;
    //}
    return 0;
}

void detectExam(Mat exam)
{
    // Creating empty grid with same size of the "exam"
    Mat grid = Mat(exam.size(), CV_8UC1);

    // Remove some noise with Gaussian Blur (5x5), smoothing the image
    GaussianBlur(exam, exam, Size(5,5), 0);

    // Threshold the image according to local neighborhood
    // Normal threshold takes the image as a whole instead, that's why we don't use it
    adaptiveThreshold(exam, grid, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);

    // Invert grid image, setting the contours white
    bitwise_not(grid, grid);

    // Creating cross structuring element (3x3)
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3,3));
    
    // Dilate image with the cross structuring element to increase grid borders 
    dilate(grid, grid, kernel);

    imshow("exam", exam);
    imshow("exam gray", grid);
    waitKey(0);
    destroyAllWindows();

    // Now we will find the maximum filled area possible (also called blob) 
    // in the captured image (that should be the answer grid)
    int count = 0;
    int max = -1;
    Point maxPoint;

    // Iterate through all grid
    for(int y = 0; y < grid.size().height; y++) {
        // Get current line
        uchar *row = grid.ptr(y);

        for(int x = 0; x < grid.size().width; x++) {
            // We only want the "whiter" areas (the contours)! (>=128)
            if(row[x] >= 128) {
                // Cover the found area darker (gray color) for better distinguish from others
                int area = floodFill(grid, Point(x, y), CV_RGB(0,0,64));
                if(area > max) {
                    // Update the maximum area found
                    maxPoint = Point(x, y);
                    max = area;
                }
            }
        }
    }

    // We fill the max blob found with white color
    floodFill(grid, maxPoint, CV_RGB(255,255,255));

    // Iterate through all grid
    for(int y = 0; y < grid.size().height; y++) {
        // Get current line
        uchar *row = grid.ptr(y);
        
        for(int x = 0; x < grid.size().width; x++) {
            // If the row is equal to 64 (the gray color covered before) or it's the
            // maximum point we don't want it there, so we fill it with black
            if(row[x]==64 && x!=maxPoint.x && y!=maxPoint.y) {
                int area = floodFill(grid, Point(x,y), CV_RGB(0,0,0));
            }
        }
    }
    
    // Erode back the grid with the cross structuring element
    erode(grid, grid, kernel);

    // Now that we have the area we want, we need to extract only the grid
    // The grid has horizontal and vertical lines, so we will need to find those lines

    // Find lines in the grid using Hough Transform
    vector<Vec2f> lines;
    HoughLines(grid, lines, 1, CV_PI/180, 270);

    // Draw the lines obtained from the Hough transform in the grid
    for(int i = 0; i < lines.size(); i++) {
        drawLine(lines[i], grid, CV_RGB(0,0,128));
    }

    imshow("exam", exam);
    imshow("exam gray lines", grid);
    waitKey(0);
    destroyAllWindows();


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
    for(int i = 0; i < lines.size(); i++) {
        // Get current line
        Vec2f currentLine = lines[i];
        
        // Get the values (point, angle)
        float p = currentLine[0];
        float theta = currentLine[1];

        // Check for impossible values on lines
        if(p == 0 && theta == -100)
            continue;

        double xInterception, yInterception;
        xInterception = p/cos(theta);
        yInterception = p/(cos(theta)*sin(theta));

        // Check if line is almost vertical (between 80 to 100 degrees angle)
        if(theta > CV_PI*80/180 && theta < CV_PI*100/180) {
            if(p < top[0])
                top = currentLine;
            if(p > bottom[0])
                bottom = currentLine;
        }
        // Check if line is almost horizontal (around 170 or almost no degrees angle)
        else if(theta < CV_PI*10/180 || theta > CV_PI*170/180) {
            if(xInterception > rightXInterception) {
                right = currentLine;
                rightXInterception = xInterception;
            }
            else if(xInterception <= leftXInterception) {
                left = currentLine;
                leftXInterception = xInterception;
            }
        }
    }

    // Draw black lines with the final lines detected
    drawLine(top, exam, CV_RGB(0,0,0));
    drawLine(bottom, exam, CV_RGB(0,0,0));
    drawLine(left, exam, CV_RGB(0,0,0));
    drawLine(right, exam, CV_RGB(0,0,0));

    imshow("exam", exam);
    waitKey(0);
    destroyAllWindows();

    // Calculate intersections of the four lines obtained
    Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;
    Point topLeft, topRight, bottomLeft, bottomRight;

    // Top intersections
    top1.x = 0;
    top1.y = top[0]/sin(top[1]);
    top2.x = grid.size().width;
    top2.y = -top2.x/tan(top[1]) + top1.y;

    // Bottom intersections
    bottom1.x = 0;
    bottom1.y = bottom[0]/sin(bottom[1]);
    bottom2.x = grid.size().width;
    bottom2.y = -bottom2.x/tan(bottom[1]) + bottom1.y;

    // Left intersections
    left1.x = 0;
    left1.y = left[0]/sin(left[1]);
    left2.x = grid.size().width;
    left2.y = -left2.x/tan(left[1]) + left1.y;
    
    // Right intersections
    right1.x = 0;
    right1.y = right[0]/sin(right[1]);
    right2.x = grid.size().width;
    right2.y = -right2.x/tan(right[1]) + right1.y;
    
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
    topLeft = Point((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft);

    // Intersection of top and right
    double detTopRight = rightA*topB - rightB*topA;
    topRight = Point((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight);

    // Intersection of right and bottom
    double detBottomRight = rightA*bottomB - rightB*bottomA;
    bottomRight = Point((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight);// Intersection of bottom and left
    
    // Intersection of left and bottom
    double detBottomLeft = leftA*bottomB - leftB*bottomA;
    bottomLeft = Point((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft);

    
    // Find the max length of the final grid (for building the final grid we need the size)
    // As it's a rectangle, we need to find the max length on the vertical/horizontal lines

    // Initially we say that the biggest length on the horizontal is the bottom line, and the vertical is the right line
    int maxLengthHori = (bottomLeft.x-bottomRight.x)*(bottomLeft.x-bottomRight.x) + (bottomLeft.y-bottomRight.y)*(bottomLeft.y-bottomRight.y);
    int maxLengthVert = (topRight.x-bottomRight.x)*(topRight.x-bottomRight.x) + (topRight.y-bottomRight.y)*(topRight.y-bottomRight.y);
    
    // Check with the other lines if they are the biggest
    int temp = (topRight.x-topLeft.x)*(topRight.x-topLeft.x) + (topRight.y-topLeft.y)*(topRight.y-topLeft.y);

    if(temp > maxLengthHori) 
        maxLengthHori = temp;

    temp = (bottomLeft.x-topLeft.x)*(bottomLeft.x-topLeft.x) + (bottomLeft.y-topLeft.y)*(bottomLeft.y-topLeft.y);

    if(temp > maxLengthVert)
        maxLengthVert = temp;

    maxLengthVert = sqrt((double)maxLengthVert);
    maxLengthHori = sqrt((double)maxLengthHori);

    // Assign coordinates to the final grid
    Point2f src[4], dst[4];

    // Top Left (0, 0)
    src[0] = topLeft;
    dst[0] = Point2f(0,0);
    
    // Top Right (maxLengthHori-1, 0)
    src[1] = topRight;
    dst[1] = Point2f(maxLengthHori-1, 0);
    
    // Bottom Right (maxLengthHori-1, 0)
    src[2] = bottomRight;
    dst[2] = Point2f(maxLengthHori-1, maxLengthVert-1);
    
    // Bottom Left (0, maxLengthVert-1)
    src[3] = bottomLeft;
    dst[3] = Point2f(0, maxLengthVert-1);

    // Build the final grid, with the coordinates assigned, correcting the perspective
    Mat finalGrid = Mat(Size(maxLengthHori, maxLengthVert), CV_8UC1);
    cv::warpPerspective(exam, finalGrid, cv::getPerspectiveTransform(src, dst), Size(maxLengthHori, maxLengthVert));

    imshow("image", exam);
    imshow("final grid", finalGrid);
    waitKey(0);
    destroyAllWindows();
}

void drawLine(Vec2f line, Mat &image, Scalar rgb) {
    if(line[1]!=0) {
        float m = -1/tan(line[1]);
        float c = line[0]/sin(line[1]);

        cv::line(image, Point(0, c), Point(image.size().width, m*image.size().width+c), rgb);
    }
    else {
        cv::line(image, Point(line[0], 0), Point(line[0], image.size().height), rgb);
    }
}
