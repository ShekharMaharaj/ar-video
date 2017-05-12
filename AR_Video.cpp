/*
 * AR_Display_v1.cpp
 *
 *  Created on: 28-May-2015
 *      Author: rudren and shekhar
 */

#include <cv.h>
#include <highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <string>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/video/background_segm.hpp"

#define PI 3.14159265
#define MARKER_ROWS 6
#define MARKER_COLS 6
#define MARKER_CORNERS 4

using namespace cv;
using namespace std;

class ARMarker;
void runAR(VideoCapture& vid, vector<ARMarker> markers);
void findSquares(const Mat& image, vector<vector<Point> >& squares);
void drawSquare(Mat& image, const vector<Point>& squares);
void markerPerspectiveTransform(Mat& image, Mat& possibleMarker,
		const vector<Point>& squares);
void decodeMarker(Mat& image, vector<int>& code, int rows, int cols);
void orderPoints(vector<Point>& unordered, vector<Point>& ordered);
void rotateCode(vector<int>& code, vector<int>& rotated);
void markCorners(Mat& image, vector<Point> corners);

int thresh = 50;


/*Each marker is stored as an object, with its equivalent code
*and its associated video stream
*/
class ARMarker {
	VideoCapture *video_stream;
	Mat marker;
	vector<int> marker_code;
	int rows, cols;

	//incomplete
	//function meant to return the corners of the found marker in order,
	//i.e. the first corner should correspond to the top left corner
	//of the stored marker to allow for rotation  identification
	void rotateCorners(vector<Point>& corners, vector<Point>& rotated, int rotation) {
		for(int i = 0; i < corners.size(); i++){
			int index = (rotation + i) % corners.size();
			rotated.push_back(corners[index]);
		}
	}

public:
	ARMarker(Mat& image, VideoCapture& video) {
		rows = MARKER_ROWS;
		cols = MARKER_COLS;
		decodeMarker(image, marker_code, rows, cols);

		image.copyTo(marker);

		//just for debugging
		//prints the decoded marker code
		cout << "Marker code:" << endl;
		for (int i = 0; i < marker_code.size(); i++) {
			if (i % MARKER_COLS == 0) {
				cout << endl;
			}
			cout << marker_code[i] << ' ';
		}
		//just for debugging


		video_stream = &video;
	}

	//incomplete
	/*is called to get next frame from
	*associated video stream of marker
	*/
	void nextFrame(Mat& image) {
		if (!video_stream->read(image)) {
					cout << "End of video stream" << endl;
			}
	}

	/*accepts code of possible marker along with its corner points
	 * on the image, and tries to match the code to that of the
	 * marker, checking for rotation as well. In case it is found,
	 * it returns the corners in order of matching with the
	 * stored marker.
	 */
	int matchMarker(vector<int>& code, vector<Point> corners,
			vector<Point>& matched_corners) {
		int offset = MARKER_COLS;
		int rotation = 0;

		int flag = 0;
		for (int j = 0; j < code.size(); j++) {
			if (marker_code[j] != code[j]) {
				flag = 1;
				break;
			}
		}

		if (flag == 0) {
			rotation = 0;
			rotateCorners(corners, matched_corners, rotation);
			return 1;
		} else {
			flag = 0;
		}

		/*Each rotateCode call rotates marker code
		 * by 90 degrees in anti-clockwise direction
		 */
		vector<int> rotated_code_1;
		rotateCode(code, rotated_code_1);

		for (int j = 0; j < rotated_code_1.size(); j++) {
			if (marker_code[j] != rotated_code_1[j]) {
				flag = 1;
				break;
			}
		}

		/*rearranging corner points still left. currently only
		 * whether found or not
		 */

		if (flag == 0) {
			rotation = 1;
			rotateCorners(corners, matched_corners, rotation);
			return 1;
		} else {
			flag = 0;
		}

		vector<int> rotated_code_2;
		rotateCode(rotated_code_1, rotated_code_2);

		for (int j = 0; j < rotated_code_2.size(); j++) {
			if (marker_code[j] != rotated_code_2[j]) {
				flag = 1;
				break;
			}
		}

		if (flag == 0) {
			rotation = 2;
			rotateCorners(corners, matched_corners, rotation);
			return 1;
		} else {
			flag = 0;
		}

		vector<int> rotated_code_3;
		rotateCode(rotated_code_2, rotated_code_3);

		for (int j = 0; j < rotated_code_3.size(); j++) {
			if (marker_code[j] != rotated_code_3[j]) {
				flag = 1;
				break;
			}
		}

		if (flag == 0) {
			rotation = 3;
			rotateCorners(corners, matched_corners, rotation);
			return 1;
		} else {
			flag = 0;
		}

		return -1;
	}

	void getWarpMatrix(vector<Point> matched_corners, Mat& warp){
		Point2f orig[4];
		Point2f trans[4];

		orig[0].x = 0;
		orig[0].y = 0;
		orig[1].x = marker.cols;
		orig[1].y = 0;
		orig[2].x = marker.cols;
		orig[2].y = marker.rows;
		orig[3].x = 0;
		orig[3].y = marker.rows;

		for(int i = 0; i < matched_corners.size(); i++){
			trans[i].x = matched_corners[i].x;
			trans[i].y = matched_corners[i].y;
		}

		warp = getPerspectiveTransform(orig, trans);

        //trans[0].x = 0;
        //trans[0].y = 0;
        //trans[1].x = 200;
        //trans[1].y = 50;
        //trans[2].x = 200;
        //trans[2].y = 200;
        //trans[3].x = 25;
        //trans[3].y = 250;
	}

	void applyWarp(Mat& frame, Mat& image, Mat& warp_matrix, double scale){

		if(image.rows == 0){
			cout<<"Cannot read image"<<endl;
			return;
		}

		cout<<endl<<endl;
		cout<<"Marker size: "<<marker.cols<<" x "<<marker.rows<<endl;
		cout<<"Image size: "<<image.cols<<" x "<<image.rows<<endl;

		double ratio = ((double)marker.rows/(double)image.rows);

		cout<<"Ratio: "<<ratio<<endl;


		Mat resized_image;

		resize(image, resized_image, Size(round(ratio * scale * image.cols), round(ratio * scale * image.rows)), 0,0);



		Mat neg_img = Mat::zeros(frame.rows, frame.cols, frame.type());
		Mat copy_img = Mat::zeros(frame.rows, frame.cols, frame.type());
		Mat blank = Mat::zeros(resized_image.rows, resized_image.cols, frame.type());
		bitwise_not(blank, blank);

		warpPerspective(resized_image, neg_img, warp_matrix, neg_img.size());
		warpPerspective(blank, copy_img, warp_matrix, copy_img.size());

		bitwise_not(copy_img, copy_img);

		bitwise_and(copy_img, frame, copy_img);
		bitwise_or(copy_img, neg_img, frame);
	}
};


int main(int argc, char** argv) {

	/*A lot of work left here. Plan is to ask user for main
	 * input video stream path, followed by paths to individual marker images
	 * and their corresponding video streams. Each marker and video stream would
	 * be used to create an ARMarker object which is then stored in a vector.
	 *
	 * Currently a single marker, a dummy video stream and the main video input
	 * stream are  manually added.
	 */

	vector<ARMarker> markers;

	//Replace with your marker path
	String marker_path =
			"/Users/shekharmaharaj/Documents/ComputerVisionXcode/marker.png";

	Mat marker_img = imread(marker_path);

	VideoCapture dummystream;
	//Replace with your video file path
	dummystream.open("/Users/shekharmaharaj/Documents/ComputerVisionXcode/video.mp4");

	VideoCapture videoInput;


    //videoInput.open("http://192.168.1.8:8080/video?x.mjpeg");
	videoInput.open(0);

	ARMarker marker_1(marker_img, dummystream);
	markers.push_back(marker_1);

	runAR(videoInput, markers);

}


void runAR(VideoCapture& vid, vector<ARMarker> markers) {

	if (!vid.isOpened()) {
		cout << "Problem opening video stream" << endl;
		return;
	}

    
   
    
	Mat src, copy;

	char input = -1;

	namedWindow("Camera Feed", 1);
	namedWindow("Transform", 1);
    namedWindow("Foreground", 1);

	//repeat for every frame of input video stream
	while ((input = waitKey(10)) < 0) {
        
        //BACKGROUND SUBTRACTOR################################
        Point2f pts[4];
        Mat frameSize;
        //flag if marker found to perform background subtraction
        int foundMarker = -1;

		if (!vid.read(src)) {
					cout << "End of video stream" << endl;
					break;
		}

		src.copyTo(copy);

		//find every identifiable square in the frame
		vector<vector<Point> > squares;
		findSquares(copy, squares);

		/*For every square found, first get its code
		 * then check for a match against every marker stored.
		 * If a match is found, its respective video frame is to
		 * be overlayed. To not have frame skipping of the individual
		 * video feeds, marker video stream frames can be extracted
		 * and only overlayed if found
		 */

		vector<Mat> frames;

		for(int i = 0; i < markers.size(); i++){
			Mat temp;
			markers[i].nextFrame(temp);
			frames.push_back(temp);
		}

		for (int i = 0; i < squares.size(); i++) {

			Mat possible_marker;

			/*Transforms the identified skew marker square
			 * into a proper square image of 200 x 200
			 * to ease decoding
			 */
			markerPerspectiveTransform(copy, possible_marker, squares[i]);
			vector<int> marker_code;

			//decode now square possible marker
			decodeMarker(possible_marker, marker_code, MARKER_ROWS,
					MARKER_COLS);

			//for every marker, try to match possible marker found
			for (int j = 0; j < markers.size(); j++) {
				vector<Point> matched_corners;
				int found = markers[j].matchMarker(marker_code, squares[i], matched_corners);
                
                
                //Mat frame;
                //markers[j].nextFrame(frame);

				//currently only drawing lines around identified marker
				if(found != -1){
                    //drawSquare(copy, matched_corners);
					Mat warpTransform;
					markers[j].getWarpMatrix(matched_corners, warpTransform);

                    //Mat sample = imread("/home/rudren/Pictures/url.jpeg");
                    //Mat sample(400, 600, CV_8UC3, Scalar(255,0,0));


					if(frames[j].rows > 0){
					markers[j].applyWarp(copy, frames[j], warpTransform, 2);
					}
                    
                    //markCorners(copy, matched_corners);
                    
                    
                    
                    //BACkGROUND SUBTRACTOR###################################
                    
                    //marker to only do background subtraction when marker found
                    foundMarker = 1;
                    
                    
                    frames[j].copyTo(frameSize);
                    for(int i = 0; i < matched_corners.size(); i++){
                        pts[i].x = matched_corners[i].x;
                        pts[i].y = matched_corners[i].y;
                    }
                    
				}

			}
		}

        

        if(foundMarker == 1) {
            //0.1 learning rate gets ride of ghost image at start, still trying to find a better value to reduce noise
            
            //create a mask from the source image with a learning rate of 0.1
            pMOG2->operator()(src, fgMask, 0.1);
            
            //erode & dilate mask to reduce noise and increase coagulation
            int erosion_size = 1;
            //erosion kernel
            Mat erosion_element = getStructuringElement(MORPH_ELLIPSE, Size(2*erosion_size + 1, 2*erosion_size+1),
                                                        Point(erosion_size, erosion_size));
            erode(fgMask, fgMask, erosion_element);
            
            int dilation_size = 1;
            //dilation kernel
            Mat dilation_element = getStructuringElement(MORPH_ELLIPSE, Size(2*dilation_size + 1, 2*dilation_size+1),
                                                         Point(dilation_size, dilation_size));
            dilate(fgMask, fgMask, dilation_element);
            
            
            //get colour values from original image, white pixels from mask will be replaced with colour ones from original image
            Mat foreground = Mat::zeros(src.rows, src.cols, src.type());
            src.copyTo(foreground, fgMask);
            //add foreground images in front of augmented screen
            addWeighted(copy, 1.0, foreground, 1.0, 0.0, foreground);
            
            //flip before displaying for correct display orientation
            flip(foreground, foreground,1);
            flip(copy,copy,1);
            imshow("Foreground", foreground);
        } else {
            
            //if marker not found then just output the screen with no background subtraction
            flip(copy,copy,1);
            imshow("Foreground", copy);
        }
        
        
       
        
        
        
        
        flip(src,src,1);
        
		imshow("Camera Feed", src);
		imshow("Transform", copy);
        
	}
}


//Identifying all squares in the image frame
void findSquares(const Mat& image, vector<vector<Point> >& squares) {
	squares.clear();

	Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	// down-scale and upscale the image to filter out the noise
	pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
	pyrUp(pyr, timg, image.size());
	vector<vector<Point> > contours;

	cvtColor(timg, gray0, CV_BGR2GRAY);

	/*have to try adaptive threshold instead of otsu
	 * as it might be more versatile to localized
	 * brightness variation.
	 */
	threshold(gray0, gray0, 150, 255, THRESH_OTSU);
	adaptiveThreshold(gray0, gray0, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 0);


	Canny(gray0, gray, 0, thresh, 5);
	// dilate canny output to remove potential
	// holes between edge segments
	dilate(gray, gray, Mat(), Point(-1, -1));

	findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	vector<Point> approx;

	// test each contour
	for (size_t i = 0; i < contours.size(); i++) {
		// approximate contour with accuracy proportional
		// to the contour perimeter
		approxPolyDP(Mat(contours[i]), approx,
				arcLength(Mat(contours[i]), true) * 0.02, true);

		// square contours should have 4 vertices after approximation
		// relatively large area (to filter out noisy contours)
		// and be convex.
		// Note: absolute value of an area is used because
		// area may be positive or negative - in accordance with the
		// contour orientation
		if (approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000
				&& isContourConvex(Mat(approx))) {

			int repeat = 0;

			Rect current = boundingRect(approx);

			for (int j = 0; j < squares.size(); j++) {
				Rect saved = boundingRect(squares[j]);

				//checking for overlapping rectangles
				Rect diff = saved & current;

				if (diff.area() >= (0.5 * current.area())) {
					repeat = 1;
					break;
				}
			}


			if (repeat == 0) {
				vector<Point> ordered;
				/*ordering the 4 corners' labeling so
				 * the square is labeled starting first from top left corner
				 * in clockwise direction
				 */
				orderPoints(approx, ordered);
				squares.push_back(ordered);
			} else {
				repeat = 0;
			}
		}
	}
}

//the function draws the square in the image using the given images
void drawSquare(Mat& image, const vector<Point>& square) {



	const Point* p = &square[0];
	int n = (int) square.size();
	polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, CV_AA);

}

//to transform the skew image of the possible marker to 200 x 200 square image
void markerPerspectiveTransform(Mat& image, Mat& possibleMarker,
		const vector<Point>& square) {

	Mat marker(200, 200, image.type());

	Point2f input[4];
	Point2f output[4];

	for (int j = 0; j < 4; j++) {
		input[j] = Point2f(square[j].x, square[j].y);
	}

	output[0] = Point2f(0, 0);
	output[1] = Point2f(200, 0);
	output[2] = Point2f(200, 200);
	output[3] = Point2f(0, 200);

	Mat transform = getPerspectiveTransform(input, output);
	warpPerspective(image, marker, transform, marker.size());
	marker.copyTo(possibleMarker);
}

//To order points of found marker in clockwise direction staring from top left corner
void orderPoints(vector<Point>& unordered, vector<Point>& ordered) {
	/*This function basically finds a rough centre of all the marker's
	 * corner points in order to shift the co-ordinate system for easier calculations.
	 * Then, each point is converted to polar co-ordinates so we can sort by angle
	 * in clockwise direction.
	 */

	//finding rough centre
	double centre_x = 0, centre_y = 0;

	for (int i = 0; i < unordered.size(); i++) {
		centre_x += unordered[i].x;
		centre_y += unordered[i].y;
	}

	centre_x /= unordered.size();
	centre_y /= unordered.size();

	vector<double> polar_angles;

	//calculating angle part of polar equivalent for
	//each point
	for (int i = 0; i < unordered.size(); i++) {
		double temp_x = unordered[i].x - centre_x;
		double temp_y = centre_y - unordered[i].y;

		double r = sqrt((temp_x * temp_x) + (temp_y * temp_y));
		double fraction = abs(temp_y) / r;

		double angle = asin(fraction) * 180 / PI;
		if (temp_x >= 0) {
			if (temp_y >= 0) {
				angle = 180 - angle;
			} else {
				angle = 180 + angle;
			}
		} else {
			if (temp_y < 0) {
				angle = 360 - angle;
			}
		}
		polar_angles.push_back(angle);
	}

	//rearranging points in clockwise order from 180 to -180 degrees
	for (int i = 0; i < polar_angles.size(); i++) {
		double temp = INFINITY;
		int index = 0;
		for (int j = 0; j < polar_angles.size(); j++) {
			if (polar_angles[j] < temp) {
				index = j;
				temp = polar_angles[j];
			}
		}
		polar_angles[index] = INFINITY;
		ordered.push_back(unordered[index]);

	}
}

//Decodes possible marker from its image
void decodeMarker(Mat& image, vector<int>& code, int rows, int cols) {

	/*The input rectangular image is split into cells
	 * based on the number of rows and colums expected in
	 * the marker, also considering the marker border.
	 * The average of each cell is taken and thresholded
	 * to  0 or a 1 for black or white respectively.
	 */
	Mat gray;
	int cell_width = image.cols / (cols + 2);
	int cell_height = image.rows / (rows + 2);

	cvtColor(image, gray, CV_BGR2GRAY);

	threshold(gray, gray, 100, 255, THRESH_OTSU);

	for (int i = 1; i <= rows; i++) {
		for (int j = 1; j <= cols; j++) {
			Mat cell(gray,
					Rect(Point(cell_width * j, cell_height * i),
							Size(cell_width, cell_height)));
			Scalar avg = mean(cell);
			if (avg.val[0] > 128) {
				code.push_back(1);
			} else {
				code.push_back(0);
			}
		}
	}
}

//Rotating the marker code
void rotateCode(vector<int>& code, vector<int>& rotated) {
	/*Rotating the marker code instead of rotating the image
	 * and decoding again. Might save on processing
	 */

	for (int i = (MARKER_COLS - 1); i >= 0; i--) {
		for (int j = i; j < code.size(); j += MARKER_ROWS) {
			rotated.push_back(code[j]);
		}
	}
}

//just a debugging function to print corner labels in order
void markCorners(Mat& image, vector<Point> corners){
	char ch = 'A';
	for(int i = 0; i < corners.size(); i++){
		string text(1, ch);
		putText(image, text , corners[i], FONT_HERSHEY_SIMPLEX, 3, Scalar(0,0,255));
		ch++;
	}
}


