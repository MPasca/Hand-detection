// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "math.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include <queue>
std::queue<Point2i> Q;

// checks to see if a set of coordinates is inside the image matrix
bool isInside(Mat img, int x, int y) {
	int width = img.rows;
	int height = img.cols;

	// if the x is bounded
	bool isInside = 0 < x && x < height ? true : false;

	// if not, immediately return false
	if (!isInside) {
		return false;
	}

	// if it's bounded, return whether y is also bounded
	return 0 < y && y < width ? true : false;
}

// basic function to test whether the program works properly
void testOpenImage()
{
	// opens an image file after the selected file path
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		// reads the image matrix from the .bmp file
		Mat src;
		src = imread(fname);

		// shows the colour matrix
		imshow("image", src);

		// waits for the X button to be pressed before reopening the file selection window
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		// reads the image matrix as a matrix of gray tones (uchar)
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		// init the destination matrix
		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				// val takes the value of the pixel located at the coords i, j in the img matrix
				uchar val = src.at<uchar>(i, j);

				// MAX_PATH = 260 => neg image takes the complementary values of the values found in the matrix
				uchar neg = MAX_PATH - val;
				// add the new value to the destination matrix
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		// shows the original image and the negative image side by side
		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}


void addImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				//uchar neg;
				//if (val + 50 > 255) {
				//	neg = 255;
				//}
				//else {
				//	neg = val + 50;
				//}
				
				dst.at<uchar>(i, j) = val + 50 > 255 ? 255 : val + 50;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void subImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				//uchar neg;
				//if (val + 50 > 255) {
				//	neg = 255;
				//}
				//else {
				//	neg = val + 50;
				//}

				dst.at<uchar>(i, j) = val - 100 < 0 ? 0 : val - 100;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}



void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

Mat testColor2Gray(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1);

	for (int i=0; i<height; i++)
	{
		for (int j=0; j<width; j++)
		{
			Vec3b v3 = src.at<Vec3b>(i,j);
			uchar b = v3[0];
			uchar g = v3[1];
			uchar r = v3[2];
			dst.at<uchar>(i,j) = (r+g+b)/3;
		}
	}

	return dst;
}

void testGray2BlackWhite()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar crtValue = src.at<uchar>(i, j);
				
				dst.at<uchar>(i, j) = crtValue >= 120 ? 255 : 0;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}


void RGBGray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat dstRed = Mat(height, width, CV_8UC3);
		Mat dstGreen = Mat(height, width, CV_8UC3);
		Mat dstBlue = Mat(height, width, CV_8UC3);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dstRed.at<Vec3b>(i, j) = Vec3b(0, 0, (r + g + b) / 3);
				dstGreen.at<Vec3b>(i, j) = Vec3b(0, (r + g + b) / 3, 0);
				dstBlue.at<Vec3b>(i, j) = Vec3b((r + g + b) / 3, 0, 0);
			}
		}

		imshow("input image", src);
		imshow("red image", dstRed);
		imshow("green image", dstGreen);
		imshow("blue image", dstBlue);

		waitKey();
	}
}


void testColorToGrayScale()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = 0.21 * r + 0.71 * g + 0.072 * b;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testColor2Neg()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC3);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<Vec3b>(i, j) = Vec3b(r, g, b);
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testColor2Color()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC3);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<Vec3b>(i, j) = Vec3b(b, g, r);
			}
		}

		for (int i = 0; i < 100; i++)
		{
			for (int j = 0; j < 100; j++)
			{
				dst.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			}
		}

		imshow("input image", src);
		imshow("green square", dst);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<Vec3b>(i, j) = Vec3b(b, g, r);
			}
		}

		for (int i = height / 2 - 50; i < height / 2 + 50; i++)
		{
			for (int j = width / 2 - 50; j < width / 2 + 50; j++)
			{
				dst.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
			}
		}

		imshow("yellow square", dst);

		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele de culoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		Mat HSV = Mat(height, width, CV_8UC3);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255

				HSV.at<Vec3b>(i, j) = Vec3b(lpH[gi], lpS[gi], lpV[gi]);
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);
		imshow("HSV", HSV);
		waitKey();
	}
}

void testRGB2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b rgb = src.at<Vec3b>(i, j);
				float r = (float)rgb[2]/255;
				float g = (float)rgb[1]/255;
				float b = (float)rgb[0]/255;

				float M = max(r, g, b);
				float m = min(r, g, b);

				float C = M - m;

				V.at<uchar>(i, j) = M;

				if (V.at<uchar>(i, j) != 0) {
					S.at<uchar>(i, j) = C / V.at<uchar>(i, j);
				}
				else {
					S.at<uchar>(i, j) = 0;
				}

				if (C != 0) {
					if (M == r) {
						H.at<uchar>(i, j) = 60 * (g - b) / C;
					}
					else if (M == g) {
						H.at<uchar>(i, j) = 120 + 60 * (b - r) / C;
					}
					else {
						H.at<uchar>(i, j) = 240 + 60 * (r - g) / C;
					}
				}
				else {
					H.at<uchar>(i, j) = 0;
				}

				if (H.at<uchar>(i, j) < 0) {
					H.at<uchar>(i, j) = H.at<uchar>(i, j) + 360;
				}

				H.at<uchar>(i, j) = H.at<uchar>(i, j) * 255 / 360;
				S.at<uchar>(i, j) = S.at<uchar>(i, j) * 255;
				V.at<uchar>(i, j) = V.at<uchar>(i, j) * 255;
			}
		}



		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);
		waitKey();
	}
}


void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
int M;
int* prag = (int *)malloc(256 * sizeof(int));
int maxCnt = 0;
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++) {
		if (hist[i] > max_hist)
			max_hist = hist[i];
	}

	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}


	imshow(name, imgHist);
}

Mat calcHistogram(Mat src) {	
	int* hist = (int*)calloc(256, sizeof(int));

	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int crtValue = src.at<uchar>(i, j);
			hist[crtValue]++;
		}
	}

	float* FDP = (float*)calloc(256, sizeof(float));
	M = height * width;

	for (int i = 0; i < 256; i++) {
		FDP[i] = (float)hist[i] / M;
		//printf("%.2f ", FDP[i]);
	}

	int WH = 5;
	int threshold = 0.0003;

	prag[maxCnt++] = 0;
	for (int k = 0 + WH; k < 255 - WH; k++) {
		float med = 0;

		bool isKMax = true;
		for (int c = k - WH; c < k + WH; c++) {
			med += FDP[c];
			if (k != c && FDP[c] > FDP[k]) {
				isKMax = false;
			}
		}
		med = med / (2 * WH + 1);

		if (FDP[k] > med + threshold && isKMax) {
			prag[maxCnt++] = k;
		}
	}

	prag[maxCnt++] = 255;

	Mat dst = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int crtValue = src.at<uchar>(i, j);
			int newVal = crtValue;
			int absVal = 255;
			for (int k = 0; k < maxCnt; k++) {
				if (abs(crtValue - prag[k]) < absVal) {
					newVal = prag[k];
					absVal = abs(crtValue - prag[k]);
				}
			}
			dst.at<uchar>(i, j) = newVal;
		}
	}

	return dst;
}

bool isObj(Vec3b pixel, Vec3b givenColour) {
	return pixel == givenColour;
}

int area(Mat src) {
	int height = src.rows;
	int width = src.cols;

	int objArea = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			objArea += isObj(pixel, Vec3b(0, 0, 0));
		}
	}

	return objArea;
}

Vec2f massPoint(Mat src) {
	int height = src.rows;
	int width = src.cols;

	float objRowCoord = 0;
	float objColCoord = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			objRowCoord += j * isObj(pixel, Vec3b(0, 0, 0));
			objColCoord += i * isObj(pixel, Vec3b(0, 0, 0));
		}
	}

	int objArea = area(src);

	objRowCoord *= (1.0 / objArea);
	objColCoord *= (1.0 / objArea);

	return Vec2f(objRowCoord, objColCoord);
}

float axis(Mat src) {
	int height = src.rows;
	int width = src.cols;

	float objRowCols = 0;
	float objRows = 0;
	float objCols = 0;

	Vec2f massP = massPoint(src);
	int objArea = area(src);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			bool isObject = isObj(pixel, Vec3b(0, 0, 0));
			objRowCols += (j - massP[1]) * (i - massP[0]) * isObject;
			objRows += (i - massP[0]) * (i - massP[0]) * isObject;
			objCols += (j - massP[1]) * (j - massP[1]) * isObject;
		}
	}

	objRowCols *= 2;
	return atan2(objRowCols, objCols - objRows);
}

bool isMargin(Mat src, float x, float y) {
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			if (i == 0 && j == 0) {
				continue;
			}

			bool neighbPixel = isInside(src, x + i, y + j);
			if (!neighbPixel) {
				return true;
			}

			if (!isObj(src.at<Vec3b>(x + i, y + j), Vec3b(0, 0, 0))) {
				return true;
			}
		}
	}
	return false;
}

int perimeter(Mat src) {
	int height = src.rows;
	int width = src.cols;

	int perimeter = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			bool isObject = isObj(pixel, Vec3b(0, 0, 0));
			if (isObject) {
				perimeter = isObject ? perimeter + isMargin(src, i, j) : perimeter;
			}
		}
	}

	return perimeter;
}

float aspectRatio(Mat src) {
	int cMin = -1, cMax = -1;
	int rMin = -1, rMax = -1;

	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);

			if (isObj(pixel, Vec3b(0, 0, 0))) {
				if (cMin == -1) {
					cMin = j;
				}
				else if (cMin > j) {
					cMin = j;
				}
				if (cMax == -1) {
					cMax = j;
				}
				else if (cMax < j) {
					cMax = j;
				}
				if (rMin == -1) {
					rMin = i;
				}
				else if (rMin > i) {
					rMin = i;
				}
				if (rMax == -1) {
					rMax = i;
				}
				else if (rMax < i) {
					rMax = i;
				}
			}
		}

		
	}
	float ratio = (1.0 * cMax - cMin + 1) / (1.0 * rMax - rMin + 1);
	return ratio;
}

void horizontal(Mat src) {
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC3);

	int* noPixels = (int*)calloc(height, sizeof(int));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			noPixels[i] += isObj(pixel, Vec3b(0, 0, 0));
		}
	}

	for (int i = 0; i < height; i++) {
		for(int c = 0; c < noPixels[i]; c++){
			dst.at<Vec3b>(i, width - c - 1) = Vec3b(0, 0, 0);
		}
	}

	imshow("horizontal projection", dst);
}

void vertical(Mat src) {
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC3);

	int* noPixels = (int*)calloc(width, sizeof(int));

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			noPixels[j] += isObj(pixel, Vec3b(0, 0, 0));
		}
	}

	for (int i = 0; i < width; i++) {
		for (int c = 0; c < noPixels[i]; c++) {
			dst.at<Vec3b>(height - c - 1, i) = Vec3b(0, 0, 0);
		}
	}

	imshow("vertical projection", dst);
}

void testGeometry(int option)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);

		switch (option) {
		case 0:
			printf("Area is: %d\n", area(src));
			break;

		case 1:
			printf("Mass point is: %.2f, %.2f", massPoint(src)[0], massPoint(src)[1]);
			break;

		case 2:
			printf("The axis is: %.2f\n", axis(src));
			break;

		case 3:
			printf("The perimeter is: %d\n", perimeter(src));
			break;

		case 4:
			printf("Aspect ratio is: %.2f\n", aspectRatio(src));
			break;
		
		case 5:
			horizontal(src);
			break;
		
		case 6:
			vertical(src);
			break;
		}

		waitKey();
	}
}

int label = 0;
Mat generateLabels(Mat src) {
	int height = src.rows;
	int width = src.cols;

	
	Mat labels = Mat::zeros(height, width, CV_32SC1);
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<char>(i, j) == 0 && labels.at<int>(i,j) == 0) {
				Point2i crtNode;
				crtNode.x = i;
				crtNode.y = j;

				Q.push(crtNode);

				label++;
				labels.at<int>(i, j) = label;

				while (!Q.empty()) {
					Point2i crtParent = Q.front();
					Q.pop();
					
					for (int ni = -1; ni <= 1; ni++) {
						for (int nj = -1; nj <= 1; nj++) {
							if (ni == 0 && nj == 0) {
								continue;
							}

							if (!isInside(src, crtParent.x + ni, crtParent.y + nj)) {
								continue;
							}

							if (src.at<char>(crtParent.x + ni, crtParent.y + nj) == 0 
								&& labels.at<int>(crtParent.x + ni, crtParent.y + nj) == 0) {

								labels.at<int>(crtParent.x + ni, crtParent.y + nj) = labels.at<int>(crtParent.x, crtParent.y);

								Q.push(Point2i(crtParent.x + ni, crtParent.y + nj));
							}
						}
					}

				}
			}
		}
	}

	return labels;
}

#include <random>

void generateImgByLabels() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("image", src);

		Mat dst = Mat(src.rows, src.cols, CV_8UC3);
		std::vector<Vec3b> color;

		std::default_random_engine gen;
		std::uniform_int_distribution<int> d(0, 255);

		Mat labels = generateLabels(src);

		for (int i = 0; i <= label; i++) {
			Vec3b c = Vec3b(d(gen), d(gen), d(gen));
			color.push_back(c);
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst.at<Vec3b>(i, j) = color.at(labels.at<int>(i, j));
			}
		}
		
		imshow("result", dst);
		waitKey();
	}
}

std::pair<int, int> computeNext(int dir) {
	switch (dir) {
	case 0:
		return std::pair<int, int>(0, 1);
	case 1:
		return std::pair<int, int>(-1, 1);
	case 2:
		return std::pair<int, int>(-1, 0);
	case 3:
		return std::pair<int, int>(-1, -1);
	case 4:
		return std::pair<int, int>(0, -1);
	case 5:
		return std::pair<int, int>(1, -1);
	case 6:
		return std::pair<int, int>(1, 0);
	case 7:
		return std::pair<int, int>(1, 1);
	}

	return std::pair<int, int>(0, 0);
}

void colourContour(Mat *dst, int i, int j, std::vector<int> directions) {
	dst->at<uchar>(i, j) = 1;

	int prevI = i, prevJ = j;
	int maxSize = directions.size();

	for (int c = 0; c < maxSize; c++) {
		dst->at<uchar>(prevI, prevJ) = 1;

		int crtDir = directions[0];
		std::pair<int, int> off = computeNext(directions[c]);

		prevI = prevI + off.first;
		prevJ = prevJ + off.second;
	}
}

void contour() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("image", src);

		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		std::vector<Vec3b> contour;
		int dir = 7;
		int n = 0;

		bool found = false;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0) {
					found = true;
					printf("%d, %d - dir: %d\n", i, j, dir);
					contour.push_back(Vec3b(i, j, dir));
					break;
				}
			}
			if (found) {
				break;
			}
		}

		bool stopCond;
		do {
			found = false;
			int startDir = dir % 2 == 0 ? (dir + 7) % 8 : (dir + 6) % 8;
			int d = startDir;
			do {
				std::pair<int, int> nextDir = computeNext(d);
				int crtI = contour[n][0] + nextDir.first;
				int crtJ = contour[n][1] + nextDir.second;

				if (src.at<uchar>(crtI, crtJ) == 0) {
					found = true;
					contour.push_back(Vec3b(crtI, crtJ, d));

					dir = d;
					n++;
				}
				else {
					d = (d + 1) % 8;
				}
			} while (!found);

			stopCond = (contour[n][0] == contour[1][0] && contour[n][1] == contour[1][1] && contour[n-1][0] == contour[0][0] && contour[n-1][1] == contour[0][1]) && n > 2;
		} while (!stopCond);

		for (int c = 0; c <= n; c++) {
			dst.at<uchar>(contour[c][0], contour[c][1]) = 0;

		}

		imshow("result", dst);
		waitKey();
	}
}

void getContourFromText() {
	FILE* info;
	info = fopen("reconstruct.txt", "r");
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);

		int i, j, noDir;
		fscanf(info, "%d %d %d", &i, &j, &noDir);

		std::vector<int> directions;
		directions.push_back(7);
		int dir;
		for (int i = 0; i < noDir; i++) {
			fscanf(info, "%d ", &dir);
			directions.push_back(dir);
		}

		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		colourContour(&dst, i, j, directions);

		imshow("result", dst);
		waitKey();
	}

	fclose(info);
}

Mat dilation(Mat src) {

	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	src.copyTo(dst);

	for (int i = 0; i < src.rows - 1; i++) {
		for (int j = 0; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int x = -1; x <= 1; x++) {
					for (int y = -1; y <= 1; y++) {
						if (isInside(src, i+x, j+y)) {
							dst.at<uchar>(i+x, j+y) = 0;
						}
					}
				}
			}
		}
	}

	return dst;
}

Mat erosion(Mat src) {

	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	src.copyTo(dst);


	for (int i = 0; i < src.rows - 1; i++) {
		for (int j = 0; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == 0) {
				for (int x = -1; x <= 1; x++) {
					for (int y = -1; y <= 1; y++) {
						if (isInside(src, i + x, j + y) && src.at<uchar>(i+x, j + y) == 255) {
							dst.at<uchar>(i, j) = 255;
						}
					}
				}
			}
		}
	}

	return dst;
}

Mat closing(Mat src) {
	Mat dst = dilation(src);
	dst = erosion(dst);

	return dst;
}

Mat opening(Mat src) {
	Mat dst = erosion(src);
	dst = dilation(dst);


	return dst;
}

void borderExtraction() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("image", src);

		Mat aux = erosion(src);
		
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0 && aux.at<uchar>(i, j) == 255) {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}

		imshow("result", dst);
		waitKey();
	}
}

Mat complement(Mat src) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = 255 - src.at<uchar>(i, j);
		}
	}

	return dst;
}

void fill() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("image", src);

		Mat aux = complement(src);

		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		bool ok = false;
		for (int i = 0; i < src.rows; i++) {
			ok = false;
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0) {
					dst.at<uchar>(i, j) = 0;
					ok = !ok;
				}
				else if (src.at<uchar>(i, j) != 0 && ok == true) {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}


		imshow("result", dst);
		waitKey();
	}
}


Mat extractBackground(Mat img1, Mat img2) {

	int height = img1.rows;
	int width = img1.cols;
	Mat result = Mat(height, width, CV_8UC1);
	result.setTo(0);
	int th = 20;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if ( abs(img1.at<uchar>(i, j) - img2.at<uchar>(i, j)) > th) {
				result.at<uchar>(i, j) = 0;
			}
			else {
				result.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("hand pre post processing", result);
	
	return closing(result);
}

void test() {

	char fname1[MAX_PATH], fname2[MAX_PATH];
	while (openFileDlg(fname1) && openFileDlg(fname2))
	{
		// reads the image matrix from the .bmp file

		Mat img1, img2;
		Mat resizedImg1, resizedImg2;
		img1 = imread(fname1, CV_LOAD_IMAGE_GRAYSCALE);
		img2 = imread(fname2, CV_LOAD_IMAGE_GRAYSCALE);
		resize(img1, resizedImg1, Size(350, 450), INTER_LINEAR);
		resize(img2, resizedImg2, Size(350, 450), INTER_LINEAR);
		Mat res = extractBackground(resizedImg1, resizedImg2);

		// shows the colour matrix
		imshow("hand", res);

		// waits for the X button to be pressed before reopening the file selection window
		waitKey();
	}
}

void testSubstractBackground() {
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];

	Mat background;
	Mat exhibit;

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	char c;
	int frameNum = -1;
	int frameCount = 0;

	Mat imageSrc;
	Mat mask;

	int history = 1;				// no of frames that affect the final result
	float varThreshold = 16.0;		// threshold's value
	bool shadowDetection = false;	// as the name says

	int th = 20;

	Ptr<BackgroundSubtractor> bckgSub = createBackgroundSubtractorMOG2();
	for(;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imageSrc = testColor2Gray(frame);

		bckgSub->apply(frame, mask, 0.005);

		Mat backgroundImg;
		bckgSub->getBackgroundImage(backgroundImg);

		imshow("frame", frame);
		imshow("mask", mask);
		imshow("background img", backgroundImg);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  // ESC pressed
		}
	}
}


Mat getHue(Mat src) {
	int height = src.rows;
	int width = src.cols;

	Mat H = Mat(height, width, CV_8UC1);

	uchar* lpH = H.data;

	Mat hsvImg;
	cvtColor(src, hsvImg, CV_BGR2HSV);

	uchar* hsvDataPtr = hsvImg.data;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int hi = i * width * 3 + j * 3;
			int gi = i * width + j;

			lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255

		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (H.at<uchar>(i, j) > 150 || H.at<uchar>(i, j) < 50) {
				H.at<uchar>(i, j) = 255;
				continue;
			}

			H.at<uchar>(i, j) = 0;
		}
	}
	return H;
}

Mat getSaturation(Mat src) {
	int height = src.rows;
	int width = src.cols;

	Mat S = Mat(height, width, CV_8UC1);
	uchar* lpS = S.data;

	Mat hsvImg;
	cvtColor(src, hsvImg, CV_BGR2HSV);
	uchar* hsvDataPtr = hsvImg.data;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int hi = i * width * 3 + j * 3;
			int gi = i * width + j;

			lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (S.at<uchar>(i, j) >= 10) {
				S.at<uchar>(i, j) = 255;
			}
			else {
				S.at<uchar>(i, j) = 0;
			}
		}
	}

	return S;
}

void testSkinSegmentation() {
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	char c;
	int frameNum = -1;
	int frameCount = 0;

	Mat skinDetected;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		skinDetected = Mat(frame.rows, frame.cols, CV_8UC1);

		for (int i = 0; i < frame.rows; i++) {
			for (int j = 0; j < frame.cols; j++) {
				int r = frame.at<Vec3b>(i, j)[2];
				int g = frame.at<Vec3b>(i, j)[1];
				int b = frame.at<Vec3b>(i, j)[0];

				// Case 1 - uniform daylight illum
				if (r > 95 && g > 40 && b > 20) {
					if (max(r, g, b) - min(r, g, b) > 15) {
						if (r - g > 15 && r > b) {
							skinDetected.at<uchar>(i, j) = 255;
							continue;
						}
					}
				}

				// Case 2 - lateral illum
				if (r > 220 && g > 210 && b > 170) {
					if (abs(r - g) <= 15 && b < r && b < g) {
						skinDetected.at<uchar>(i, j) = 255;
						continue;
					}
				}
				skinDetected.at<uchar>(i, j) = 0;
			}
		}

		imshow("frame", frame);
		imshow("skin detection", getSaturation(frame) & getHue(frame) & skinDetected);
		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  // ESC pressed
		}
	}
}

Mat skinSegmentation(Mat src) {
	Mat skinRGB = Mat(src.rows, src.cols, CV_8UC1);

	Mat hue = getHue(src);
	Mat saturation = getSaturation(src);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int r = src.at<Vec3b>(i, j)[2];
			int g = src.at<Vec3b>(i, j)[1];
			int b = src.at<Vec3b>(i, j)[0];

			// Case 1 - uniform daylight illum
			if (r > 95 && g > 40 && b > 20) {
				if (max(r, g, b) - min(r, g, b) > 15) {
					if (r - g > 15 && r > b) {
						skinRGB.at<uchar>(i, j) = 255;
						continue;
					}
				}
			}

			// Case 2 - lateral illum
			if (r > 220 && g > 210 && b > 170) {
				if (abs(r - g) <= 15 && b < r && b < g) {
					skinRGB.at<uchar>(i, j) = 255;
					continue;
				}
			}
			skinRGB.at<uchar>(i, j) = 0;
		}
	}


	return skinRGB & hue & saturation;
}

void testHandDetection() {
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	Mat background;
	Mat exhibit;

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	char c;
	int frameNum = -1;
	int frameCount = 0;

	Mat imageSrc;
	Mat mask;

	int history = 1;				// no of frames that affect the final result
	float varThreshold = 16.0;		// threshold's value
	bool shadowDetection = false;	// as the name says

	int th = 20;

	Mat destination;

	Ptr<BackgroundSubtractor> bckgSub = createBackgroundSubtractorMOG2();
	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imageSrc = testColor2Gray(frame);

		bckgSub->apply(frame, mask, 0.005);

		destination = Mat(frame.rows, frame.cols, CV_8UC1);

		Mat detectedSkin = skinSegmentation(frame);

		destination = mask & detectedSkin;

		imshow("frame", frame);
		imshow("final frame", destination);
		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, destination);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}

			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  // ESC pressed
		}
	}
}

// etichetare a mainii pe un fundal fix
//	o treime din bounding box (partea de sus a mainii)
//	suprapunem peste masca pt recunoasterea degetelor

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf("\n\n");
		printf(" 32 - Test extract  background\n");
		printf(" 33 - Test substract background from live photo\n");
		printf(" 34 - Test skin detection\n");
		printf(" 35 - Test hand detection\n");
		printf(" 0  - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				addImage();
				break;
			case 11:
				subImage();
				break;
			case 12:
				testColor2Color();
				break;
			case 13:
				RGBGray();
				break;
			case 14:
				testColorToGrayScale();
				break;
			case 15:
				testGray2BlackWhite();
				break;
			case 16:
				testRGB2HSV();
				break;

			case 18:
				testGeometry(0);
				break;
			case 19:
				testGeometry(1);
				break;
			case 20:
				testGeometry(2);
				break;
			case 21:
				testGeometry(3);
				break;
			case 22:
				testGeometry(4);
				break;
			case 23:
				testGeometry(5);
				break;
			case 24:
				testGeometry(6);
				break;
			case 25:
				generateImgByLabels();
				break;
			case 26:
				contour();
				break;
			case 27:
				getContourFromText();
				break;

			case 30:
				borderExtraction();
				break;
			case 31:
				fill();
				break;

			case 32:
				test();
				break;
			case 33:
				testSubstractBackground();
				break;
			case 34:
				testSkinSegmentation();
				break;
			case 35:
				testHandDetection();
				break;
		}
	}
	
	while (op!=0);
	return 0;
}