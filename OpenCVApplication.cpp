// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>
#include <iostream>


std::default_random_engine gen;
std::uniform_int_distribution<int> d(0, 255);
uchar x = d(gen);

const int alpha_slider_max = 255;
int alpha_slider;
double alpha;
double beta;

typedef enum {Aria = 0, CentruDeMasaX = 1, AxaDeAlungire = 2, Perimetru = 3, FactorulDeSubtiere = 4, Elongatia = 5,CentruDeMasaY = 6} PropertyType;
std::map<PropertyType, float> map;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
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
		imshow("source", frame);
		imshow("gray", grayFrame);
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
		if (c == 115) { //'s' pressed - snapp the image to a file
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
void showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

																		 //computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
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

void threeImg()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		Mat c1(256, 256, CV_8UC1);
		Mat c2(256, 256, CV_8UC1);
		Mat c3(256, 256, CV_8UC1);
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
			{
				//Vec3b pixel = src.at< Vec3b>(i, j);
				c1.at<uchar>(i, j) = src.at< Vec3b>(i, j)[0];
				c2.at<uchar>(i, j) = src.at< Vec3b>(i, j)[1];
				c3.at<uchar>(i, j) = src.at< Vec3b>(i, j)[2];
			}
		imshow("image1", c1);
		imshow("image2", c2);
		imshow("image3", c3);
		waitKey();
	}
}

void Grayscale()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		Mat c1(256, 256, CV_8UC1);
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
			{
				c1.at<uchar>(i, j) = (src.at< Vec3b>(i, j)[0] + src.at< Vec3b>(i, j)[1] + src.at< Vec3b>(i, j)[2]) / 3;
			}
		imshow("image1", c1);
		waitKey();
	}
}

void Convert(int prag)
{
	char fname[MAX_PATH];

	Mat src = imread("D:/30237/PuscasVlad/OpenCVApplication-VS2013_2413_basic/Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat c1(256, 256, CV_8UC1);
	//src = imread(fname, CV_LOAD_IMAGE_COLOR);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) < prag)
				c1.at<uchar>(i, j) = 0;
			else
				c1.at<uchar>(i, j) = 255;
		}
	imshow("image1", c1);
	waitKey();

}

void on_trackbar(int, void*)
{
	alpha = (double)alpha_slider;

	Convert(alpha);

}

void HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat c1(src.rows, src.cols, CV_8UC1);
		Mat c2(src.rows, src.cols, CV_8UC1);
		Mat c3(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
			{
				float b = (float)src.at<Vec3b>(i, j)[0];// / 255;
				float g = (float)src.at<Vec3b>(i, j)[1];// / 255;
				float r = (float)src.at<Vec3b>(i, j)[2];// / 255;
				float M = max(max(r, g), b);
				float m = min(max(r, g), b);
				float C = M - m;
				c1.at<uchar>(i, j) = M;
				if (C)
					c2.at<uchar>(i, j) = C / c1.at<uchar>(i, j) * 255;
				else
					c2.at<uchar>(i, j) = 0;
				if (C)
				{
					if (M == r)
						c3.at<uchar>(i, j) = 60 * (g - b) / C;
					if (M == g)
						c3.at<uchar>(i, j) = 120 + 60 * (b - r) / C;
					if (M == b)
						c3.at<uchar>(i, j) = 240 + 60 * (r - g) / C;

				}
				else
				{
					c3.at<uchar>(i, j) = 0;
				}
				if (c3.at<uchar>(i, j) < 0)
					c3.at<uchar>(i, j) = c3.at<uchar>(i, j) + 360;
				c3.at<uchar>(i, j) = c3.at<uchar>(i, j) * 255 / 360;
			}
		imshow("V", c1);
		imshow("S", c2);
		imshow("H", c3);
		waitKey();
	}
}

void computeHistogram(int* hist)
{
	Mat src = imread("D:/30237/PuscasVlad/OpenCVApplication-VS2013_2413_basic/Images/Lena_gray.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			hist[src.at<uchar>(i, j)]++;
}

void computeFDP(Mat &src, float* fdp, int* h)
{
	for (int i = 0; i < 256; i++)
		fdp[i] = (float)h[i] / (src.rows * src.cols);

}

void computeHistogramM(Mat &src, int bins, int* hist)
{
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			int index = src.at<uchar>(i, j) / 256.0 * bins;
			hist[index]++;
		}
	showHistogram("Histo", hist, bins, 100);
}

void computeReducedGray(float* fdp, int th, float wh, Mat &src)
{
	vector<int> m;
	m.push_back(0);
	for (int i = wh; i < 255 - wh; i++)
	{
		float v;
		for (int j = i - wh; j <= i + wh; j++)
		{
			v += fdp[j];
		}
		v /= (2 * wh + 1);
		if (fdp[i]>v + th)
		{
			bool k = true;
			for (int j = i - wh; j <= i + wh; j++)
			{
				if (fdp[i] < fdp[j])
					k = false;
			}
			if (k)
				m.push_back(fdp[i]);
		}
	}
	m.push_back(255);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{

			//binary search for closest max

			/*pixel_vechi: = pixel(x, y)
			pixel_nou : = cel_mai_apropiat_maxim_din_histogramă(pixel_vechi)
			pixel(x, y) : = pixel_nou
			eroare : = pixel_vechi - pixel_nou
			pixel(x + 1, y) : = pixel(x + 1, y) + 7 * eroare / 16
			pixel(x - 1, y + 1) : = pixel(x - 1, y + 1) + 3 * eroare / 16
			pixel(x, y + 1) : = pixel(x, y + 1) + 5 * eroare / 16
			pixel(x + 1, y + 1) : = pixel(x + 1, y + 1) + eroare / 16*/
		}
}

float computeArray(Mat src, int x, int y)
{
	Vec3b collor = src.at<Vec3b>(x, y);
	int aria = 0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j) == collor)
				aria++;
		}
	}
	return aria;
}

Point centruDeMasa(Mat src, int x, int y)
{
	Vec3b collor = src.at<Vec3b>(x, y);
	int aria = computeArray(src, x, y);
	float newX = 0, newY = 0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j) == collor)
			{
				newX += i;
				newY += j;
			}
				
		}
	}
	Point center(newX / aria, newY / aria);
	return center;
}

float computeAxaDeAlungire(Mat src, int x, int y)
{
	Vec3b collor = src.at<Vec3b>(x, y);
	Point center = centruDeMasa(src, x, y);
	float numarator=0, numitor1=0, numitor2=0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j) == collor)
			{
				numarator += (i - center.x)*(j - center.y);
				numitor1 += (j - center.y)*(j - center.y);
				numitor2 += (i - center.x)*(i - center.x);
			}
		}
	}
	numarator *= 2;
	
	return atan2(numarator, numitor1 - numitor2);
}

int computePerimetru(Mat src, int x, int y)
{
	Vec3b collor = src.at<Vec3b>(x, y);
	int perimetru = 0;
	for (int i = 1; i < src.rows-1; i++)
	{
		for (int j = 1; j < src.cols-1; j++)
		{
			if (src.at<Vec3b>(i, j) == collor)
			{
				if (src.at<Vec3b>(i - 1, j - 1) != src.at<Vec3b>(i, j)
					|| src.at<Vec3b>(i - 1, j) != src.at<Vec3b>(i, j)
					|| src.at<Vec3b>(i - 1, j + 1) != src.at<Vec3b>(i, j)
					|| src.at<Vec3b>(i, j - 1) != src.at<Vec3b>(i, j)
					|| src.at<Vec3b>(i, j + 1) != src.at<Vec3b>(i, j)
					|| src.at<Vec3b>(i + 1, j - 1) != src.at<Vec3b>(i, j)
					|| src.at<Vec3b>(i + 1, j + 1) != src.at<Vec3b>(i, j)
					|| src.at<Vec3b>(i + 1, j) != src.at<Vec3b>(i, j))
					perimetru++;
			}
		}
	}
	return perimetru;
}

float computeSubtiere(Mat src, int x, int y)
{
	int aria = computeArray(src, x, y);
	int perimetru = computePerimetru(src, x, y);
	return (4 * PI*(aria / (perimetru*perimetru)));
}

float computeElongatia(Mat src, int x, int y)
{
	Vec3b collor = src.at<Vec3b>(x, y);
	int xMin = src.cols, xMax = 0, yMin = src.rows, yMax = 0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j) == collor)
			{
				if (j < xMin)
					xMin = j;
				if (j > xMax)
					xMax = j;
				if (i < yMin)
					yMin = i;
				if (i > yMax)
					yMax = i;
			}
		}
	}
	return ((yMax - yMin + 1) / (xMax - yMin + 1));
}

void computeProiectii(Mat src, int x, int y, int* oriz, int* vert)
{
	Vec3b collor = src.at<Vec3b>(x, y);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j) == collor)
			{
				oriz[i]++;
				vert[j]++;
			}
		}
	}
}

void computeProcGreom(Mat &src,int x, int y)
{
	map[Aria] = computeArray(src, x, y);
	float xC = centruDeMasa(src, x, y).x;
	float yC = centruDeMasa(src, x, y).y;
	map[CentruDeMasaX] = xC;
	map[CentruDeMasaY] = yC;
	map[AxaDeAlungire] = computeAxaDeAlungire(src, x, y);
	map[Perimetru] = computePerimetru(src, x, y);
	map[FactorulDeSubtiere] = computeSubtiere(src, x, y);
	map[Elongatia] = computeElongatia(src, x, y);
	int* oriz = new int[src.rows*src.cols];
	int* vert = new int[src.rows*src.cols];
	computeProiectii(src, x, y, oriz, vert);
	showHistogram("Oriz", oriz, src.cols, src.rows);
	showHistogram("Vert", vert, src.cols, src.rows);
}

void callback(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		computeProcGreom(*src, y, x);
		printf("Aria = %f\n", map[Aria]);
		printf("CX = %f\n", map[CentruDeMasaX]);
		printf("CY = %f\n", map[CentruDeMasaY]);
		printf("Axa = %f\n", map[AxaDeAlungire]);
		printf("Perimetru = %f\n", map[Perimetru]);
		printf("Subtiere = %f\n", map[FactorulDeSubtiere]);
		printf("Elongatia = %f\n", map[Elongatia]);
	}
}


Mat label(Mat &src, int nr)
{
	Mat dest = Mat::zeros(src.rows, src.cols, CV_8UC3);
	Mat lables = Mat::zeros(src.rows, src.cols, CV_32S);

	int dx[] = { -1,0,1,0,-1,1,1,-1 };
	int dy[] = { 0,1,0,-1,1,1,-1,-1 };
	int label = 0;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0 && lables.at<int>(i, j) == 0) {
				label++;
				std::queue<Point> q;
				lables.at<int>(i, j) = label;
				Point p = Point(i, j);
				q.push(p);
				while (!q.empty()) {
					Point s = q.front();
					q.pop();
					for (int k = 0; k < nr; k++) {
						if (src.at<uchar>(s.x,s.y) == 0 && lables.at<int>(s.x + dx[k], s.y + dy[k]) == 0) {
							lables.at<int>(s.x + dx[k], s.y + dy[k]) = label;
							Point x = Point(s.x + dx[k], s.y + dy[k]);
							q.push(x);
						}
					}
				}
			}
		}
	}

	std::vector<Vec3b> colors;
	for (int i = 0; i < label; i++) {
		colors.push_back(Vec3b(d(gen), d(gen), d(gen)));
	}
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (lables.at<int>(i, j) != 0) {
				dest.at<Vec3b>(i, j) = colors[lables.at<int>(i, j)];
			}
		}
	}
	
	return dest;
}

Mat labelTwoPass(Mat &src, int x=0) {
	Mat dest = Mat::zeros(src.rows, src.cols, CV_8UC3);
	Mat lables = Mat::zeros(src.rows, src.cols, CV_32S);

	int dx[] = { -1,-1,0,1 };
	int dy[] = { 0,1,1,1 };
	int label = 0;

	std::vector<vector<int>> edges;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0 && lables.at<int>(i, j) == 0) {
			std:vector<int> L;
				
				for (int k = 0; k < 4; k++) {
					if (lables.at<int>(i + dx[k], j + dy[k]) > 0) {
						L.push_back(lables.at<int>(i + dx[k], j + dy[k]));
					}
					if (L.size() == 0) {
						label++;
						lables.at<int>(i, j) = label;
						edges.push_back(vector<int>());
					}
					else {
						int m = *std::min_element(L.begin(), L.end());
						lables.at<int>(i, j) = m;
						for (int l = 0; l < L.size(); l++) {
							if (L[l] != m) {
								edges[m].push_back(L[l]);
								edges[L[l]].push_back(m);
							}
						}
					}
				}
			}
		}
	}

	int newLabel = 0;
	std::vector<int> newLables = std::vector<int>(label + 1, 0);
	for (int i = 0; i < label; i++) {
		if (newLables[i] == 0) {
			newLabel++;
			std::queue<int> q;
			newLables[i] = newLabel;
			q.push(i);
			while (!q.empty()) {
				int x = q.front();
				q.pop();
				for (int j = 0; j < edges[x].size(); j++) {
					if (newLables[j] == 0) {
						newLables[j] = newLabel;
						q.push(j);
					}
				}
			}
		}
	}
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.rows; j++) {
			lables.at<int>(i, j) = newLables[lables.at<int>(i, j)];
		}
	}
	std::vector<Vec3b> colors;
	for (int i = 0; i < label; i++) {
		colors.push_back(Vec3b(d(gen), d(gen), d(gen)));
	}
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (lables.at<int>(i, j) != 0) {
				dest.at<Vec3b>(i, j) = colors[lables.at<int>(i, j)];
			}
		}
	}
	return dest;
}

void parcurgeContur(Mat &src) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	dst.setTo(0);
	Point start;

	bool ok = true;
	for (int y = 0; y < src.rows && ok == true; y++) {
		for (int x = 0; x < src.cols && ok == true; x++) {
			if (src.at<uchar>(y, x) == 0) {
				start = Point(x, y);
				ok = false;
			}
		}
	}


	vector<int>chain = vector<int>();
	vector<Point>vecini = { Point(1,0),Point(1,-1),Point(0,-1),Point(-1,-1),Point(-1,0),Point(-1,1),Point(0,1),Point(1,1) };
	int dir = 7;
	Point crt = start;
	do {
		if (dir % 2 == 0)
			dir = (dir + 7) % 8;
		else
			dir = (dir + 6) % 8;
		while (src.at<uchar>(crt + vecini[dir]) != 0) {
			dir = (dir + 1) % 8;
		}
		chain.push_back(dir);
		dst.at<uchar>(crt + vecini[dir]) = 128;
		crt = crt + vecini[dir];
	} while (chain.size() < 2 || start != (crt - vecini[chain[0]]) && (start + vecini[chain[1]]) != crt);

	for (int i = 0; i < chain.size(); i++) {
		printf("%d ", chain[i]);
	}

	vector<int> der = vector<int>();
	int d;
	for (int i = 0; i < chain.size() - 1; i++) {
		d = chain[i + 1] - chain[i];
		if (d < 0) {
			d += 8;
		}
		der.push_back(d);
	}
	der[chain.size() - 1] = chain[chain.size() - 1] - chain[0];
	printf("\n");
	for (int i = 0; i < der.size(); i++) {
		printf("%d ", der[i]);
	}
	imshow("contur", dst);
	waitKey();
}
void citere() {
	Point start;
}

Mat dilatare(Mat src, int et, int ks) {
	ks = ks | 1;
	Mat dst;
	do {
		dst = src.clone();
		for (int y = ks / 2; y < src.rows - ks / 2; y++) {
			for (int x = ks / 2; x < src.cols - ks / 2; x++) {
				if (src.at<uchar>(y, x) == 0) {
					for (int dy = -ks / 2; dy <= ks / 2; dy++) {
						for (int dx = -ks / 2; dx <= ks / 2; dx++) {
							dst.at<uchar>(y + dy, x + dx) = 0;
						}
					}
				}
			}
		}
		src = dst;
		et--;
	} while (et);
	return dst;
}

Mat eroziune(Mat src, int et, int ks) {
	ks = ks | 1;
	Mat dst;
	do {
		dst = src.clone();
		for (int y = ks / 2; y < src.rows - ks / 2; y++) {
			for (int x = ks / 2; x < src.cols - ks / 2; x++) {
				if (src.at<uchar>(y, x) == 255) {
					for (int dy = -ks / 2; dy <= ks / 2; dy++) {
						for (int dx = -ks / 2; dx <= ks / 2; dx++) {
							dst.at<uchar>(y + dy, x + dx) = 255;
						}
					}
				}
			}
		}
		src = dst;
		et--;
	} while (et);
	return dst;
}

Mat deschidere(Mat src, int ks, int et) {
	Mat d;
	do {
		d = eroziune(src, 1, ks);
		d = dilatare(d, 1, ks);
		et--;
	} while (et);
	return d;
}
Mat inchidere(Mat src, int ks, int et) {
	Mat d;
	do {
		d = dilatare(src, 1, ks);
		d = eroziune(d, 1, ks);
		et--;
	} while (et);
	return d;
}
Mat dif(Mat src, Mat src2) {
	Mat dst=Mat(src.rows,src.cols,CV_8UC1);
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			dst.at<uchar>(x, y) = 255;
		}
	}
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			if (src.at<uchar>(x, y) == 0 && src2.at<uchar>(x, y) == 255) {
				dst.at<uchar>(x, y) = 0;
			}
		}
	}
	return dst;
}
Mat contur(Mat src) {
	Mat dst;
	dst = eroziune(src, 1, 8);
	dst = dif(src, dst);
	return dst;
}
Mat intersectie(Mat src, Mat s)
{
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			dst.at<uchar>(y, x) = 255;
		}
	}
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (src.at<uchar>(y, x) == s.at<uchar>(y, x) && src.at<uchar>(y, x) == 0)
				dst.at<uchar>(y, x) = 0;
		}
	}
	return dst;
}

bool compare(Mat s1, Mat s2)
{
	for (int y = 0; y < s1.rows; y++)
	{
		for (int x = 0; x < s1.cols; x++)
		{
			if (s1.at<uchar>(y, x) != s2.at<uchar>(y, x))
				return false;
		}
	}
	return true;
}

Mat complement(Mat src)
{
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			dst.at<uchar>(y, x) = 255;
		}
	}
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (src.at<uchar>(y, x) == 0)
				dst.at<uchar>(y, x) = 255;
			else
				dst.at<uchar>(y, x) = 0;
		}
	}
	return dst;
}
Mat reuniune(Mat src, Mat s)
{
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			dst.at<uchar>(y, x) = 255;
		}
	}
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (src.at<uchar>(y, x) == 0 || s.at<uchar>(y, x) == 0)
				dst.at<uchar>(y, x) = 0;
		}
	}
	return dst;
}
Mat umplere(Mat src)
{
	bool ok = true;
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			dst.at<uchar>(y, x) = 255;
		}
	}
	dst.at<uchar>(72, 74) = 0;
	Mat d1;
	Mat compl = complement(src);
	ok = true;
	do {
		d1 = dilatare(dst, 1, 3);
		d1 = intersectie(d1, compl);
		if (compare(d1, dst))
			ok = false;
		else
			dst = d1;
	} while (ok);
	dst = reuniune(src, dst);
	return dst;
}

int main()
{
	int op;
	int* hist = new int[256]();
	float* fdp = new float[256]();
	Mat src = imread("E:/Facultate/PI/OpenCVApplication-VS2013_2413_basic/Images/reg1neg1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src1 = imread("D:/Facultate/PI/OpenCVApplication-VS2013_2413_basic/trasaturi_geom.bmp", CV_LOAD_IMAGE_COLOR);
	Mat dest, dest1, aux;

	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Resize image\n");
		printf(" 4 - Process video\n");
		printf(" 5 - Snap frame from live video\n");
		printf(" 6 - Mouse callback demo\n");
		printf(" 7 - 3 imagini\n");
		printf(" 8 - Convert to grayscale\n");
		printf(" 9 - Black & White\n");
		printf(" 10 - HSV\n");
		printf(" 11 - Histogram\n");
		printf(" 12 - Compute FDP\n");
		printf(" 13 - Compute reduced histo\n");
		printf(" 14 - Reduced nr of gray\n");
		printf(" 15 - Lab4\n");
		printf(" 16 - Lab5\n");
		printf(" 17 - Lab5/2\n");
		printf(" 18 - Lab6\n");
		printf(" 19 - Lab7\n");
		printf(" 20 - Lab7/2\n");
		printf(" 21 - Lab7/3\n");
		printf(" 22 - Lab7/4\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);

		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testResize();
			break;
		case 4:
			testVideoSequence();
			break;
		case 5:
			testSnap();
			break;
		case 6:
			testMouseClick();
			break;
		case 7:
			threeImg();
			break;
		case 8:
			Grayscale();
			break;
		case 9:
			namedWindow("Linear Blend");
			createTrackbar("image1", "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar);
			on_trackbar(alpha_slider, 0);
			break;
		case 10:
			HSV();
			break;
		case 11:
			computeHistogram(hist);
			showHistogram("Histo", hist, 255, 100);
			break;
		case 12:
			computeHistogram(hist);
			computeFDP(src, fdp, hist);
			break;
		case 13:
			computeHistogramM(src, 50, hist);
			break;
		case 14:
			break;
		case 15:
			imshow("im", src1);
			setMouseCallback("im", callback, &src1);
			waitKey();
			break;
		case 16:
			dest = label(src, 4);
			dest1 = label(src, 8);
			imshow("dest", dest);
			imshow("dest1", dest1);
			waitKey();
			break;
		case 17:
			dest = labelTwoPass(src);
			imshow("dest", dest);
			waitKey();
			break;
		case 18:
			parcurgeContur(src);
			break;
		case 19:
			aux = dilatare(src, 2, 8);
			imshow("1", src);
			imshow("2", aux);
			waitKey();
			break;
		case 20:
			aux = eroziune(src, 2, 8);
			imshow("1", src);
			imshow("2", aux);
			waitKey();
			break;
		case 21:
			aux = deschidere(src, 2, 8);
			imshow("1", src);
			imshow("2", aux);
			waitKey();
			break;
		case 22:
			aux = inchidere(src, 5, 8);
			imshow("1", src);
			imshow("2", aux);
			waitKey();
			break;
		case 23:
			aux = contur(src);
			imshow("1", src);
			imshow("2", aux);
			waitKey();
			break;
		case 24:
			aux = umplere(src);
			imshow("1", src);
			imshow("2", aux);
			waitKey();
			break;
		}
		
	}
	while (op != 0);

	return 0;
}