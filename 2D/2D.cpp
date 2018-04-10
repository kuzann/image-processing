/*
PROGRAM PENGOLAHAN CITRA 2 DIMENSI 
STUDI KASUS MENGHITUNG KOEFISIEN RESTITUSI BOLA PINGPONG DAN BOLA TENIS TERHADAP LANTAI
By : Ryan Gifari (10213094)
ryangifari@gmail.com
*/

#include<iostream>
#include<fstream>
#include<iomanip>
#include<sstream>
#include<ctime>
#include<math.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/// inisiasi variabel global
int input_menu = 0;
VideoCapture kamera;
ofstream fout;
clock_t t1, t2, t3, start, finish;
int width = 640;
int height = 480;
int cam_port = 0;
const double pi = 3.14159265359;
string ddd = "data_";

struct HSV {
	int h;
	int s;
	int v;
};

/// sub program - capture image
void stereo_capture() {
	Mat img;
	kamera.open(cam_port);

	kamera.set(CV_CAP_PROP_FRAME_WIDTH, width);		//atur panjang
	kamera.set(CV_CAP_PROP_FRAME_HEIGHT, height);	//atur tinggi
		
	char cek;
	string filename, num;
	int i = 1;

	if (kamera.isOpened()) {	//cek apa kamera terbuka atau tidak

		cout << "\nPress 1 to capture and press ESC key to exit...\n";
		for (;;)
		{
			kamera.grab();
			kamera.retrieve(img);
			imshow("Cam 0", img);
			cek = waitKey(1);
			if (cek == '1') {
				stringstream ss;
				ss << i;
				num = ss.str();
				filename = "capture/capture_" + num + ".jpg";
				imwrite(filename, img);
				i++;
			}
			if (cek == 27) { destroyAllWindows(); kamera.release(); break; } //loop break saat tombol "esc" ditekan
		}
	}
	else
	{
		cout << "\nKamera can't open! Back to main menu...\n";
		kamera.release();
	}
}

/// sub program - cari nilai HSV bola dengan trackbar
void data_hsv() {
	Mat src, hsv, tresh;
	string namefile;
	char cek;

	cout << "Input name of an image (.jpg, .gif, .png) : "; cin >> namefile;
	src = imread(namefile);

	if (src.empty()) {
		cout << "Could not open or find the image. Back to main menu ...\n" << endl;
		return;
	}

	HSV max, min;

	min.h = 0; max.h = 180;
	min.s = 0; max.s = 255;
	min.v = 0; max.v = 255;

	cvtColor(src, hsv, CV_BGR2HSV);

	namedWindow("track", CV_WINDOW_AUTOSIZE);

	for (;;)
	{
		createTrackbar("Hue Min", "track", &min.h, 180);
		createTrackbar("Hue Max", "track", &max.h, 180);
		createTrackbar("Sat Min", "track", &min.s, 255);
		createTrackbar("Sat Max", "track", &max.s, 255);
		createTrackbar("Val Min", "track", &min.v, 255);
		createTrackbar("Val Max", "track", &max.v, 255);
		inRange(hsv, Scalar(min.h, min.s, min.v), Scalar(max.h, max.s, max.v), tresh);
		imshow("Original Image", src);
		imshow("Threshold Image", tresh);
		cek = waitKey(1);
		if (cek == 27) { destroyAllWindows(); break; } //loop break saat tombol "esc" ditekan
	}
}

/// sub program - konversi ke HSV
void convert_image(Mat &imgBlur, Mat &imgHSV) 
{
	GaussianBlur			// noise reduction dengan gaussian blur
	(imgBlur,				// input image
		imgBlur,			// output image
		Size(3, 3),			// smoothing window width and height in pixels
		0);					// sigma value, determines how much the image will be blurred

	Mat structuringElement = getStructuringElement(
		MORPH_RECT,
		Size(3, 3),
		Point(-1, -1));

	dilate(imgBlur, imgBlur, structuringElement);	//gambar dominan gelap jadi lebih terang
	erode(imgBlur, imgBlur, structuringElement);	//gambar dominan terang jadi lebih gelap

	cvtColor(imgBlur, imgHSV, CV_BGR2HSV);		//konversi ke HSV
}

/// sub program - filter kontur berdasarkan area
void filter_contour(vector<vector<Point>> &contours,vector<Vec4i> &hierarchy) 
{
	int min_area = 100;		// minimum area threshold
	for (int i = 0; i < contours.size(); i++) // iterasi tiap kontur, min area
	{
		int area = contourArea(contours[i], false); // cari area kontur
		if (area < min_area || area == 0) {
			contours.erase(contours.begin() + i);	// hapus kontur yang tidak sesuai
		}	
	}
}

/// sub program - cari kontur, moment, dan mass center
void find_contour(Mat &imgInput, Mat &imgOutput, vector<vector<Point>> &contours,
	vector<Vec4i> &hierarchy, vector<Moments> &mu, vector<Point2f> &mc) 
{
	findContours(imgInput, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);	// cari kontur
	
	//filter kontur berdasarkan area
	filter_contour(contours, hierarchy);
	filter_contour(contours, hierarchy);
	filter_contour(contours, hierarchy);
	filter_contour(contours, hierarchy);
	filter_contour(contours, hierarchy);
	filter_contour(contours, hierarchy);
	filter_contour(contours, hierarchy);
	filter_contour(contours, hierarchy);
	filter_contour(contours, hierarchy);
	filter_contour(contours, hierarchy);
	
	vector<Moments> muf(contours.size());
	for (int i = 0; i < contours.size(); i++)	// iterasi moment
	{
		muf[i] = moments(contours[i], false);	// cari moment
	}

	vector<Point2f> mcf(contours.size());
	for (int i = 0; i < contours.size(); i++)	// iterasi mass center
	{
		mcf[i] = Point2f(muf[i].m10 / muf[i].m00, muf[i].m01 / muf[i].m00);	// cari mass center
	}

	imgOutput = Mat::zeros(imgInput.size(), CV_8UC1);	// matriks penyimpan gambar hasil deteksi kontur
	for (int i = 0; i< contours.size(); i++)	// iterasi gambar kontur
	{
		Scalar color = Scalar(255);
		drawContours(imgOutput, contours, i, color, 2, 8, hierarchy, 0);	// gambar kontur
		circle(imgOutput, mcf[i], 2, color, -1, 8, 0);
	}
	mu = muf;
	mc = mcf;
}


/// sub program - video offline, olah data offline
void read_video(HSV &max, HSV &min) 
{
	Mat frame, imgOri, imgHSV, imgContour, imgOutput;	// memori penyimpan gambar
	vector<vector<Point> > contours;	// vektor kontur
	vector<Vec4i> hierarchy;		// vektor hirarki
	string filename;		// nama file

	cout << "Input video file name (.avi, .mp4) : "; cin >> filename;	// input filename
	
	kamera.open(filename);		// buka file video

	if (!kamera.isOpened()) {
		cout << "Could not open or find the video. Back to main menu ...\n" << endl;
		kamera.release();
		return;
	}

	int r_fps = 0;		// inisiasi variabel fps dan jumlah frame
	int frame_count = kamera.get(CV_CAP_PROP_FRAME_COUNT);		// ambil total frame video
	double fps = kamera.get(CV_CAP_PROP_FPS);		// ambil fps video
	width = kamera.get(CV_CAP_PROP_FRAME_WIDTH);		// ambil lebar resolusi video
	height = kamera.get(CV_CAP_PROP_FRAME_HEIGHT);		// ambil tinggi resolusi video
	cout << "Video frame size : " << width << "x" << height << endl;	// print resolusi video

	do // iterasi untuk menghitung real frame, mencari frame loss video
	{
		kamera >> frame;
		if (frame.empty()) { break; }
		r_fps++;
	} while (!frame.empty());
	cout << "Frame lost is " << frame_count - r_fps << " and total real frame is " << r_fps
		<< " with " << fps << " fps.\n";	// print frame loss

	double t_total = frame_count / fps;		// panjang waktu video
	double t = 0;		// partisi waktu video
	double dt = (t_total / r_fps);		// waktu 1 frame
	cout << "video length : " << t_total << " .dt : " << dt << endl;	// print waktu video

	kamera.open(filename);			//Buka file video lagi

	double x[10000], y[10000], h_max[10000];		// inisiasi variabel posisi mass center
	int n = -1;		// variabel nilai data awal kemunculan mass center
	int k = 0;		// variabel k buat nilai hmax

	string namadata = "data_position_xy_" + filename + ".txt";		// nama data untuk file output
	fout.open(namadata);		// buka output file
	fout << "n\tt\tx\ty\n";
	start = clock();	// simpan waktu awal seluruh proses
	
	for (int i = 0; i < r_fps; i++)
	{
		///Pengolahan citra - mencari pusat massa pada gambar
		t1 = clock();	// simpan waktu awal satu proses
		kamera >> imgOri;	// simpan memori kamera ke matriks
		if (imgOri.empty()) { destroyAllWindows(); break; }		// cek kalo frame kosong

		convert_image(imgOri, imgHSV);		// konversi gambar
		inRange(imgHSV, Scalar(min.h, min.s, min.v), Scalar(max.h, max.s, max.v), imgContour);	// deteksi warna bola pingpong 

		vector<Moments> mu(contours.size());		// inisiasi variabel momen
		vector<Point2f> mc(contours.size());		// inisiasi variabel titik tengah
		find_contour(imgContour, imgOutput, contours, hierarchy, mu, mc);	// cari kontur

		imshow("Gambar Asli", imgOri);		// tampilkan gambar asli
		imshow("Kontur", imgOutput);		// tampilkan gambar kontur
		imshow("Deteksi", imgContour);		// tampilkan gambar deteksi

		waitKey(int(r_fps / fps));		//Atur waktu jeda antar frame pada video

		///Simpan setiap frame yang dihasilkan pada folder gambar
		stringstream ss;
		ss << i;
		string sk = ss.str();
		string sk1 = "gambar/original_file" + sk + ".jpg";
		string sk2 = "gambar/detection_file" + sk + ".jpg";
		string sk3 = "gambar/masscenter_file" + sk + ".jpg";
		imwrite(sk1, imgOri);
		imwrite(sk2, imgOutput);
		imwrite(sk3, imgContour);

		t2 = clock();	// simpan waktu akhir proses olah citra
		t = t + dt;		// variabel waktu
		
		///Penyimpanan posisi mass center yang diperoleh pada sebuah file txt
		fout << fixed << setprecision(4);	// set presisi nilai file output
		if (mc.size() > 0) // simpan posisi mc ke variabel x dan y
		{
			n++;
			x[i] = int(mc[0].x + 0.5); y[i] = int((height - mc[0].y) + 1 + 0.5);
			x[i] = x[i] * (0.87 / width); y[i] = y[i] * (0.54 / height);		//konversi piksel ke m
			fout << i + 1 << "\t" << t << "\t" << x[i] << "\t" << y[i] << "\t";	// simpan file
		}
		else
		{	//jika nilai mc 0, masuk kebagian ini
			if (
				((x[i] > 0 && x[i] < width) &&
				(y[i] > 0 && y[i] < height)) ||
					((x[i - 1] > 0 && x[i - 1] < width) &&
				(y[i - 1] > 0 && y[i - 1] < height))
				)
			{
				x[i] = x[i - 1];
				y[i] = y[i - 1];
				fout << i + 1 << "\t" << t << "\t" << x[i] << "\t" << y[i] << "\t\n";
				n++;
			}
		}

		///Cari posisi benda saat posisi tertinggi
		double h1, h2, h3;	//inisiasi variabel ketinggian

		if (i>1 && (y[i - 1] >= y[i - 2] && y[i - 1] > y[i]))	//perbandingan ketinggian bola
		{
			h1 = y[i - 2];
			h2 = y[i - 1];
			h3 = y[i];
			int j = i;
			while ((h1 == h2 && h2 >= h1) && j >= (i - n))	// bila nilai h1 dan h2 sama
			{
				h1 = y[j - 2];
				j--;
			}
			if (h2 > h1 && h2 > h3)
			{
				h_max[k] = h2;
				k++;
			}
		}
		t3 = clock();		// simpan waktu akhir keseluruhan satu proses
		if (i + 1 == r_fps) { destroyAllWindows(); break; }	// break windows saat frame sudah kosong	
	}

	finish = clock();		// simpan waktu akhir keseluruhan proses
	fout << "\nData succesfully saved with time process = "
		<< ((double)(finish - start) / CLOCKS_PER_SEC) << " s.\n";
	fout.close();		// close file text
	kamera.release();	// close file video
	
	///Hitung koefisien restitusi
	namadata = "data_restitusi" + filename + ".txt";		//nama file output
	fout.open(namadata);		//buka file output
	fout << "h1\th2\te(h)\n";
	dt = 0;
	for (int i = 1; i < k; i++)		// simpan dalam file
	{
		double eh = sqrt(h_max[i] / h_max[i - 1]);	// e bergantung h
		fout << fixed << setprecision(3);			// presisi data
		fout << h_max[i - 1] << "\t" << h_max[i] << "\t" << eh << endl;	// tulis file ke data
	}
	fout.close();		// tutup file
}

/// sub program - get all frame from a video file
void get_frame() {
	
	string filename;
	cout << "Input video file name (.avi, .mp4) : "; cin >> filename;

	kamera.open(filename);

	if (!kamera.isOpened()) {
		cout << "Could not open or find the video. Back to main menu ...\n" << endl;
		kamera.release();
		return;
	}

	Mat frame, save;

	fout.open(filename+"_properties.txt");

	int r_fps = 0;		// inisiasi variabel fps dan jumlah frame
	int frame_count = kamera.get(CV_CAP_PROP_FRAME_COUNT);		// ambil total frame video
	double fps = kamera.get(CV_CAP_PROP_FPS);		// ambil fps video
	width = kamera.get(CV_CAP_PROP_FRAME_WIDTH);		// ambil lebar resolusi video
	height = kamera.get(CV_CAP_PROP_FRAME_HEIGHT);		// ambil tinggi resolusi video
	fout << "Video frame size : " << width << "x" << height << endl;	// print resolusi video

	do { // iterasi untuk menghitung real frame, mencari frame loss video
		kamera >> frame;
		if (frame.empty()) { break; }
		r_fps++;
	} while (!frame.empty());

	fout << "Frame lost is " << frame_count - r_fps << " and total real frame is " << r_fps
		<< " with " << fps << " fps.\n";	// print frame loss ke file txt

	double t_total = frame_count / fps;		// panjang waktu video
	double t = 0;		// partisi waktu video
	double dt = (t_total / r_fps);		// waktu 1 frame
	fout << "video length : " << t_total << " .dt : " << dt << endl;	// print waktu video ke file txt

	kamera.open(filename);	//buka file video lagi

	for (int j = 0; j < r_fps; j++) {
		kamera >> save;
		stringstream ss;
		ss << j;
		string sk = ss.str();

		string outfile;		//penamaan file output
		if (j < 10) { outfile = "video/frame000000" + sk + ".jpg"; }
		if (j < 100 && j >= 10) { outfile = "video/frame00000" + sk + ".jpg"; }
		if (j < 1000 && j >= 100) { outfile = "video/frame0000" + sk + ".jpg"; }
		if (j < 10000 && j >= 1000) { outfile = "video/frame000" + sk + ".jpg"; }
		if (j < 100000 && j >= 10000) { outfile = "video/frame00" + sk + ".jpg"; }
		if (j < 1000000 && j >= 100000) { outfile = "video/frame0" + sk + ".jpg"; }
		if (j < 10000000 && j >= 1000000) { outfile = "video/frame" + sk + ".jpg"; }

		imwrite(outfile, save);		//simpan frame ke sebuah folder
	}
	cout << "The process is done! Back to main menu ... \n";
	fout.close();
	kamera.release();
}

/// Main Program
int main() 
{
	char select = 0;	// variabel input menu
	HSV max, min;
	min.h = 0; max.h = 32;
	min.s = 109; max.s = 215;
	min.v = 120; max.v = 255;

	cout << "\n-------------------------------------------------------------"; 
	cout << "\nPROGRAM PENGOLAHAN CITRA DIGITAL 2 DIMENSI MENGGUNAKAN KAMERA";
	cout << "\n                  CREATED BY RYAN GIFARI";
	cout << "\n                   ryangifari@gmail.com";
	cout << "\n-------------------------------------------------------------\n";

	do {
		cout << "\nUntuk memilih menu, input nilai yang sesuai dengan nomor menu\n";
		cout << "\n1. Ambil gambar dari kamera(pastikan memiliki nama folder \"capture\" dalam folder yang sama dengan program)";
		cout << "\n2. Ambil frame dari file video (pastikan memiliki nama folder \"video\" dalam folder yang sama dengan program)";
		cout << "\n3. Cari nilai data HSV dari sebuah gambar";
		cout << "\n4. Ambil data dari file video hasil kamera stereo";
		cout << "\n5. Cek default nilai HSV, resolusi kamera (width,height), dan nomor port kamera stereo";
		cout << "\n6. Ubah resolusi kamera";
		cout << "\n7. Ubah nomor port kamera";
		cout << "\n8. Ubah nilai HSV untuk pengolahan citra";
		cout << "\n0. Exit Program";
		cout << "\n\nSelect Menu = "; cin >> input_menu;
	
		while (input_menu < 0 || input_menu > 8) {
			cout << "Wrong menu input. Input menu again = "; cin >> input_menu;
		}

		if (input_menu == 1) {
			stereo_capture();
		}

		if (input_menu == 2) {
			get_frame();
		}

		if (input_menu == 3) {
			data_hsv();
		}

		if (input_menu == 4) {
			read_video(max, min);
		}

		if (input_menu == 5) {
			cout << "\nBatas atas nilai HSV : (" << max.h << "," << max.s << "," << max.v << ")";
			cout << "\nBatas bawah nilai HSV : (" << min.h << "," << min.s << "," << min.v << ")";
			cout << "\n\nResolusi kamera : " << width << "x" << height;
			cout << "\nNomor port kamera : " << cam_port << "\n\n";
		}

		if (input_menu == 6) {
			cout << "\nInput panjang (width) kamera : "; cin >> width;
			cout << "Input tinggi (height) kamera : "; cin >> height;
			cout << "\nResolusi kamera berhasil diganti! Back to main menu ...\n\n";
		}

		if (input_menu == 7) {
			cout << "\nInput nomor port baru : "; cin >> cam_port;
			cout << "\nNomor port kamera berhasil diganti! Back to main menu ...\n\n";
		}
		
		if (input_menu == 8) {
			cout << "\nInput nilai batas atas H (Hue) : "; cin >> max.h;
			cout << "Input nilai batas atas S (Saturation) : "; cin >> max.s;
			cout << "Input nilai batas atas V (Value) : "; cin >> max.v;
			cout << "\nInput nilai batas bawah H (Hue) : "; cin >> min.h;
			cout << "Input nilai batas bawah S (Saturation) : "; cin >> min.s;
			cout << "Input nilai batas bawah V (Value) : "; cin >> min.v;
			cout << "\nNilai HSV berhasil diganti! Back to main menu ...\n\n";
		}
	} while (input_menu != 0);
	return 0;
}
