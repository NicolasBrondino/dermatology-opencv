#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include "Utilities.h"
//for the clear console command
#include <cstdlib>


using namespace std;
using namespace cv;
int delaftuse = 0;
int del2 = 0;
int del3 = 0;
int del4 = 0;

template <typename T>
bool IsInBounds(const T& value, const T& low, const T& high) {
	return !(value < low) && (value < high);
}

static void Draw1DHistogram(MatND histograms[], int number_of_histograms, Mat& display_image)
{
	int number_of_bins = histograms[0].size[0];
	double max_value = 0, min_value = 0;
	double channel_max_value = 0, channel_min_value = 0;
	for (int channel = 0; (channel < number_of_histograms); channel++)
	{
		minMaxLoc(histograms[channel], &channel_min_value, &channel_max_value, 0, 0);
		max_value = ((max_value > channel_max_value) && (channel > 0)) ? max_value : channel_max_value;
		min_value = ((min_value < channel_min_value) && (channel > 0)) ? min_value : channel_min_value;
	}
	float scaling_factor = ((float)256.0) / ((float)number_of_bins);

	Mat histogram_image((int)(((float)number_of_bins)*scaling_factor) + 1, (int)(((float)number_of_bins)*scaling_factor) + 1, CV_8UC3, Scalar(255, 255, 255));
	display_image = histogram_image;
	line(histogram_image, Point(0, 0), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
	line(histogram_image, Point(histogram_image.cols - 1, histogram_image.rows - 1), Point(0, histogram_image.rows - 1), Scalar(0, 0, 0));
	int highest_point = static_cast<int>(0.9*((float)number_of_bins)*scaling_factor);
	for (int channel = 0; (channel < number_of_histograms); channel++)
	{
		int last_height;
		for (int h = 0; h < number_of_bins; h++)
		{
			float value = histograms[channel].at<float>(h);
			int height = static_cast<int>(value*highest_point / max_value);
			int where = (int)(((float)h)*scaling_factor);
			if (h > 0)
				line(histogram_image, Point((int)(((float)(h - 1))*scaling_factor) + 1, (int)(((float)number_of_bins)*scaling_factor) - last_height),
					Point((int)(((float)h)*scaling_factor) + 1, (int)(((float)number_of_bins)*scaling_factor) - height),
					Scalar(channel == 0 ? 255 : 0, channel == 1 ? 255 : 0, channel == 2 ? 255 : 0));
			last_height = height;
		}
	}
}

bool firstResult = false;

int main(int argc, const char** argv)
{
	int picloop = 1;
	// do for each pic then start again at beginning of pic list
	while(1)
	{
		/*********************** pic loading ********************************/
		//constructs a path name for each of the files in the media folder
		string name1 = "C:\\Users\\User\\Desktop\\health - test 01\\media\\pic";
		string name2 = ".jpg";
		string currentpiccount = to_string(picloop);
		// each name is pic[number].jpg ex: pic1.jpg to pic12.jpg
		picloop++;
		if (picloop == 13)
			picloop = 1;
		//default testing picture
		Mat canard = imread("c:\\canard.jpg");
		// current image is loaded
		Mat cherry = imread(name1 + currentpiccount + name2);
		/********************eof pic loading ****************************/

		/************** some preprocessing ***********************************/
		//seems a little useful, question this at optimization phase
		Mat cherryClone = cherry.clone();
		Mat HLScherry;
		cvtColor(cherry, HLScherry, CV_BGR2HLS);
		cvtColor(HLScherry, cherry, CV_HLS2BGR);
		/************** eof some preprocessing ***********************************/

		/************** contrast then...... ***********************************/
		double alpha = 0.7;
		int beta = 1;
		int optimalPicFound = 0;
		/* This large loop will try various contrast setting and then stop at the optimal on...*/
		for (double i = 0.1; i < 0.3; i = i + 0.01)//from dark to clear
		{
			if (optimalPicFound == 0)
			{
				/* try a contrast setting*/
				Mat cherryTemp = cherry.clone();
				alpha = alpha + i;
				//alter contrast for all pixels
				for (int y = 0; y < cherry.rows; y++)
				{
					for (int x = 0; x < cherry.cols; x++)
					{
						for (int c = 0; c < 3; c++)
						{
							//alter contrast of current pixel
							cherryTemp.at<Vec3b>(y, x)[c] =
								saturate_cast<uchar>(alpha*(cherryTemp.at<Vec3b>(y, x)[c]) + beta);
						}
					}
				}


				int h = cherry.rows;
				int w = cherry.cols;
				int Xresize = 218;//218
				int Yresize = 195;//195
				resize(cherryClone, cherryClone, Size(Xresize, Yresize));
				resize(cherry, cherry, Size(Xresize, Yresize));
				resize(cherryTemp, cherryTemp, Size(Xresize, Yresize));
				//show test image
				imshow("voici un canard", canard);
				Mat cherry_gray;
				//step 1 : set pic to grayscale
				cvtColor(cherryTemp, cherry_gray, CV_BGR2GRAY);
				Mat Tgray, Tcolor;
				// step 2 : apply threshold to grayscale pic 
				threshold(cherry_gray, Tgray, 128, 255, THRESH_BINARY);
				threshold(cherry, Tcolor, 128, 255, THRESH_BINARY);
				Mat canny;
				// step 3 : extract edges from threshold picture
				Canny(Tgray, canny, 80, 150);
				vector<vector<Point>> contours;
				vector<Vec4i> hierarchy;
				Mat canny_edge_image_copy = canny.clone();
				//step 4 : extract contours from edge picture
				findContours(canny_edge_image_copy, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

				// check quantity of contours. we just stop at the first picture that haw few contours but not none.
				if (contours.size() < 23 && contours.size() != 0)
					optimalPicFound = 1;

				//work on our chosen optimal version of the image
				if (optimalPicFound == 1)
				{
					vector<vector<Point>> approx_contours(contours.size());
					vector<Rect> boundRect(contours.size());
					Mat conts, rects;
					cherry.copyTo(conts);
					cherry.copyTo(rects);
					// black mat to copy found elements like mole in, later
					//Mat tempResult = Mat::zeros(Yresize, Xresize, cherry.type());
					Mat tempResult = cherry.clone();

					int Nb_elements_found = 0;
					int refined_nb_elements_found = 0;
					//iterate on the contours of our optimal version of the image
					for (int i = 0; i < contours.size(); i++)
					{
						Scalar colour(rand() & 0xFF + 10, rand() & 0xFF + 10, rand() & 0xFF + 10);
						//draw all raw contours found on
						drawContours(conts, contours, i, colour, 3, 1, hierarchy, 0, Point());
						Rect coords;
						coords = boundingRect(contours[i]);
						// will extract (unfinished) all pre selected contours in a separate image
						Mat selectedContour, mask;
						selectedContour = cherry(coords);
						mask = cherry(coords);
						selectedContour.copyTo(tempResult(cv::Rect(coords.x, coords.y, selectedContour.cols, selectedContour.rows)));
						boundRect[i] = boundingRect(Mat(contours[i]));

						int margin = 5;
						//cosmetic : minimum size for a contour to be shown as a contour and not in a circle
						int minShow = 4;
						//decision : minimum size for a contour to be considered a detected element
						int maxSize = 100;
						if (coords.height < maxSize && coords.width < maxSize)
						{
							Nb_elements_found++;
							stringstream strs;
							strs << Nb_elements_found;
							string temp_str = strs.str();
							char* char_type = (char*)temp_str.c_str();

							/*checking closeness of the current contour to others. UNUSED
							This could mean two things :
							1-closeness close to a border of the picture, probably a contrast error
							2- closeness in the rest of the picture : those contours probably should be merged*/
							for (int j = 0; j < contours.size(); j++)
							{
								Rect compareCoords;
								// so compareCoords is going to be all of the contours
								compareCoords = boundingRect(contours[j]);
								//not checking a contour with itself
								if (i != j)
								{
									int boundMargin = 7;
									//if (coords.x == compareCoords.x || coords.y == compareCoords.y)
									if (IsInBounds(coords.x, compareCoords.x - boundMargin, compareCoords.x + boundMargin) && IsInBounds(coords.y, compareCoords.y - boundMargin, compareCoords.y + boundMargin))
									{
										1 == 1;
										//colour = (100,100,100);
									}
								}
							}
							/**/


							/*drawing contours on RECTS image then write raw contour number*/
							//contour big enough : show it directly
							if (coords.height > minShow && coords.width > minShow)
							{
								drawContours(rects, contours, i, colour, 1, 1, hierarchy, 0, Point());
							}
							else//contour too small, show a circle around it
							{
								Point circleCenter;
								circleCenter.x = boundRect[i].x + boundRect[i].height / 2;
								circleCenter.y = boundRect[i].y + boundRect[i].width / 2;
								circle(rects, circleCenter, 5, colour, 1);
							}
							writeText(rects, char_type, coords.y + margin, coords.x + margin, colour, 0.5, 1);
							/**/

							//compute histogram for all pre selected contours and show it on the result image
							// only if the contour is of minimal size
							int minSize = 3;
							if (coords.height > minSize && coords.width > minSize)
							{

								refined_nb_elements_found++;
								cout << "ref->" << refined_nb_elements_found << endl;
								stringstream strs;
								strs << refined_nb_elements_found;
								string temp_str = strs.str();
								char* char_type = (char*)temp_str.c_str();
								del3++;
								//cout << "del3 " << del3 << " printed char " << char_type << " at coords " << coords.y << " " << coords.x << endl;
								//writeText(tempResult, char_type, coords.y + margin, coords.x + margin, colour, 0.5, 1);
								//writeText(tempResult, char_type, 15, refined_nb_elements_found * 20, (255,255,255), 0.5, 1);
								//drawContours(tempResult, contours, i, colour, 1, 1, hierarchy, 0, Point());



								const int* channel_numbers = { 0 };
								float channel_range[] = { 0.0, 255.0 };
								const float* channel_ranges = channel_range;
								int number_bins = 64;
								//3 mats histogram for rgb image usually
								MatND* colour_histogram = new MatND[selectedContour.channels()];
								//image will be split in this vector of mats.
								vector<Mat> colour_channels(selectedContour.channels());
								split(selectedContour, colour_channels);
								// compute histogram for the R,G,B channels of the selected contour
								for (int chan = 0; chan < selectedContour.channels(); chan++)
									calcHist(&(colour_channels[chan]), 1, channel_numbers, Mat(),
										colour_histogram[chan], 1, &number_bins, &channel_ranges);
								//for each channel,B,G,R, of the histogram, we will count the total of "units" of the given color.
								int totalBlue = 0;
								int totalGreen = 0; 
								int totalRed = 0;
								// the next two are for computing the percentage of values that are at 0 in the hist. 
								// many values at zero mean a very flat/continuous image, and may be a false positive.
								int howManyNulls = 0;
								float percentOfNulls;
								float total3cNulls = 0;
								//feat highreds : a counter for the bright reds.
								int totalBrightReds = 0;
								//feat isDark : a counter of dark colors.
								int darkColors = 0;
								// a list of features detected or not per each picture
								bool isDark = false;
								bool isFalsePositive = false;
								bool isBrightRed = false;
								/*show in console value of histogram*/
								cout << "*********** image start, element number" << refined_nb_elements_found << endl;
								delaftuse++;
								for (int channel = 0; (channel < selectedContour.channels()); channel++)
								{
									int maxOfChannel = 0;
									cout << "channel start" << endl;
									float value;
									float counter = 0;
									for (int currentBin = 0; currentBin < number_bins; currentBin++)
									{
										counter++;
										//cout << "uuu";
										value = colour_histogram[channel].at<float>(currentBin);
										//counting dark colors. first half.
										if (currentBin < number_bins / 2)
											darkColors = darkColors + value;
										if (value > maxOfChannel)
											maxOfChannel = value;
										// counting how much units of blue appear in the blue hist
										if (channel == 0)
											totalBlue = totalBlue + value;
										if (channel == 1)
											totalGreen = totalGreen + value;
										if (channel == 2)
											totalRed = totalRed + value;
										//feat highreds. on red channel, second half(bright colors),
										if (channel == 2 && currentBin > number_bins / 2)
											totalBrightReds = totalBrightReds + value;
										//feat %nulls : check how many bins contains null numbers
										if (value == 0)
											howManyNulls++;

										/*if (channel == 1)
											colour_histogram[channel].at<float>(h) = 30;*/
											cout << value << " ";
									}
									cout << "nb values(bins) : " << counter << endl;
									cout << "channel end" << endl;
									percentOfNulls = ((float)howManyNulls / (float)number_bins)*100;
									cout << "quantity of nulls : " << howManyNulls << endl;
									cout << " percentage of nulls : " << percentOfNulls << " %" << endl;
									total3cNulls = total3cNulls + percentOfNulls;
									howManyNulls = 0;
									cout << "biggest value on this channel : " << maxOfChannel << endl;
								}
								
								cout << "units of blue : " << totalBlue << endl;
								cout << "units of green : " << totalGreen << endl;
								cout << "units of red : " << totalRed << endl;
								//feat highreds.
								cout << "_____________________________________________________" << endl;
								cout << "total of units (resolution) of colors in this fragment : " << totalBlue + totalGreen + totalRed << " points of colors." << endl;
								cout << "total of units of bright red : " << totalBrightReds << endl;
								float ratioRtoA = ((float)totalBrightReds / (float)(totalBlue + totalGreen + totalRed)) * 100;
								cout << " bright red to total amount of color ratio : " <<  ratioRtoA << " %." << endl;
								// usually when approx 30% or more of used colors are bright red, it's a cherry angioma. except if it's a false positive.
								if (ratioRtoA > 30)
								{
									cout << "--------------> [ELEMENT DETECTED AS CHERRY ANGIOMA]" << endl;
									isBrightRed = true;
								}
								else
									cout << "--------------> [element not detected as cherry angioma.]" << endl;
								
								cout << "_____________________________________________________" << endl;
								cout << " quantity of dark colors / total color resolution : " << darkColors <<  " / " << totalBlue + totalGreen + totalRed << endl;
								int ratioDtoT = ((float)darkColors / (float)(totalBlue + totalGreen + totalRed)) * 100;
								cout << " ratio dark to total colors : " << ratioDtoT << " %" << endl;
								// dark colors are a dark mole but may be a melanoma
								if (ratioDtoT > 70)
								{
									cout << "--------------> [DARK ELEMENT DETECTED, DARK MOLE, OR MAY BE MELANOMA]" << endl;
									isDark = true;
								}
								else
									cout << "--------------> [not dark. is a mole if no other feats detected.]" << endl;								
								cout << "_____________________________________________________" << endl;
								cout << "average percentage of null-colors : " << total3cNulls / 3 << " %" << endl;
								//generally, result above 90 are flat colored, with few variations. they are 
								//generally false positives.
								cout << "*********** image end *************" << endl;
								if (total3cNulls / 3 > 88)
								{
									cout << "--------------> [ELEMENT DETECTED AS FALSE POSITIVE]" << endl << endl;
									isFalsePositive = true;
								}
								else
									cout << "--------------> [element not detected as false positive.]" << endl ;
								cout << "_____________________________________________________" << endl;
								// not combining some logic on features found, or not :
								cout << "element has close features of ";
								if (isFalsePositive == true)
									cout << "a false positive.";
								else
								{
									if (isBrightRed == true)
										cout << "a cherry angioma.";
									if (isDark == true)
										cout << "a dark mole or a melanoma.";
									if (!isBrightRed && !isDark)
										cout << "a mole.";
								}
								cout << endl;
								cout << "_____________________________________________________" << endl << endl;

								int deca = 15;
								/*draw contours of all refined found contours minus the false positive,
								and write what feature has been detected*/
								//draw if relevant (not false positive)
								if (!isFalsePositive)
								{
									drawContours(tempResult, contours, i, colour, 1, 1, hierarchy, 0, Point());
									// no. refined contour
									//writeText(tempResult, char_type, coords.y + margin, coords.x + margin, colour, 0.5, 1);
									if (isBrightRed == true)
										writeText(tempResult, "Cherry Angioma", coords.y + margin, coords.x + margin + deca, colour, 0.5, 1);
									if (isDark == true)
										writeText(tempResult, "Melanoma", coords.y + margin, coords.x + margin + deca, colour, 0.5, 1);
									if (!isBrightRed && !isDark)
										writeText(tempResult, "Mole", coords.y + margin, coords.x + margin + deca, colour, 0.5, 1);
								}



								/*draw histogram of selected contour. max 8 allowed. */
								// image is reshaped later to the hist's need
								// histograms are created with an auto size of 257*257. 
								Mat color_Hist_of_Selected_Contour = Mat::zeros(Yresize, Xresize, cherry.type());
								Draw1DHistogram(colour_histogram, selectedContour.channels(), color_Hist_of_Selected_Contour);
								writeText(color_Hist_of_Selected_Contour, char_type, coords.y + margin, coords.x + margin, colour, 0.5, 1);
								Mat testt = Mat::zeros(color_Hist_of_Selected_Contour.cols, color_Hist_of_Selected_Contour.rows, color_Hist_of_Selected_Contour.type());
								Rect r20(0, 0, color_Hist_of_Selected_Contour.cols, color_Hist_of_Selected_Contour.rows);//cherry.cols, cherry.rows);
								Mat roi20 = testt(r20);
								color_Hist_of_Selected_Contour.copyTo(roi20);
								int visual = 1;
								del2++;
								//cout << "zz" << del2 << endl;
								if (visual == 1)
									switch (refined_nb_elements_found)
									{
									case 1:
									{
										imshow("1 hist", color_Hist_of_Selected_Contour);
										imshow("1 cont", selectedContour);
										break;
									}
									case 2:
									{
										imshow("2 hist", color_Hist_of_Selected_Contour);
										imshow("2 cont", selectedContour);
										break;
									}
									case 3:
									{
										imshow("3 hist", color_Hist_of_Selected_Contour);
										imshow("3 cont", selectedContour);
										break;
									}
									case 4:
									{
										imshow("4 hist", color_Hist_of_Selected_Contour);
										imshow("4 cont", selectedContour);
										break;
									}
									case 5:
									{
										cout << "aaazerererera" << refined_nb_elements_found;
										imshow("5 hist", color_Hist_of_Selected_Contour);
										imshow("5 cont", selectedContour);
										break;
									}
									case 6:
									{
										imshow("6 hist", color_Hist_of_Selected_Contour);
										imshow("6 cont", selectedContour);
										break;
									}
									case 7:
									{
										imshow("7 hist", color_Hist_of_Selected_Contour);
										imshow("7 cont", selectedContour);
										break;
									}
									case 8:
									{
										imshow("8 hist", color_Hist_of_Selected_Contour);
										imshow("8 cont", selectedContour);
										break;
									}
									}
								/**/
							}
						}
					}

					/* build the first serie of image-steps to result */
					{
						stringstream strs;
						strs << contours.size();
						string temp_str = strs.str();
						char* char_type = (char*)temp_str.c_str();
						//show how many contours are found 
						writeText(conts, char_type, 25, 5, 1, 1);
						//cherryTemp Tgray canny conts rects 218 195
						cvtColor(Tgray, Tgray, CV_GRAY2BGR);
						cvtColor(cherry_gray, cherry_gray, CV_GRAY2BGR);
						cvtColor(canny, canny, CV_GRAY2BGR);
						Mat bigWindow = Mat::zeros(Yresize, Xresize * 5, cherry.type());
						Rect r(0, 0, cherry.cols, cherry.rows);
						Mat roi = bigWindow(r);
						cherry.copyTo(roi);
						Rect r3(Xresize, 0, cherry.cols, cherry.rows);
						Mat roi3 = bigWindow(r3);
						Tgray.copyTo(roi3);
						Rect r4(Xresize * 2, 0, cherry.cols, cherry.rows);
						Mat roi4 = bigWindow(r4);
						canny.copyTo(roi4);
						Rect r5(Xresize * 4, 0, cherry.cols, cherry.rows);
						Mat roi5 = bigWindow(r5);
						rects.copyTo(roi5);
						Rect r7(Xresize * 3, 0, cherry.cols, cherry.rows);
						Mat roi7 = bigWindow(r7);
						conts.copyTo(roi7);
						imshow("     Original                                                   Thres                                                           Edges                                                         Raw Contours                                            Refined ", bigWindow);
					}
					/**/

					/* build the second serie of image-steps to result (big)*/
					{
						int newdimX = 370;
						int newdimY = 340;
						Mat visualResult = rects.clone();
						Mat visualRaw = cherry.clone();
						Mat visualThres = Tgray.clone();
						resize(visualResult, visualResult, Size(newdimX, newdimY));
						resize(visualRaw, visualRaw, Size(newdimX, newdimY));
						resize(tempResult, tempResult, Size(newdimX, newdimY));
						Mat resultsWindow = Mat::zeros(newdimY, newdimX * 2+1, visualRaw.type());
						Rect r8(0, 0, visualRaw.cols, visualRaw.rows);
						Mat roi8 = resultsWindow(r8);
						visualRaw.copyTo(roi8);
						//Rect r9(newdimX + 1, 0, visualRaw.cols, visualRaw.rows);
						//Mat roi9 = resultsWindow(r9);
						//visualResult.copyTo(roi9);
						Rect r10(newdimX +1, 0, visualRaw.cols, visualRaw.rows);
						Mat roi10 = resultsWindow(r10);
						tempResult.copyTo(roi10);
						imshow("examining results", resultsWindow);
					}
					/**/

					//this avoids examining manually cases where there is an obvious contrast problem
					// hundreds of contours found = too much contrast, none = not enough
					if (contours.size() < 23 && contours.size() != 0)
					{
						optimalPicFound = 1;
						waitKey(0);
					}


				}// optimal picture found,
			}
			

		}//end of contrast loop
		 /************** eof contrast then...... ***********************************/

		/************************ should work on histograms phase******************/
		 //should work on hists
		//cout << "should work on hists here... optimal pic should be found, what about the list of optimized contours ?";
		//waitKey(0);
		system("cls");
		destroyAllWindows();
		/************************ eof should work on histograms phase******************/

	}//end of loop on pics

	return(0);
}