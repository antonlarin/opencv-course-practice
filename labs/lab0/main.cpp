#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


cv::Mat canny(cv::Mat source)
{
    const int CANNY_LOWER_THRESHOLD = 110;
    const int CANNY_UPPER_THRESHOLD = 220;

    cv::Mat grayscaleSource;
    cv::cvtColor(source, grayscaleSource, CV_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(grayscaleSource, edges, CANNY_LOWER_THRESHOLD,
            CANNY_UPPER_THRESHOLD);
    cv::cvtColor(edges, edges, CV_GRAY2BGR);

    int w = source.size().width;
    int h = source.size().height;
    cv::Mat result;
    result.create(h * 2, w, CV_8UC3);
    cv::Mat before(result, cv::Rect(0, 0, w, h));
    cv::Mat after(result, cv::Rect(0, h, w, h));
    source.copyTo(before);
    edges.copyTo(after);

    return result;
}

cv::Mat eqhist(cv::Mat source)
{
    std::vector<cv::Mat> sourceChannels;
    cv::split(source, sourceChannels);
    std::vector<cv::Mat> equalizedChannels(sourceChannels.size());
    for (int i = 0; i < sourceChannels.size(); i++)
    {
        cv::equalizeHist(sourceChannels[i], equalizedChannels[i]);
    }
    cv::Mat equalizedImage(source.size(), source.type());
    cv::merge(equalizedChannels, equalizedImage);

    int w = source.size().width;
    int h = source.size().height;
    cv::Mat result;
    result.create(h * 2, w, CV_8UC3);
    cv::Mat before(result, cv::Rect(0, 0, w, h));
    cv::Mat after(result, cv::Rect(0, h, w, h));
    source.copyTo(before);
    equalizedImage.copyTo(after);

    return result;
}

cv::Mat goodftrs(cv::Mat source)
{
    const int MAX_FEATURES = 50;
    const double MIN_DIST = 10;
    const double QUALITY = 0.15;

    cv::Mat grayscaleSource;
    cv::cvtColor(source, grayscaleSource, CV_BGR2GRAY);

    std::vector<cv::Point> features;
    cv::goodFeaturesToTrack(grayscaleSource, features,
            MAX_FEATURES, QUALITY, MIN_DIST);
    
    cv::Mat result = source.clone();
    cv::Scalar red(0, 0, 255);
    for (auto f = features.begin(); f != features.end(); f++)
    {
        cv::circle(result, *f, 4, red, 2);
    }

    return result;
}


int main(int argc, char** argv)
{
    std::string function;
    std::string filename;
    if (argc < 3) {
        std::cout << "Usage: ocvtest <function> <image>\n" <<
            "where <function> is one of:\n" <<
            "\tcanny - Canny edge detector;\n" <<
            "\teqhist - histogram equalizer;\n" <<
            "\tgoodftrs - good features for tracking.\n";
        return 1;
    }
    else
    {
        function = argv[1];
        filename = argv[2];
    }

    cv::Mat source;
    source = cv::imread(filename);
    if (!source.data)
    {
        std::cout << "Could not find or open the image.\n";
        return 1;
    }

    cv::Mat result;
    if (function == "canny")
    {
        result = canny(source);
    }
    else if (function == "eqhist")
    {
        result = eqhist(source);
    }
    else if (function == "goodftrs")
    {
        result = goodftrs(source);
    }
    else
    {
        std::cout << "Unsuppported function. Run 'ocvtest' for list of supported functions.\n";
        return 1;
    }

    std::string windowTitle = "OpenCV test - " + function;
    cv::namedWindow(windowTitle);
    cv::imshow(windowTitle, result);
    cv::waitKey(0);

    return 0;
}
