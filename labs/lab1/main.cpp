#include <iostream>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat adaptiveFiltering(cv::Mat input)
{
    static const int maxKernelSize = 105;
    static const int cannyLowerThreshold = 110;
    static const int cannyUpperThreshold = 220;

    cv::Mat grayscaleInput;;
    cv::cvtColor(input, grayscaleInput, CV_BGR2GRAY);

    cv::Mat edges;
    edges.create(input.size(), input.type());
    cv::Canny(grayscaleInput, edges, cannyLowerThreshold, cannyUpperThreshold);

    cv::Mat invertedEdges = cv::Scalar::all(255) - edges;
    cv::Mat distanceMap;
    cv::distanceTransform(invertedEdges, distanceMap, CV_DIST_L2,
        CV_DIST_MASK_PRECISE);
    double minDistance, maxDistance;
    cv::minMaxLoc(distanceMap, &minDistance, &maxDistance);

    std::vector<cv::Mat> channels;
    cv::split(input, channels);
    for (auto ch = channels.begin(); ch != channels.end(); ch++)
    {
        int paddingSize = maxKernelSize / 2;
        cv::Mat paddedChannel;
        paddedChannel.create(
            ch->size() + cv::Size(2 * paddingSize, 2 * paddingSize),
            CV_8U
        );
        cv::copyMakeBorder(
            *ch, paddedChannel, paddingSize,
            paddingSize, paddingSize, paddingSize,
            cv::BORDER_REPLICATE
        );

        cv::Mat integral;
        integral.create(paddedChannel.size() + cv::Size(1, 1), CV_32S);
        cv::integral(paddedChannel, integral, CV_32S);

        for (int j = 0; j < ch->size().height; j++)
        {
            int padJ = j + paddingSize;
            for (int i = 0; i < ch->size().width; i++)
            {
                int padI = i + paddingSize;
                int kernelSize = static_cast<int>(
                    ceil(distanceMap.at<float>(j, i) * maxKernelSize / maxDistance)
                );
                kernelSize += (kernelSize % 2 == 0) ? 1 : 0;
                int halfKernelSize = kernelSize / 2;
                int kernelNorm = kernelSize * kernelSize;
                int convolutionValue =
                    integral.at<int>(padJ + halfKernelSize + 1, padI + halfKernelSize + 1) -
                    integral.at<int>(padJ - halfKernelSize, padI + halfKernelSize + 1) -
                    integral.at<int>(padJ + halfKernelSize + 1, padI - halfKernelSize) +
                    integral.at<int>(padJ - halfKernelSize, padI - halfKernelSize);

                typedef unsigned char uchar;
                ch->at<uchar>(j, i) = convolutionValue / kernelNorm;
            }
        }
                
    }
    cv::Mat result;
    result.create(input.size(), input.type());
    cv::merge(channels, result);

    return result;
}

int main(int argc, char** argv)
{
    std::string filename;
    if (argc < 2)
    {
        std::cout << "Usage: af <image_filename>" << std::endl;
        return 1;
    }
    else
    {
        filename = argv[1];
    }

    cv::Mat source;
    source = cv::imread(filename);

    if (!source.data)
    {
        std::cout << "Could not find or open the image." << std::endl;
        return 1;
    }

    cv::namedWindow("Adaptive filtering");
    cv::Mat result = adaptiveFiltering(source);
    cv::imshow("Adaptive filtering", result);
    cv::waitKey(0);

    return 0;
}
