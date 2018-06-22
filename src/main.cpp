#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <caffe/caffe.hpp>
#include "Pipeline.h"
// ----------------------------------------------------------------------

static void Help()
{
    printf(
           "Usage: \n"
           "          ./SSD <path to movie file> [--example]=<number of example 0..5> [--start_frame]=<start a video from this position> [--end_frame]=<play a video to this position> [--end_delay]=<delay in milliseconds after video ending> [--out]=<name of result video file> [--show_logs]=<show logs> \n\n"
           "Press:\n"
           "\'m\' key for change mode: play|pause. When video is paused you can press any key for get next frame. \n\n"
           "Press Esc to exit from video \n\n"
    );
}

const char* keys =
        {
                "{ @1             |../data/v1.mp4      | input video file | }"
                "{ e  example     |0                   | number of example | 0 - SSD }"
                "{ sf start_frame |0                   | Start a video from this position | }"
                "{ ef end_frame   |400                   | Play a video to this position (if 0 then played to the end of file) | }"
                "{ ed end_delay   |0                   | Delay in milliseconds after video ending | }"
                "{ o  output      |../data/o1.avi                    | Name of result video file | }"
                "{ sl show_logs   |1                   | Show Trackers logs | }"
                "{ m model        |../models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt | Model file for detection | }"
                "{ w weight       |../models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel  | Weight file for detection | }"
                "{ th threshold   |0.1                   |  | }"
                "{ ocl opencl     |0                   |  | }"
        };

// ----------------------------------------------------------------------

int main(int argc, char** argv)
{
    Help();

    cv::CommandLineParser parser(argc, argv, keys);

    // Log set up
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

    bool useOCL = parser.get<int>("opencl") ? 1 : 0;
    cv::ocl::setUseOpenCL(useOCL);
    std::cout << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;

    int exampleNum = parser.get<int>("example");

    switch (exampleNum)
    {
        case 0:
        {
            SSDExample ssdExample(parser);
            ssdExample.Process();
            break;
        }
        default:
            std::cerr << "Wrong example number!" << std::endl;
            break;
    }


    cv::destroyAllWindows();
    return 0;
}
