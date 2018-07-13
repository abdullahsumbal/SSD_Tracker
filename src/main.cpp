#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <caffe/caffe.hpp>
#include "Pipeline.h"
// ----------------------------------------------------------------------

const char* keys =
        {
                "{help h usage ?  |                    | Print usage| }"
                "{ @input_video   |../data/v1.mp4      | Input video file | }"
                "{ e  example     |0                   | Number of example: 0 - SSD }"
                "{ ocl opencl     |1                   | use opencl | }"

                "{ sf start_frame |0                   | Frame modification parameter: Start a video from this position | }"
                "{ ef end_frame   |100000              | Frame modification parameter: Play a video to this position (if 0 then played to the end of file) | }"
                "{ crop           |1                   | Frame modification parameter: use location of interest | }"
                "{ crop_x         |600                   | Frame modification parameter: x coordinate of location of interest | }"
                "{ crop_y         |350                   | Frame modification parameter: y coordinate of location of interest | }"
                "{ crop_width     |600                 | Frame modification parameter: width of location of interest | }"
                "{ crop_height    |350                 | Frame modification parameter: height of location of interest | }"

                "{ m model        |../models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt | Detection parameter: Model file | }"
                "{ w weight       |../models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel  | Detection parameter: Weight file | }"
                "{ lm label_map   |../models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel  | Detection parameter: Label map  file  | }"
                "{ th threshold   |0.1                 | Detection parameter: Confidence percentage of detected objects must exceed this value to be reported as a detected object. | }"

                "{ co count       |1                   | Counting parameter: use counting  | }"
                "{ d direction    |1                   | Counting parameter: Variable to allow counting in a certain direction. 0 - left to right, 1 - right to left, 2 - both | }"
                "{ o  output      |../data/o1.avi      | Writing parameter: Name of output video file | }"
        };

// ----------------------------------------------------------------------

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Log set up
    ::google::InitGoogleLogging(argv[0]);
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = true;

    bool useOCL = parser.get<int>("opencl") != 0;
    cv::ocl::setUseOpenCL(useOCL);
    LOG(INFO) << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;

    auto exampleNum = parser.get<int>("example");

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
