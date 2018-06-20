//
// Created by sumbal on 20/06/18.
//

#ifndef PROJECT_PIPELINE_H
#define PROJECT_PIPELINE_H

#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <opencv/cxmisc.h>
#include "ssd_detect.h"

DEFINE_string(mean_file, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
              "If specified, can be one value or can be same as image channels"
              " - would subtract from the corresponding channel). Separated by ','."
              "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
              "The file type in the list_file. Currently support image and video.");

class Pipeline
{
public:
    Pipeline(const cv::CommandLineParser &parser)
    {

    }

    void Process(){

        LOG(INFO) << "Process start" << std::endl;
        startDetector();

#ifndef GFLAGS_GFLAGS_H_
        namespace gflags = google;
#endif

    }

protected:
    virtual bool startDetector() = 0;
private:

};

class SSDExample : public Pipeline{
public:
    SSDExample(const cv::CommandLineParser &parser) : Pipeline(parser){
        model_file = parser.get<std::string>("model");
        weights_file = parser.get<std::string>("weight");
        file_type = FLAGS_file_type;
        mean_value = FLAGS_mean_value;
        mean_file = FLAGS_mean_file;
        confidence_threshold = parser.get<int>("threshold");
    }
private:
    std::string model_file;
    std::string weights_file;
    std::string mean_file;
    std::string mean_value;
    std::string file_type;
    std::string out_file;
    float confidence_threshold;

    bool startDetector(){
        LOG(INFO) << "starting detector" << std::endl;
    }

};

#endif //PROJECT_PIPELINE_H