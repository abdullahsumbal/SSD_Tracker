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
DEFINE_string(file_type, "video",
              "The file type in the list_file. Currently support image and video.");

class Pipeline
{
public:
    Pipeline(const cv::CommandLineParser &parser)
    {
        outFile = parser.get<std::string>("output");
        inFile = parser.get<std::string>(0);
    }

    void Process(){

        LOG(INFO) << "Process start" << std::endl;

#ifndef GFLAGS_GFLAGS_H_
        namespace gflags = google;
#endif
        // Set the output mode.
        std::streambuf* buf = std::cout.rdbuf();
        std::ofstream outfile;
        if (!outFile.empty()) {
            outfile.open(outFile.c_str());
            if (outfile.good()) {
                buf = outfile.rdbuf();
            }
        }
        std::ostream out(buf);

        // Set up input
        cv::VideoCapture cap(inFile);
        if (!cap.isOpened()) {
            LOG(FATAL) << "Failed to open video: " << inFile;
        }
        cv::Mat frame;
        int frame_count = 0;

        double tStart  = cv::getTickCount();

        while (true) {
            bool success = cap.read(frame);
            if (!success) {
                LOG(INFO) << "Process " << frame_count << " frames from " << inFile;
                break;
            }
            CHECK(!frame.empty()) << "Error when read frame";
            std::vector<vector<float> > detections = detectframe(frame);
            ++frame_count;
        }

        double tEnd  = cv::getTickCount();
        double runTime = (tEnd - tStart)/cv::getTickFrequency();
        LOG(INFO)  << "work time = " << runTime << " | Frame rate: "<< frame_count/runTime << std::endl;
        if (cap.isOpened()) {
            cap.release();
        }
    }
protected:
    virtual std::vector<vector<float> > detectframe(cv::Mat frame)= 0;
private:
    std::string outFile;
    std::string inFile;

};

class SSDExample : public Pipeline{
public:
    SSDExample(const cv::CommandLineParser &parser) : Pipeline(parser){
        modelFile = parser.get<std::string>("model");
        weightsFile = parser.get<std::string>("weight");
        fileType = FLAGS_file_type;
        meanValue = FLAGS_mean_value;
        meanFile = FLAGS_mean_file;
        confidenceThreshold = parser.get<int>("threshold");
        // Initialize the network.
        detector.initDetection(modelFile, weightsFile, meanFile, meanValue);
    }
private:
    std::string modelFile;
    std::string weightsFile;
    std::string meanFile;
    std::string meanValue;
    std::string fileType;
    float confidenceThreshold;
    Detector detector;
protected:
    std::vector<vector<float> > detectframe(cv::Mat frame){
        return detector.Detect(frame);
    }
};

#endif //PROJECT_PIPELINE_H