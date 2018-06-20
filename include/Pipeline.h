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


#endif //PROJECT_PIPELINE_H

class Pipeline
{
public:
    Pipeline(const cv::CommandLineParser &parser){

    }

    void Process(){
        std::cout << "Starting Process" << std::endl;
    }
protected:
private:
};

class SSDExample : public Pipeline{
public:
    SSDExample(const cv::CommandLineParser &parser) : Pipeline(parser){

    }
private:
};