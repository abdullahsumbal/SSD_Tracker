//
// Created by sumbal on 20/06/18.
//

#ifndef PROJECT_PIPELINE_H
#define PROJECT_PIPELINE_H

#pragma once

#include "Ctracker.h"
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <opencv/cxmisc.h>
#include "ssd_detect.h"
#include "defines.h"
#include <math.h>
#include <algorithm>

DEFINE_string(mean_file, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
              "If specified, can be one value or can be same as image channels"
              " - would subtract from the corresponding channel). Separated by ','."
              "Either mean_file or mean_value should be provided, not both.");

class Pipeline
{
public:
    explicit Pipeline(const cv::CommandLineParser &parser)
    {
        outFile = parser.get<std::string>("output");
        inFile = parser.get<std::string>(0);
        endFrame = parser.get<int>("end_frame");
        startFrame =  parser.get<int>("start_frame");
        m_fps = 30;
        saveVideo = parser.get<bool>("save_video");
        enableCount = parser.get<bool>("count");
        drawCount = parser.get<bool>("draw_count");
        drawOther = parser.get<bool>("draw_other");
        direction = parser.get<int>("direction");
        useCrop = parser.get<bool>("crop");
        cropFrameWidth = parser.get<int>("crop_width");
        cropFrameHeight = parser.get<int>("crop_height");
        cropRect = cv::Rect(parser.get<int>("crop_x"), parser.get<int>("crop_y"), cropFrameWidth, cropFrameHeight);
        detectThreshold = parser.get<float>("threshold");
        desiredDetect = parser.get<bool>("desired_detect");
        desiredObjectsString = parser.get<std::string>("desired_objects");


        if (!parser.check())
        {
            parser.printErrors();
        }

        // Different color used for path lines in tracking
        // Add more if you are a colorful person.
        m_colors.emplace_back(cv::Scalar(255, 0, 0));
        m_colors.emplace_back(cv::Scalar(0, 255, 0));
        m_colors.emplace_back(cv::Scalar(0, 0, 255));
        m_colors.emplace_back(cv::Scalar(255, 255, 0));
        m_colors.emplace_back(cv::Scalar(0, 255, 255));
        m_colors.emplace_back(cv::Scalar(255, 0, 255));
        m_colors.emplace_back(cv::Scalar(255, 127, 255));
        m_colors.emplace_back(cv::Scalar(127, 0, 255));
        m_colors.emplace_back(cv::Scalar(127, 0, 127));
    }

    void Process(){

        // Prepossessing step. May be make a new function to do prepossessing (TODO)
        // Converting desired object into float.
        std::vector <float> desiredObjects;
        std::stringstream ss(desiredObjectsString);
        while( ss.good() )
        {
            string substring;
            getline( ss, substring, ',' );
            desiredObjects.push_back( std::stof(substring) );
        }

        LOG(INFO) << "Process start" << std::endl;

#ifndef GFLAGS_GFLAGS_H_
        namespace gflags = google;
#endif

        // Set up input
        cv::VideoCapture cap(inFile);
        if (!cap.isOpened()) {
            LOG(FATAL) << "Failed to open video: " << inFile;
        }
        cv::Mat frame;
        int frameCount = 0;

        // video output
        cv::VideoWriter writer;
        auto frame_width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
        auto frame_height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
        if(useCrop)
        {
            writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), m_fps, cv::Size(cropFrameWidth, cropFrameHeight), true);
        }
        else
        {
            writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), m_fps, cv::Size(frame_width, frame_height), true);
        }

        std::map <string,  int> countObjects_LefttoRight;
        std::map <string,  int> countObjects_RighttoLeft;
        double fontScale = CalculateRelativeSize(frame_width, frame_height);

        double tFrameModification = 0;
        double tDetection = 0;
        double tTracking = 0;
        double tCounting = 0;
        double tDTC = 0;
        double tStart  = cv::getTickCount();

        // Process one frame at a time
        while (true) {

            double tStartFrameModification = cv::getTickCount();
            bool success = cap.read(frame);
            if (!success) {
                LOG(INFO) << "Process " << frameCount << " frames from " << inFile;
                break;
            }
            if(frameCount < startFrame)
            {
                continue;
            }

            if (frameCount > endFrame)
            {
                std::cout << "Process: reached last " << endFrame << " frame" << std::endl;
                break;
            }
            CHECK(!frame.empty()) << "Error when read frame";

            // Focus on interested area in the frame
            if (useCrop)
            {
                cv::Mat copyFrame(frame, cropRect);
                // Deep copy (TODO)
                //copyFrame.copyTo(frame);
                // Shallow copy
                frame = copyFrame;
            }
            tFrameModification += cv::getTickCount() - tStartFrameModification;

            // Get all the detected objects.
            double tStartDetection = cv::getTickCount();
            regions_t tmpRegions;
            std::vector<vector<float> > detections = detectframe(frame);

            // Filter out all the objects based
            // 1. Threshold
            // 2. Desired object classe
            for (auto const& detection : detections){
                const vector<float> &d = detection;
                // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                CHECK_EQ(d.size(), 7);
                const float score = d[2];
                const float fLabel= d[1];
                if(desiredDetect)
                {
                    if (!(std::find(desiredObjects.begin(), desiredObjects.end(), fLabel) != desiredObjects.end()))
                    {
                        continue;
                    }
                }
                std::string label = std::to_string(static_cast<int>(fLabel));
                if (score >= detectThreshold) {

                    auto xLeftBottom = static_cast<int>(d[3] * frame.cols);
                    auto yLeftBottom = static_cast<int>(d[4] * frame.rows);
                    auto xRightTop = static_cast<int>(d[5] * frame.cols);
                    auto yRightTop = static_cast<int>(d[6] * frame.rows);
                    cv::Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
                    tmpRegions.push_back(CRegion(object, label, score));
                }
                //cv::imshow("Video", frame);
            }
            tDetection += cv::getTickCount() - tStartDetection;

            double tStartTracking = cv::getTickCount();
            // Update Tracker
            cv::UMat clFrame;
            clFrame = frame.getUMat(cv::ACCESS_READ);
            m_tracker->Update(tmpRegions, clFrame, m_fps);
            tTracking += cv::getTickCount() - tStartTracking;

            if(enableCount)
            {
                double tStartCounting = cv::getTickCount();
                // Update Counter
                CounterUpdater(frame, countObjects_LefttoRight, countObjects_RighttoLeft);
                tCounting += cv::getTickCount() - tStartCounting;

                if(drawCount){
                    DrawCounter(frame, fontScale, countObjects_LefttoRight, countObjects_RighttoLeft);
                }

            }

            if(drawOther){
                DrawData(frame, frameCount, fontScale);
            }

            if (writer.isOpened() and saveVideo)
            {
                writer << frame;
            }
            ++frameCount;
        }
        if (cap.isOpened()) {
            cap.release();
        }

        // Calculate Time for components
        double tEnd  = cv::getTickCount();
        double totalRunTime = (tEnd - tStart)/cv::getTickFrequency();
        double tFrameModificationRuntTime = tFrameModification/cv::getTickFrequency();
        double detectionRunTime = tDetection/cv::getTickFrequency();
        double trackingRunTime = tTracking/cv::getTickFrequency();
        double countingRunTime = tCounting/cv::getTickFrequency();
        double FDTCRuntime = tFrameModificationRuntTime + detectionRunTime + trackingRunTime + countingRunTime;

        // Display and write output
        std::ofstream csvFile;
        csvFile.open ("../data/D2.csv");
        csvFile << "Frame Modification time" << ",";
        csvFile << "Detection time" << ",";
        csvFile << "Tracking time" << ",";
        csvFile << "Counting time" << ",";
        csvFile << "FDTC time" << ",";
        csvFile << "Total time" << ",";
        csvFile << "FDTC frame rate" << ",";
        csvFile << "Total frame rate" << "\n";
        LOG(INFO)  << "Frame Modification time = " << tFrameModificationRuntTime << " seconds" << std::endl;
        csvFile << tFrameModificationRuntTime << ",";
        LOG(INFO)  << "Detection time = " << detectionRunTime << " seconds" << std::endl;
        csvFile << detectionRunTime << ",";
        LOG(INFO)  << "Tracking time = " << trackingRunTime << " seconds" << std::endl;
        csvFile << trackingRunTime<< ",";
        LOG(INFO)  << "Counting time = " << countingRunTime << " seconds" << std::endl;
        csvFile << countingRunTime << ",";
        LOG(INFO)  << "FDTC time = " << FDTCRuntime << " seconds " << std::endl;
        csvFile << FDTCRuntime << ",";
        LOG(INFO)  << "Total time = " << totalRunTime << " seconds " << std::endl;
        csvFile << totalRunTime << ",";
        LOG(INFO)  << " FDTC frame rate: "<< frameCount/FDTCRuntime << " fps" <<std::endl;
        csvFile << frameCount/FDTCRuntime  << ",";
        LOG(INFO)  << " Total frame rate: "<< frameCount/totalRunTime << " fps" << std::endl;
        csvFile << frameCount/totalRunTime << "\n";
        LOG(INFO)  << "Left to Right or Top to Bottom ";
        csvFile << "Object label" << "," << "count Left to Right" << "\n";
        for(auto elem : countObjects_LefttoRight)
        {
            LOG(INFO) << elem.first << " " << elem.second << "\n";
            csvFile << elem.first << "," << elem.second << "\n";
        }
        LOG(INFO)  << "Right to Left or Bottom to Top";
        csvFile << "Object label" << "," << "count Right to Left" << "\n";
        for(auto elem : countObjects_RighttoLeft)
        {
            LOG(INFO) << elem.first << " " << elem.second << "\n";
            csvFile << elem.first << "," << elem.second << "\n";
        }

        csvFile.close();
    }
protected:
    std::unique_ptr<CTracker> m_tracker;
    float m_fps;
    bool enableCount;
    int direction;

    virtual std::vector<vector<float> > detectframe(cv::Mat frame)= 0;
    virtual void DrawData(cv::Mat frame, int framesCounter, double fontScale) = 0;
    virtual void CounterUpdater(cv::Mat frame, std::map <string,  int> &countObjects_LefttoRight, std::map <string,  int> &countObjects_RighttoLeft) = 0;
    virtual void DrawCounter(cv::Mat frame, double fontScale, std::map <string,  int> &countObjects_LefttoRight, std::map <string,  int> &countObjects_RighttoLeft) = 0;

    void DrawTrack(cv::Mat frame,
                   int resizeCoeff,
                   const CTrack& track,
                   bool drawTrajectory = true,
                   bool isStatic = false
    )
    {
        auto ResizeRect = [&](const cv::Rect& r) -> cv::Rect
        {
            return cv::Rect(resizeCoeff * r.x, resizeCoeff * r.y, resizeCoeff * r.width, resizeCoeff * r.height);
        };
        auto ResizePoint = [&](const cv::Point& pt) -> cv::Point
        {
            return cv::Point(resizeCoeff * pt.x, resizeCoeff * pt.y);
        };

        if (isStatic)
        {
            cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(255, 0, 255), 2, CV_AA);
        }
        else
        {
            cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(0, 255, 0), 1, CV_AA);
        }

        if (drawTrajectory)
        {
            cv::Scalar cl = m_colors[track.m_trackID % m_colors.size()];

            for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
            {
                const TrajectoryPoint& pt1 = track.m_trace.at(j);
                const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);

                cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, CV_AA);
                if (!pt2.m_hasRaw)
                {
                    //cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, CV_AA);
                }
            }
        }
    }

    double CalculateRelativeSize(int frame_width, int frame_height)
    {
        int baseLine = 0;
        double countBoxWidth = frame_width * 0.1;
        double countBoxHeight = frame_height * 0.1;
        cv::Rect countBoxRec(0, 200, int(countBoxWidth), int(countBoxHeight));
        std::string counterLabel_Left = "Count : " + std::to_string(0);
        cv::Size rect = cv::getTextSize(counterLabel_Left, cv::FONT_HERSHEY_PLAIN, 1.0, 1, &baseLine);
        double scalex = (double)countBoxRec.width / (double)rect.width;
        double scaley = (double)countBoxRec.height / (double)rect.height;
        return std::min(scalex, scaley);
    }
private:
    bool saveVideo;
    bool drawCount;
    bool drawOther;
    bool useCrop;
    int endFrame;
    int startFrame;
    cv::Rect cropRect;
    bool desiredDetect;
    int cropFrameWidth;
    std::string inFile;
    std::string outFile;
    int cropFrameHeight;
    float detectThreshold;
    std::vector<cv::Scalar> m_colors;
    std::string desiredObjectsString;

};

class SSDExample : public Pipeline{
public:
    explicit SSDExample(const cv::CommandLineParser &parser) : Pipeline(parser){
        meanFile = FLAGS_mean_file;
        meanValue = FLAGS_mean_value;
        line1_x1 = parser.get<int>("l1p1_x");
        line1_x2 = parser.get<int>("l1p2_x");
        line1_y1 = parser.get<int>("l1p1_y");
        line1_y2 = parser.get<int>("l1p2_y");
        line2_x1 = parser.get<int>("l2p1_x");
        line2_x2 = parser.get<int>("l2p2_x");
        line2_y1 = parser.get<int>("l2p1_y");
        line2_y2 = parser.get<int>("l2p2_y");
        modelFile = parser.get<std::string>("model");
        weightsFile = parser.get<std::string>("weight");

        // Initialize the Detector
        detector.initDetection(modelFile, weightsFile, meanFile, meanValue);

        // Initialize the tracker
        config_t config;

        // TODO: put these variables in main
        TrackerSettings settings;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 100;                          // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = (size_t)(1 * m_fps);  // Maximum allowed skipped frames
        settings.m_maxTraceLength = (size_t)(5 * m_fps);               // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);
    }
private:
    std::string modelFile;
    std::string weightsFile;
    std::string meanFile;
    std::string meanValue;
    Detector detector;
    int line1_x1;
    int line1_x2;
    int line1_y1;
    int line1_y2;
    int line2_x1;
    int line2_x2;
    int line2_y1;
    int line2_y2;
protected:
    std::vector<vector<float> > detectframe(cv::Mat frame){
        return detector.Detect(frame);
    }
    void DrawData(cv::Mat frame, int framesCounter, double fontScale){
        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                DrawTrack(frame, 1, *track);
                std::string objectName;
                if (track->m_lastRegion.m_type == "2.0"){
                    objectName = "Bicycle";
                }
                if (track->m_lastRegion.m_type == "15.0"){
                    objectName = "People";
                }

                std::string label = objectName + ": " + std::to_string((track->m_lastRegion.m_confidence * 10000)/100) + " %";
                //std::string label = std::to_string(track->m_trace.m_firstPass) + " | " + std::to_string(track->m_trace.m_secondPass);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                auto rect(track->GetLastRect());
                cv::rectangle(frame, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

    }

    void DrawCounter(cv::Mat frame, double fontScale, std::map <string,  int> &countObjects_LefttoRight, std::map <string,  int> &countObjects_RighttoLeft){

        // Line
        cv::line( frame, cv::Point( line2_x1, line2_y1 ), cv::Point( line2_x2, line2_y2), cv::Scalar( 120, 220, 0 ),  2, 8 );

        // Counter label
        std::string counterLabel_L;
        std::string counterLabel_R;
        for(auto elem : countObjects_LefttoRight){
            counterLabel_L += elem.first + ": " + std::to_string(elem.second) + "|";
        }
        for(auto elem : countObjects_RighttoLeft){
            counterLabel_R += elem.first + ": " + std::to_string(elem.second) + "|";
        }


        cv::putText(frame, counterLabel_L, cv::Point(0, (int)(frame.size().width * 0.1)), cv::FONT_HERSHEY_PLAIN, fontScale, cv::Scalar(255, 255, 255), 1, 8);
        cv::putText(frame, counterLabel_R, cv::Point(0, (int)(frame.size().width * 0.2)), cv::FONT_HERSHEY_PLAIN, fontScale, cv::Scalar(255, 255, 255), 1, 8);
    }

    void CounterUpdater(cv::Mat frame, std::map <string,  int> &countObjects_LefttoRight, std::map <string,  int> &countObjects_RighttoLeft)
    {

        for (const auto& track : m_tracker->tracks)
        {
            if(track->m_trace.size() >= 2)
            {
                track_t pt1_x = track->m_trace.at(track->m_trace.size() - 2).m_prediction.x;
                track_t pt1_y = track->m_trace.at(track->m_trace.size() - 2).m_prediction.y;
                track_t pt2_x = track->m_trace.at(track->m_trace.size() - 1).m_prediction.x;
                track_t pt2_y = track->m_trace.at(track->m_trace.size() - 1).m_prediction.y;

                float pt1_position_line1 = (line1_y2 - line1_y1) * pt1_x + (line1_x1 - line1_x2) * pt1_y + (line1_x2 * line1_y1 - line1_x1 * line1_y2);
                float pt2_position_line1 = (line1_y2 - line1_y1) * pt2_x + (line1_x1 - line1_x2) * pt2_y + (line1_x2 * line1_y1 - line1_x1 * line1_y2);
                float pt1_position_line2 = (line2_y2 - line2_y1) * pt1_x + (line2_x1 - line2_x2) * pt1_y + (line2_x2 * line2_y1 - line2_x1 * line2_y2);
                float pt2_position_line2 = (line2_y2 - line2_y1) * pt2_x + (line2_x1 - line2_x2) * pt2_y + (line2_x2 * line2_y1 - line2_x1 * line2_y2);

                if(direction == 0)
                {
                    if(pt1_position_line1 < 0  && pt2_position_line1 >= 0)
                    {
                        track->m_trace.FirstPass();
                    }
                    if (track->m_trace.GetFirstPass() && pt2_position_line2 >= 0 && !track->m_trace.GetSecondPass() )
                    {
                        track->m_trace.SecondPass();
                        std::pair<std::map<string, int>::iterator,bool> ret;
                        ret = countObjects_LefttoRight.insert ( std::pair<string, int>(track->m_lastRegion.m_type, 1));
                        if (!ret.second) {
                            ret.first->second = ret.first->second + 1;
                        }
                    }
                }else if (direction == 1)
                {
                    if(pt2_position_line2 <= 0  && pt1_position_line2 > 0)
                    {
                        track->m_trace.FirstPass();
                    }
                    if (track->m_trace.GetFirstPass() && pt2_position_line1 <= 0 && !track->m_trace.GetSecondPass() )
                    {
                        track->m_trace.SecondPass();
                        std::pair<std::map<string, int>::iterator,bool> ret;
                        ret = countObjects_RighttoLeft.insert ( std::pair<string, int>(track->m_lastRegion.m_type, 1));
                        if (!ret.second) {
                            ret.first->second = ret.first->second + 1;
                        }
                    }
                }else
                {
                    if(pt2_position_line2 <= 0  && pt1_position_line2 > 0){
                        track->m_trace.FirstPass();
                        track->m_trace.m_directionFromLeft = true;
                    }
                    else if(pt1_position_line1 < 0  && pt2_position_line1 >= 0)
                    {
                        track->m_trace.FirstPass();
                        track->m_trace.m_directionFromLeft = false;
                    }
                    if (track->m_trace.GetFirstPass() && pt2_position_line1 <= 0 && !track->m_trace.GetSecondPass() && track->m_trace.m_directionFromLeft)
                    {
                        track->m_trace.SecondPass();
                        std::pair<std::map<string, int>::iterator,bool> ret;
                        ret = countObjects_RighttoLeft.insert ( std::pair<string, int>(track->m_lastRegion.m_type, 1));
                        if (!ret.second) {
                            ret.first->second = ret.first->second + 1;
                        }
                    }
                    else if (track->m_trace.GetFirstPass() && pt2_position_line2 >= 0 && !track->m_trace.GetSecondPass() && !track->m_trace.m_directionFromLeft)
                    {
                        track->m_trace.SecondPass();
                        std::pair<std::map<string, int>::iterator,bool> ret;
                        ret =  countObjects_LefttoRight.insert ( std::pair<string, int>(track->m_lastRegion.m_type, 1));
                        if (!ret.second) {
                            ret.first->second = ret.first->second + 1;
                        }
                    }
                }
            }
        }
    }
};

#endif //PROJECT_PIPELINE_H