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
        endFrame = parser.get<int>("end_frame");
        startFrame =  parser.get<int>("start_frame");
        m_fps = 30;
        enableCount = parser.get<bool>("count");
        direction = parser.get<int>("direction");
        counter = 0;

        m_colors.push_back(cv::Scalar(255, 0, 0));
        m_colors.push_back(cv::Scalar(0, 255, 0));
        m_colors.push_back(cv::Scalar(0, 0, 255));
        m_colors.push_back(cv::Scalar(255, 255, 0));
        m_colors.push_back(cv::Scalar(0, 255, 255));
        m_colors.push_back(cv::Scalar(255, 0, 255));
        m_colors.push_back(cv::Scalar(255, 127, 255));
        m_colors.push_back(cv::Scalar(127, 0, 255));
        m_colors.push_back(cv::Scalar(127, 0, 127));
    }

    void Process(){

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

        double tStart  = cv::getTickCount();

        while (true) {
            bool success = cap.read(frame);
            if (!success) {
                LOG(INFO) << "Process " << frameCount << " frames from " << inFile;
                break;
            }
            if (frameCount > endFrame)
            {
                std::cout << "Process: reached last " << endFrame << " frame" << std::endl;
                break;
            }
            CHECK(!frame.empty()) << "Error when read frame";
            cv::Mat copyFrame(frame, cv::Rect(600, 350, 600, 350));

            std::vector<vector<float> > detections = detectframe(copyFrame);

            regions_t tmpRegions;
            for (int i = 0; i < detections.size(); ++i) {
                const vector<float> &d = detections[i];
                // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                CHECK_EQ(d.size(), 7);
                const float score = d[2];
                std::string label = std::to_string(d[1]);
                  if (score >= 0.5) {
                      int xLeftBottom = d[3] * copyFrame.cols;
                      int yLeftBottom = d[4] * copyFrame.rows;
                      int xRightTop = d[5] * copyFrame.cols;
                      int yRightTop = d[6] * copyFrame.rows;
                      cv::Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
                      tmpRegions.push_back(CRegion(object, label, score));
                  }

                //cv::imshow("Video", frame);
            }

            // Update Tracker
            cv::UMat clFrame;
            clFrame = copyFrame.getUMat(cv::ACCESS_READ);
            m_tracker->Update(tmpRegions, clFrame, m_fps);

            DrawData(copyFrame, frameCount);

            if (!writer.isOpened())
            {
                writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), m_fps, copyFrame.size(), true);
            }
            if (writer.isOpened())
            {
                writer << copyFrame;
            }
            ++frameCount;
        }

        double tEnd  = cv::getTickCount();
        double runTime = (tEnd - tStart)/cv::getTickFrequency();
        LOG(INFO)  << "Total time = " << runTime << " seconds | Frame rate: "<< frameCount/runTime << " fps" <<std::endl;
        if (cap.isOpened()) {
            cap.release();
        }
    }
protected:
    std::unique_ptr<CTracker> m_tracker;
    float m_fps;
    bool enableCount;
    int direction;
    int counter;
    virtual std::vector<vector<float> > detectframe(cv::Mat frame)= 0;
    virtual void DrawData(cv::Mat frame, int framesCounter) = 0;

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
            cv::rectangle(frame, ResizeRect(track.GetLastRect()), cv::Scalar(0, 255, 0), 0.5, CV_AA);
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

    void UpdateCount(cv::Mat frame, CTrack& track){
            if(track.m_trace.size() >= 2)
            {
                const int pt1_x = track.m_trace.at(track.m_trace.size() - 2).m_prediction.x;
                const int pt1_y = track.m_trace.at(track.m_trace.size() - 2).m_prediction.y;
                const int pt2_x = track.m_trace.at(track.m_trace.size() - 1).m_prediction.x;
                const int pt2_y = track.m_trace.at(track.m_trace.size() - 1).m_prediction.y;
                int line1_x1, line1_x2, line1_y1, line1_y2;
                int line2_x1, line2_x2, line2_y1, line2_y2;
                line1_x1 = 350;
                line1_x2 = 350;
                line1_y1 = 0;
                line1_y2 = 350;
                line2_x1 = 430;
                line2_x2 = 380;
                line2_y1 = 0;
                line2_y2 = 350;
                int pt1_position_line1 = (line1_y2 - line1_y1) * pt1_x + (line1_x1 - line1_x2) * pt1_y + (line1_x2 * line1_y1 - line1_x1 * line1_y2);
                int pt2_position_line1 = (line1_y2 - line1_y1) * pt2_x + (line1_x1 - line1_x2) * pt2_y + (line1_x2 * line1_y1 - line1_x1 * line1_y2);
                int pt1_position_line2 = (line2_y2 - line2_y1) * pt1_x + (line2_x1 - line2_x2) * pt1_y + (line2_x2 * line2_y1 - line2_x1 * line2_y2);
                int pt2_position_line2 = (line2_y2 - line2_y1) * pt2_x + (line2_x1 - line2_x2) * pt2_y + (line2_x2 * line2_y1 - line2_x1 * line2_y2);

                if(direction == 0)
                {
                    if(pt1_position_line1 < 0  && pt2_position_line1 >= 0)
                    {
                        track.m_trace.m_firstPass = true;
                    }
                    if (track.m_trace.m_firstPass == true && pt2_position_line2 >= 0 && track.m_trace.m_secondPass == false )
                    {
                        track.m_trace.m_secondPass = true;
                        counter++;
                    }
                }else if (direction == 1)
                {
                    if(pt2_position_line2 <= 0  && pt1_position_line2 > 0)
                    {
                        track.m_trace.m_firstPass = true;
                    }
                    if (track.m_trace.m_firstPass == true && pt2_position_line1 <= 0 && track.m_trace.m_secondPass == false )
                    {
                        track.m_trace.m_secondPass = true;
                        counter++;
                    }
                }else
                    {
                        if(pt2_position_line2 <= 0  && pt1_position_line2 > 0){
                            track.m_trace.m_firstPass = true;
                            track.m_trace.m_directionFromLeft = true;
                        }
                        else if(pt1_position_line1 < 0  && pt2_position_line1 >= 0)
                        {
                            track.m_trace.m_firstPass = true;
                            track.m_trace.m_directionFromLeft = false;
                        }
                        if (track.m_trace.m_firstPass == true && pt2_position_line1 <= 0 && track.m_trace.m_secondPass == false && track.m_trace.m_directionFromLeft == true)
                        {
                            track.m_trace.m_secondPass = true;
                            counter++;
                        }
                        else if (track.m_trace.m_firstPass == true && pt2_position_line2 >= 0 && track.m_trace.m_secondPass == false && track.m_trace.m_directionFromLeft == false)
                        {
                            track.m_trace.m_secondPass = true;
                            counter++;
                        }
                }

            }
    }

    // Draw count on frame relative to image size
    void drawtorect(cv::Mat & mat, cv::Rect target, int face, int thickness, cv::Scalar color, const std::string & str)
    {
        cv::Size rect = cv::getTextSize(str, face, 1.0, thickness, 0);
        double scalex = (double)target.width / (double)rect.width;
        double scaley = (double)target.height / (double)rect.height;
        double scale = std::min(scalex, scaley);
        int marginx = scale == scalex ? 0 : (int)((double)target.width * (scalex - scale) / scalex * 0.5);
        int marginy = scale == scaley ? 0 : (int)((double)target.height * (scaley - scale) / scaley * 0.5);
        cv::putText(mat, str, cv::Point(target.x + marginx, target.y + target.height - marginy), face, scale, color, thickness, 8, false);
    }


private:
    std::string outFile;
    std::string inFile;
    std::vector<cv::Scalar> m_colors;
    int endFrame;
    int startFrame;

    struct FrameInfo
    {
        cv::Mat m_frame;
        cv::UMat m_gray;
        regions_t m_regions;
        int64 m_dt;

        FrameInfo()
                : m_dt(0)
        {

        }
    };
    FrameInfo m_frameInfo;

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
        // Initialize the Detector
        detector.initDetection(modelFile, weightsFile, meanFile, meanValue);
        // Initialize the tracker
        config_t config;
        TrackerSettings settings;
        //settings.m_useLocalTracking = m_useLocalTracking;
        settings.m_distType = tracking::DistRects;
        settings.m_kalmanType = tracking::KalmanLinear;
        settings.m_filterGoal = tracking::FilterRect;
        settings.m_lostTrackType = tracking::TrackKCF;       // Use KCF tracker for collisions resolving
        settings.m_matchType = tracking::MatchHungrian;
        settings.m_dt = 0.3f;                                // Delta time for Kalman filter
        settings.m_accelNoiseMag = 0.1f;                     // Accel noise magnitude for Kalman filter
        settings.m_distThres = 100;              // Distance threshold between region and object on two frames
        settings.m_maximumAllowedSkippedFrames = 1 * m_fps;  // Maximum allowed skipped frames
        settings.m_maxTraceLength = 5 * m_fps;               // Maximum trace length

        m_tracker = std::make_unique<CTracker>(settings);
    }
private:
    TrackerSettings settings;
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
    void DrawData(cv::Mat frame, int framesCounter){
        for (const auto& track : m_tracker->tracks)
        {
            if (track->IsRobust(5,                           // Minimal trajectory size
                                0.2f,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
                                cv::Size2f(0.1f, 8.0f))      // Min and max ratio: width / height
                    )
            {
                //DrawTrack(frame, 1, *track);
                std::string label = track->m_lastRegion.m_type + ": " + std::to_string(track->m_lastRegion.m_confidence);
                //std::string label = std::to_string(track->m_trace.m_firstPass) + " | " + std::to_string(track->m_trace.m_secondPass);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                auto rect(track->GetLastRect());
                cv::rectangle(frame, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
            if (enableCount){
                UpdateCount(frame, *track);
            }
        }
        // Draw counter
        if (enableCount) {
            float scale = 0.2;
            float countBoxWidth = frame.size().width * scale;
            float countBoxHeight = frame.size().height * scale;
            //cv::rectangle(frame, cv::Point(0,0), cv::Point(countBoxWidth, countBoxHeight), cv::Scalar(0, 255, 0), 1, CV_AA);
            std::string counterLabel = "Count : " + std::to_string(counter);
            drawtorect(frame,
                       cv::Rect(0, 200, int(countBoxWidth), int(countBoxHeight)),
                       cv::FONT_HERSHEY_PLAIN,
                       1,
                       cv::Scalar(255, 255, 255), counterLabel);
            cv::line( frame, cv::Point( 440, 0 ), cv::Point( 380, 350), cv::Scalar( 120, 220, 0 ),  2, 8 );
            //cv::line( frame, cv::Point( 200, 0 ), cv::Point( 200, 300), cv::Scalar( 120, 220, 0 ),  2, 8 );
        }

    }
};

#endif //PROJECT_PIPELINE_H