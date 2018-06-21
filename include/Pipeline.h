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
        m_fps = 25;

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
        int frame_count = 0;

        // video output
        cv::VideoWriter writer;

        double tStart  = cv::getTickCount();

        while (true) {
            bool success = cap.read(frame);
            if (!success) {
                LOG(INFO) << "Process " << frame_count << " frames from " << inFile;
                break;
            }
            CHECK(!frame.empty()) << "Error when read frame";
            std::vector<vector<float> > detections = detectframe(frame);

            regions_t tmpRegions;
            for (int i = 0; i < detections.size(); ++i) {
                const vector<float> &d = detections[i];
                // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                CHECK_EQ(d.size(), 7);
                const float score = d[2];
                std::string label = std::to_string(d[1]);
                  if (score >= 0.5) {
                      int xLeftBottom = d[3] * frame.cols;
                      int yLeftBottom = d[4] * frame.rows;
                      int xRightTop = d[5] * frame.cols;
                      int yRightTop = d[6] * frame.rows;
                      cv::Rect object(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
                      tmpRegions.push_back(CRegion(object, label, score));
                  }

                //cv::imshow("Video", frame);
            }

            // Update Tracker
            cv::UMat clFrame;
            clFrame = frame.getUMat(cv::ACCESS_READ);
            m_tracker->Update(tmpRegions, clFrame, m_fps);
            ++frame_count;
            DrawData(frame, frame_count);
            if (!writer.isOpened())
            {
                writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), m_fps, frame.size(), true);
            }
            if (writer.isOpened())
            {
                writer << frame;
            }

        }

        double tEnd  = cv::getTickCount();
        double runTime = (tEnd - tStart)/cv::getTickFrequency();
        LOG(INFO)  << "work time = " << runTime << " | Frame rate: "<< frame_count/runTime << std::endl;
        if (cap.isOpened()) {
            cap.release();
        }
    }
protected:
    std::unique_ptr<CTracker> m_tracker;
    float m_fps;
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
                    cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, CV_AA);
                }
            }
        }
    }
private:
    std::string outFile;
    std::string inFile;
    std::vector<cv::Scalar> m_colors;

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
                DrawTrack(frame, 1, *track);
                std::string label = track->m_lastRegion.m_type + ": " + std::to_string(track->m_lastRegion.m_confidence);
                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                auto rect(track->GetLastRect());
                cv::rectangle(frame, cv::Rect(cv::Point(rect.x, rect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }
    }
};

#endif //PROJECT_PIPELINE_H