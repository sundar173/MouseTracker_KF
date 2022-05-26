// Include required libraries

#include<iostream>
#include<opencv2/opencv.hpp>
// #include<opencv2/video/tracking.hpp>
// #include<opencv2/imgproc/imgproc.hpp>
#include<vector>

class Tracker
{
    private:

    int stateSize, measurementSize, controlsize;
    unsigned int F_type;

    cv::KalmanFilter KF;
    cv::Mat state;
    cv::Mat meas;
    cv::Mat filtered;
    cv::Point mousePose;

    cv::Mat display_image = cv::Mat(600, 800, CV_8UC3);
    cv::Point predictedState;
    cv::Point measuredState;
    cv::Point filteredState;

    std::vector<cv::Point> GroundTruthPosition;
    std::vector<cv::Point> EstimatedPosition;

    public:

    // Constructor
    Tracker(int n, int m, int q, unsigned int d_type)
        :stateSize(n), measurementSize(m), controlsize(q), F_type(d_type)
    {
        // Intialization of Kalman Filter
        // 1. OpenCV Kalman Filter
        // 2. Transition matrix
        // 3. Measurement matrix
        // 4. Process covariance matrix
        // 5. Measurement covariance matrix
        
        KF = cv::KalmanFilter(stateSize, measurementSize, controlsize, F_type);
        
        state = cv::Mat(stateSize, 1, F_type);
        meas = cv::Mat(measurementSize, 1, F_type);

        cv::setIdentity(KF.transitionMatrix);
        cv::setIdentity(KF.measurementMatrix);
        cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(KF.errorCovPost, cv::Scalar::all(0.1));
        
        
        // Creating display image
        // cv::Mat display_image(600, 800, CV_8UC3);
        cv::namedWindow("Kalman Filter based Mouse Tracker");


    }

    // Mouse event
    void onMouseEvent_internal(int event, int x, int y, int d, void* ptr)
    {
        cv::Point* p = (cv::Point*)ptr;
        p->x = x;
        p->y = y;
    }

    // To draw cross
    void drawPointer(cv::Point currentPoint, int offset, cv::Scalar color)
    {
        cv::line(display_image, cv::Point(currentPoint.x - offset,currentPoint.y - offset),cv::Point(currentPoint.x + offset,currentPoint.y + offset), color, 2);
        cv::line(display_image, cv::Point(currentPoint.x - offset,currentPoint.y + offset),cv::Point(currentPoint.x + offset,currentPoint.y - offset), color, 2);

    }

    // Visualization 

    void VisualizeResults(cv::Point meas, cv::Point est, std::vector<cv::Point> Truth, std::vector<cv::Point> runningEst)
    {
        cv::imshow("Kalman Filter based Mouse Tracker",display_image);
        display_image = cv::Scalar::all(0);

        drawPointer(meas,5, cv::Scalar(0,255,0));
        drawPointer(est,5, cv::Scalar(0,0,255));

        // std::cout<<"Truth:"<<meas<<std::endl;
        // std::cout<<"Filter:"<<est<<std::endl;

        for(int i = 0; i<Truth.size()-1;i++)
            cv::line(display_image, Truth[i], Truth[i+1],cv::Scalar(255,0,0), 1);

        for(int i = 0; i < runningEst.size()-1;i++) 
            cv::line(display_image, runningEst[i], runningEst[i+1],cv::Scalar(0,0,255), 1);
    }


    // Main tracker function
    void TrackMousePointer_KF()
    {
        // Predict step
        std::cout<<"Transition"<<KF.transitionMatrix<<std::endl;
        state = KF.predict();
        predictedState = cv::Point(state.at<float>(0), state.at<float>(1));
        std::cout<<"Predicted:"<<predictedState<<std::endl;

        // Reading Mouse pointer inputs
        cv::setMouseCallback("Kalman Filter based Mouse Tracker", onMouseEvent_external, this);
        meas.at<float>(0) = mousePose.x;
        meas.at<float>(1) = mousePose.y;
        std::cout<<"Measured:"<<meas<<std::endl;
        // Update step
        filtered = KF.correct(meas);
        std::cout<<"Filter:"<<filtered<<std::endl;
            
        measuredState = cv::Point(mousePose.x, mousePose.y); 
        filteredState = cv::Point(filtered.at<float>(0), filtered.at<float>(1));
        
        GroundTruthPosition.push_back(measuredState);
        EstimatedPosition.push_back(filteredState);

        //// Visualize the results

        VisualizeResults(measuredState, filteredState, GroundTruthPosition, EstimatedPosition);

    }

    friend void onMouseEvent_external(int event, int x, int y, int d, void* obj);

};


void onMouseEvent_external(int event, int x, int y, int d, void* obj)
{
    Tracker *t = static_cast<Tracker*>(obj);
    if(obj)
        t->onMouseEvent_internal(event, x, y, d, &(t->mousePose));
}

int main()
{
    int noOfStates = 4;
    int noOfMeasurements = 2;
    int noOfControlInputs = 0;
    unsigned int d_type = CV_32F;
    char ch = 's';

    Tracker trackerObject(noOfStates, noOfMeasurements, noOfControlInputs, d_type);
    
    while(ch!='q' && ch!='Q')
    {
        trackerObject.TrackMousePointer_KF();
        ch = cv::waitKey(10);
    }
}
