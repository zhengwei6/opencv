#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <linux/unistd.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <sys/syscall.h>

using namespace std;
using namespace cv;
using namespace dnn;

#define SCHED_DEADLINE       6
#ifdef __x86_64__
#define __NR_sched_setattr           314
#define __NR_sched_getattr           315
#endif
struct sched_attr {
     __u32 size;

     __u32 sched_policy;
     __u64 sched_flags;

     /* SCHED_NORMAL, SCHED_BATCH */
     __s32 sched_nice;

     /* SCHED_FIFO, SCHED_RR */
     __u32 sched_priority;

     /* SCHED_DEADLINE (nsec) */
     __u64 sched_runtime;
     __u64 sched_deadline;
     __u64 sched_period;
};

int sched_setattr(pid_t pid,
               const struct sched_attr *attr,
               unsigned int flags)
{
     return syscall(__NR_sched_setattr, pid, attr, flags);
}


struct timespec diff(struct timespec start, struct timespec end) {
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}


int main(int argc, char** argv)
{
    std::vector<std::string> class_names;
    ifstream ifs(string("./classification_classes_ILSVRC2012.txt").c_str());
    string line;
    while (getline(ifs, line)) {
        class_names.push_back(line);
    }
    auto model = readNet("./DenseNet_121.prototxt",
                       "./DenseNet_121.caffemodel",
                       "Caffe");
    struct timespec start, end;
    double time_used;
    Mat image = imread("./dog.jpg");
    Mat blob = blobFromImage(image, 0.01, Size(224, 224), Scalar(104, 117, 123));
    struct sched_attr attr;
    Mat outputs;
    int runtime = 0;
    int miss_count = 0;
    int ret;
    unsigned int flags = 0;

    attr.size = sizeof(attr);
    attr.sched_flags = 0;
    attr.sched_nice = 0;
    attr.sched_priority = 0;

    attr.sched_policy = SCHED_DEADLINE;
    attr.sched_runtime = 105 * 1000 * 1000;
    attr.sched_period = attr.sched_deadline = 600 * 1000 * 1000;
    cv::setNumThreads(0);
    model.setInput(blob);
    model.forward();

    ret = sched_setattr(0, &attr, flags);
    if (ret < 0) {
        perror("sched_setattr");
        exit(-1);
    }
    // set the input blob for the neural network
    // forward pass the image blob through the model

    for (int i = 0; i < 100; i++) {

    	clock_gettime(CLOCK_REALTIME, &start);

	outputs = model.forward();

	clock_gettime(CLOCK_REALTIME, &end);
        struct timespec temp = diff(start, end);
        time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
        printf("Time = %f\n", time_used);

	if (((temp.tv_nsec >> 10) + runtime) > 600000) {
            miss_count += 1;
            runtime = ((temp.tv_nsec >> 10) + runtime) - 600000;
        } else {
            usleep(600000 - ((temp.tv_nsec >> 10) + runtime));
            runtime = 0;
        }
    }
    Point classIdPoint;
    double final_prob;
    minMaxLoc(outputs.reshape(1, 1), 0, &final_prob, 0, &classIdPoint);
    int label_id = classIdPoint.x;
    // Print predicted class.
    string out_text = format("%s, %.3f", (class_names[label_id].c_str()), final_prob);
    std::cout << out_text << std::endl;
    // put the class name text on top of the image
    putText(image, out_text, Point(25, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    imshow("Image", image);
    imwrite("result_image.jpg", image);
    return 0;
}
