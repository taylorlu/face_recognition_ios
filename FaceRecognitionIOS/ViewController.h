//
//  ViewController.h
//  FaceRecognitionIOS
//
//  Created by LuDong on 2019/2/13.
//  Copyright © 2019年 LuDong. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "InceptionResnet.h"
#import <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>
#import <AVFoundation/AVFoundation.h>
#import <MobileCoreServices/MobileCoreServices.h>
#import <Endian.h>
#import <CoreML/CoreML.h>
#include "mtcnn.h"
#include "net.h"
#include "hnswlib.h"

#define FACE_EMBEDDING_SIZE 512

@interface ViewController : UIViewController<AVCaptureVideoDataOutputSampleBufferDelegate> {
    
    __weak IBOutlet UIButton *enrollBtn;
    __weak IBOutlet UIImageView *imageView;
    
    bool isCapture;
    AVCaptureDevice *frontCamera;
    AVCaptureDevice *backCamera;
    uint8_t *originData;
    
    AVCaptureVideoDataOutput *output;
    AVCaptureSession     *session;
    AVCaptureDeviceInput *inputDevice;
    AVCaptureVideoPreviewLayer   *previewLayer;
    
    InceptionResnet *irModel;
    
    float *faceVector;
    uint8_t *planerData;
    double *dataPointer;
    
    MTCNN mtcnn;
    hnswlib::L2Space *l2space;
    hnswlib::HierarchicalNSW<float> *app_alg;
    NSString *hnswPath;
    FILE *labelFile;
    NSString *labelPath;
    NSMutableArray *labelArr;
}

- (IBAction)switchCamera:(id)sender;
- (IBAction)backAction:(id)sender;
- (IBAction)enrollAction:(id)sender;
-(void)setButtonDisable;

@end

