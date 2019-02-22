//
//  ViewController.m
//  FaceRecognitionIOS
//
//  Created by LuDong on 2019/2/13.
//  Copyright © 2019年 LuDong. All rights reserved.
//

#import "ViewController.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    [imageView setFrame:[[UIScreen mainScreen] bounds]];
    [imageView setContentMode:UIViewContentModeScaleAspectFit];
    
    [imageView setUserInteractionEnabled:TRUE];
    
    l2space = new hnswlib::L2Space(FACE_EMBEDDING_SIZE);
    
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *documentPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) lastObject];
    
    hnswPath = [documentPath stringByAppendingPathComponent:@"hnsw.dat"];
    labelPath = [documentPath stringByAppendingPathComponent:@"label.txt"];
    
    if([fileManager fileExistsAtPath:hnswPath]) {   // The hnsw binary file.
        app_alg = new hnswlib::HierarchicalNSW<float>(l2space, std::string([hnswPath UTF8String]));
    }
    else {
        app_alg = new hnswlib::HierarchicalNSW<float>(l2space, 10*10000, 16, 200);
    }
    if([fileManager fileExistsAtPath:labelPath]) {  // Label txt file.
        labelFile = fopen([labelPath UTF8String], "at+");
    }
    else {
        labelFile = fopen([labelPath UTF8String], "w");
    }
    
    labelArr = [[NSMutableArray alloc] init];
    NSString *contents = [NSString stringWithContentsOfFile:labelPath encoding:NSUTF8StringEncoding error:nil];
    contents = [contents stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if(![contents isEqualToString:@""]) {
        labelArr = [[NSMutableArray alloc] initWithArray:[contents componentsSeparatedByString:@"\n"]];
    }

    [self startCapture:imageView];
}

bool isInside(cv::Rect rect1, cv::Rect rect2) { // decide whether bbox out of bound.
    return (rect1 == (rect1&rect2));
}

std::vector<cv::Mat> preWhiten(cv::Mat &mat) {
    
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::Mat tmpMat = mat.reshape(1, 1);
    tmpMat.convertTo(tmpMat, CV_32F);
    meanStdDev(tmpMat, mean, stddev);
    vector<cv::Mat> xc;
    
    mat.convertTo(mat, CV_32F);
    split(mat, xc);
    for (int i = 0; i<xc.size(); i++) {
        float *data = (float *)xc[i].data;
        for(int k=0; k<xc[i].cols*xc[i].rows; k++) {
            data[k] = (data[k]- mean(0)) / stddev(0);
        }
    }
    return xc;
}

-(float *)faceEmbeddingCoreML:(cv::Mat) faceCrop {  //Facenet calculate face embedding, output 512D float.
    
    cv::resize(faceCrop, faceCrop, cv::Size(160, 160));
    std::vector<cv::Mat> mergeMat = preWhiten(faceCrop);
    
    int count = 0;
    for(int i=0; i<160*160; i++) {
        dataPointer[count++] = *((float *)mergeMat[0].data+i);
    }
    for(int i=0; i<160*160; i++) {
        dataPointer[count++] = *((float *)mergeMat[1].data+i);
    }
    for(int i=0; i<160*160; i++) {
        dataPointer[count++] = *((float *)mergeMat[2].data+i);
    }
    MLMultiArray *arr = [[MLMultiArray alloc] initWithDataPointer:dataPointer shape:[NSArray arrayWithObjects:[NSNumber numberWithInt:3], [NSNumber numberWithInt:160], [NSNumber numberWithInt:160], nil] dataType:MLMultiArrayDataTypeDouble strides:[NSArray arrayWithObjects:[NSNumber numberWithInt:160*160], [NSNumber numberWithInt:160], [NSNumber numberWithInt:1], nil] deallocator:nil error:nil];
    InceptionResnetOutput *output = [irModel predictionFromData:arr error:nil];
    
    MLMultiArray *multiArr = [output flatten];
    double *vector = (double *)[multiArr dataPointer];
    float sum = 0;
    for(int i=0; i<FACE_EMBEDDING_SIZE; i++) {
        sum += pow(vector[i], 2);
    }
    sum = sqrt(sum);
    if(sum<0.0000000001) sum = 0.0000000001;
    for(int i=0; i<FACE_EMBEDDING_SIZE; i++) {
        faceVector[i] = vector[i]/sum;
    }
    return faceVector;
}

cv::Mat drawDetection(const cv::Mat &img, std::vector<Bbox> &box) { //Draw bounding box with 5 keypoints.
    cv::Mat show = img.clone();
    int num_box = (int)box.size();
    std::vector<cv::Rect> bbox;
    bbox.resize(num_box);
    for (int i = 0; i < num_box; i++) {
        bbox[i] = cv::Rect(box[i].x1, box[i].y1, box[i].x2 - box[i].x1 + 1, box[i].y2 - box[i].y1 + 1);
        
        for (int j = 0; j < 5; j = j + 1) {
            cv::circle(show, cvPoint(box[i].ppoint[j], box[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
        }
    }
    int i=0;
    for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
        if(!strcmp(box[i].text.c_str(), "UnKnown")) {
            rectangle(show, (*it), cv::Scalar(255, 127, 0), 2, 8, 0);
            cv::putText(show, box[i++].text, cvPoint((*it).x, (*it).y-10), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 127, 255), 2);
        }
        else {
            rectangle(show, (*it), cv::Scalar(0, 255, 0), 2, 8, 0);
            cv::putText(show, box[i++].text, cvPoint((*it).x, (*it).y-10), cv::FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 255, 0), 2);
        }
    }
    return show;
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat {   //  Convert Mat to UIImage, so can be displayed in UIImage.
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection  {
    
    if ([[inputDevice device] position] == AVCaptureDevicePositionBack) {
        [connection setVideoOrientation:AVCaptureVideoOrientationPortrait];
    }
    
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    
    uint8_t *baseAddress = (uint8_t *)CVPixelBufferGetBaseAddress(imageBuffer);
    
    int width = (int)CVPixelBufferGetWidth(imageBuffer);
    int height = (int)CVPixelBufferGetHeight(imageBuffer);
    ////    ---MTCNN---     ////
    
    if(planerData==NULL) {
        planerData = (uint8_t *)malloc(width*height*3);
    }
    int cnt = 0;
    int planeSize = width*height;
    for(int i=0; i<width*height; i++) {
        planerData[planeSize*2 + cnt] = baseAddress[i*4];
        planerData[planeSize + cnt] = baseAddress[i*4+1];
        planerData[cnt] = baseAddress[i*4+2];
        cnt++;
    }
    
    cv::Mat chn[] = {
        cv::Mat(height, width, CV_8UC1, planerData),  // starting at 1st blue pixel
        cv::Mat(height, width, CV_8UC1, planerData + planeSize),    // 1st green pixel
        cv::Mat(height, width, CV_8UC1, planerData + planeSize*2)   // 1st red pixel
    };
    
    cv::Mat frame;
    merge(chn, 3, frame);
    if ([[inputDevice device] position] == AVCaptureDevicePositionFront) {  //Front Camera FLIP
        transpose(frame, frame);
        flip(frame, frame, 1);
    }
    
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_RGB, frame.cols, frame.rows);
    std::vector<Bbox> finalBbox;
    double start_time = CACurrentMediaTime();
    mtcnn.detect(ncnn_img, finalBbox);
    double finish_time = CACurrentMediaTime();
    double total_time = (double)(finish_time - start_time);
//    std::cout << "cost detect: " << total_time * 1000 << "ms" << std::endl;

    ////    ---FaceNet---     ////
    for(int boxIndex = 0; boxIndex<finalBbox.size(); boxIndex++) {
        cv::Rect rect = cv::Rect(finalBbox[boxIndex].x1, finalBbox[boxIndex].y1, finalBbox[boxIndex].x2 - finalBbox[boxIndex].x1 + 1, finalBbox[boxIndex].y2 - finalBbox[boxIndex].y1 + 1);
        
        if(isInside(rect, cv::Rect(0,0,frame.cols,frame.rows))) {
            
            start_time = CACurrentMediaTime();
            
            float *ret = [self faceEmbeddingCoreML:frame(rect).clone()];
            finalBbox[boxIndex].text = std::string("UnKnown");
            
            if(app_alg->cur_element_count>0) {
                std::priority_queue<std::pair<float, hnswlib::labeltype >> queue = app_alg->searchKnn(ret, 1);
                float distance = queue.top().first;
                if(distance<0.5) {
                    NSString *name = [labelArr objectAtIndex:queue.top().second];
                    finalBbox[boxIndex].text = std::string([name UTF8String]);
                }
                else {
                    finalBbox[boxIndex].text = std::string("UnKnown");
                }
            }
            
            finish_time = CACurrentMediaTime();
            total_time = (double)(finish_time - start_time);
//            std::cout << "cost recognize: " << total_time * 1000 << "ms" << std::endl;
            
        }
    }
    cv::Mat show = drawDetection(frame, finalBbox);
    finalBbox.clear();
    
    dispatch_async(dispatch_get_main_queue(), ^{
        [imageView setImage:[self UIImageFromCVMat:show]];
    });
    
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
}

-(void)startCapture:(UIImageView *)capImageView {

    NSArray *cameraArray = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *device in cameraArray) {
        [device lockForConfiguration:nil];
        [device setActiveVideoMaxFrameDuration:CMTimeMake(1, 10)];
        [device setActiveVideoMinFrameDuration:CMTimeMake(1, 15)];
        [device unlockForConfiguration];
        
        if ([device position] == AVCaptureDevicePositionBack) {
            backCamera = device;
        }
        else if ([device position] == AVCaptureDevicePositionFront) {
            frontCamera = device;
        }
    }
    
    session = [[AVCaptureSession alloc] init];
    session.sessionPreset = AVCaptureSessionPreset640x480;
    inputDevice = [AVCaptureDeviceInput deviceInputWithDevice:backCamera error:nil];
    [session addInput:inputDevice];     //输入设备与session连接
    
    /*  设置输出yuv格式   */
    output = [[AVCaptureVideoDataOutput alloc] init];
    NSNumber *value = [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA];
    NSDictionary *dictionary = [NSDictionary dictionaryWithObject:value forKey:(NSString *)kCVPixelBufferPixelFormatTypeKey];
    [output setVideoSettings:dictionary];
    [output setAlwaysDiscardsLateVideoFrames:YES];
    
    /*  设置输出回调队列    */
    dispatch_queue_t queue = dispatch_queue_create("com.linku.queue", NULL);
    [output setSampleBufferDelegate:self queue:queue];
    //    dispatch_release(queue);
    [session addOutput:output];     //输出与session连接
    
    /////////////mtcnn-ncnn
    char *path = (char *)[[[NSBundle mainBundle] resourcePath] UTF8String];
    mtcnn.init(path);
    mtcnn.SetMinFace(40);
    planerData = NULL;
    
    ////facenet-CoreML
    irModel = [[InceptionResnet alloc] init];
    faceVector = (float *)malloc(FACE_EMBEDDING_SIZE*sizeof(float));
    dataPointer = (double *)malloc(sizeof(double)*160*160*3);
    memset(faceVector, 0, FACE_EMBEDDING_SIZE*sizeof(float));
    memset(dataPointer, 0, sizeof(double)*160*160*3);
    
    if(![session isRunning]) {
        [session startRunning];
    }
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

-(void)setButtonDisable {
    
    dispatch_async(dispatch_get_main_queue(), ^{
        [enrollBtn setHidden:YES];
    });
}

- (IBAction)switchCamera:(id)sender {

    if(![session isRunning]) {
        return;
    }
    [session removeInput:inputDevice];
    if ([[inputDevice device] position] == AVCaptureDevicePositionFront) {
        inputDevice = [AVCaptureDeviceInput deviceInputWithDevice:backCamera error:nil];
        [session addInput:inputDevice];
    }
    else if ([[inputDevice device] position] == AVCaptureDevicePositionBack) {
        inputDevice = [AVCaptureDeviceInput deviceInputWithDevice:frontCamera error:nil];
        [session addInput:inputDevice];
    }
}

- (IBAction)backAction:(id)sender {
    
    [session removeInput:inputDevice];
    [session stopRunning];
    [self.presentingViewController dismissViewControllerAnimated:YES completion:nil];
}

- (IBAction)enrollAction:(id)sender {
    [session stopRunning];
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Enrollment" message:@"Input name" preferredStyle:UIAlertControllerStyleAlert];
    
    [alert addTextFieldWithConfigurationHandler:^(UITextField * _Nonnull textField) {
        textField.placeholder = @"name";
    }];
    UIAlertAction *confirmAction = [UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        
        NSString *name = [[alert textFields][0] text];
        if([name length]==0) {
            [session startRunning];
            return;
        }

        app_alg->addPoint(faceVector, [labelArr count]);   // Add faceVector to hnsw file.
        app_alg->saveIndex(std::string([hnswPath UTF8String]));
        
        fprintf(labelFile, "%s\n", [name UTF8String]);  // Add name to label file.
        fflush(labelFile);
        [labelArr addObject:name];
        
        [session startRunning];
    }];
    UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
        [session startRunning];
    }];
    [alert addAction:confirmAction];
    [alert addAction:cancelAction];
    [self presentViewController:alert animated:YES completion:nil];
}
@end
