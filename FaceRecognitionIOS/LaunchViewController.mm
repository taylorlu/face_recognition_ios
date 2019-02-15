//
//  LaunchViewController.m
//  FaceRecognitionIOS
//
//  Created by LuDong on 2019/2/13.
//  Copyright © 2019年 LuDong. All rights reserved.
//

#import "LaunchViewController.h"
#import "ViewController.h"

@interface LaunchViewController ()

@end

@implementation LaunchViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


#pragma mark - Navigation

// In a storyboard-based application, you will often want to do a little preparation before navigation
- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    // Get the new view controller using [segue destinationViewController].
    // Pass the selected object to the new view controller.
    if([[segue identifier] isEqualToString:@"recognition"]) {
        [(ViewController *)[segue destinationViewController] setButtonDisable];
    }
}

@end
