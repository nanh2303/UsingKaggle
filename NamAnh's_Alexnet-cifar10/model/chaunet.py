import torch.nn as nn


class ChauNet(nn.Module):
    def __init__(self, num_classes, activation='relu', dropout=True):
        super().__init__()
        if activation == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activate = nn.Tanh(inplace=True)
        elif activation == 'sigmoid':
            self.activate = nn.Sigmoid(inplace=True)
        elif activation == 'leaky_relu':
            self.activate = nn.LeakyReLU(inplace=True)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            self.activate,
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            self.activate,
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            self.activate,
            
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            self.activate,
            
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            self.activate,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        
        if dropout == True:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                
                nn.Linear(9216, 512),
                self.activate,
                
                nn.Linear(512, 512),
                self.activate,
                
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(9216, 512),
                self.activate,
                
                nn.Linear(512, 512),
                self.activate,
                
                nn.Linear(512, num_classes)
            )
        
    def forward(self, x):
        x = self.feature_extraction(x)  # (6x6x256) 
        x = self.flatten(x)             # 9216
        logit = self.classifier(x)      # num_classes, ^y

        return logit