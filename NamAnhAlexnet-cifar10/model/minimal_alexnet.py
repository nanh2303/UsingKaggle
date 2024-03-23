import torch.nn as nn


class Minimal_Alexnet(nn.Module):
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
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1),
            self.activate,
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.activate,
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1),
            self.activate,
            
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            self.activate,
            
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
            self.activate,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        if dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),            # Tranh overfitting
                
                nn.Linear(1152, 1024),
                self.activate,
                
                nn.Linear(1024, 1024),
                self.activate,
                
                nn.Linear(1024, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(1152, 1024),
                self.activate,
                
                nn.Linear(1024, 1024),
                self.activate,
                
                nn.Linear(1024, num_classes)
            )
            
    def forward(self, x):
        x = self.feature_extraction(x)  # (6x6x256) 
        x = self.flatten(x)             # 9216
        logit = self.classifier(x)      # num_classes, ^y

        return logit