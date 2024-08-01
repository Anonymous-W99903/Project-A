import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models
import resnet

"""
NOTE:
All network output in this file is logits instead of softmax or other 
activation fucntion output.

"""

######################### Simple Classifier for AutoFi ###############################

class SimpleClassifier(nn.Module):
    def __init__(self, in_planes = 180, num_classes = 10):
        # nn.Module.__init__(self)
        super(SimpleClassifier, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_classes

        # network layers
        planes = [self.in_planes,300, 100]
        # planes = [self.in_planes, 5]
        self.layers = []

        in_planes = None
        out_planes = None
        for i in range(len(planes)-1):
            in_planes = planes[i]
            out_planes = planes[i+1]
            layer = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            self.layers.append(layer)

        self.net = nn.Sequential(*self.layers)
        self.out = nn.Linear(out_planes, self.num_classes)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Linear) :
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x : torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        x = self.out(x)
        return x


############################# Network Model for Activity Recognition ###########################


class FeatureExtracter2d(nn.Module):
    """
    input: #batch x #receiver x #frequence x #packet
    bs x 6 x 121 x ??128
    """

    def __init__(self, input_dim, params):
        super(FeatureExtracter2d, self).__init__()
        # self.n_layer = params['n_layer']
        # self.in_planes = 6
        self.feat_dim = params['feat_dim']
        self.planes = [input_dim, 64, 128, 256, 512, self.feat_dim]


        self.cnns = nn.ModuleList()
        for i in range(len(self.planes)-1):
            in_planes, out_planes = self.planes[i], self.planes[i+1]
            self.cnns.append(Conv2dBlock(in_planes, out_planes, 3, 1, 1))


    def forward(self, x):
        for i, cnn in enumerate(self.cnns):
            x = cnn(x)
            # print(f"Shape after cnn layer {i}: {x.shape}")
        # global avg pooling to reduce dim
        # x = F.avg_pool2d(x, kernel_size = (x.size(2), x.size(3)))
        x = torch.reshape(x, (x.size(0), -1))
        # print(f"Final reshaped feature: {x.shape}") 
        return x
    
    



class ActRecognizer(nn.Module):
    def __init__(self, params):
        super(ActRecognizer, self).__init__()
        self.feat_dim = params['feat_dim']
        self.softplus_beta = params['softplus_beta']
        self.num_classes = params['num_class']

        self.fc1 = nn.Linear(self.feat_dim * 9, self.feat_dim//4)
        self.softplus = nn.Softplus(self.softplus_beta)
        # self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.feat_dim//4, self.num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.softplus(x)
        # x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, input_dim, params):
        super(ConvNet, self).__init__()
        self.extracter = FeatureExtracter2d(input_dim, params)
        self.recognizer = ActRecognizer(params)

        # initialize parameters
        for m in self.modules() :
            if isinstance(m, nn.Linear) :
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.extracter(x)
        logits = self.recognizer(feat)
        return logits

    def extract_features(self, x):
        feat = self.extracter(x)
        x = self.recognizer.fc1(feat)
        x = self.recognizer.softplus(x)
        return x

class resnet18(nn.Module):
    def __init__(self, input_dim, params):
        self.feat_dim = params['feat_dim']
        self.softplus_beta = params['softplus_beta']
        self.num_classes = params['num_class']

        super(resnet18, self).__init__()
        self.extracter = resnet.resnet18(input_dim = input_dim, num_classes = self.feat_dim)
        self.softplus = nn.Softplus(self.softplus_beta)
        # self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.feat_dim, self.num_classes)

        # initialize recognizer parameters
        for m in self.modules() :
            if isinstance(m, nn.Linear) :
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.extracter(x)
        x = self.softplus(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
    
    
#################### DANN ##############################################

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class Domain_classifier(nn.Module):

    def __init__(self, params):
        super(Domain_classifier, self).__init__()
        self.feat_dim = params['feat_dim']
        self.num_domain = params['num_domain']
        
        self.fc1 = nn.Linear(self.feat_dim*9, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, self.num_domain)
        
    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        logits = F.relu(self.fc1(input))
        logits = self.dropout(logits)
        logits = self.fc2(logits)

        return logits
    

class DANN(nn.Module):
    def __init__(self, input_dim, params):
        super(DANN, self).__init__()
        self.extracter = FeatureExtracter2d(input_dim, params)
        self.recognizer = ActRecognizer(params)
        self.domain_classifier = Domain_classifier(params)

        # initialize parameters
        for m in self.modules() :
            if isinstance(m, nn.Linear) :
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant):
        feat = self.extracter(x)
        class_preds = self.recognizer(feat)
        domain_preds = self.domain_classifier(feat, constant)
        return class_preds, domain_preds


#################### Basic Blocks ######################################

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0):
        super(Conv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(output_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        return x

class Conv2dBlock2(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0):
        super(Conv2dBlock2, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(output_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size, stride, padding=padding)
        self.norm2 = nn.BatchNorm2d(output_dim)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        return x
#============================================================
class PrunedFeatureExtracter2d(nn.Module):
    """
    input: #batch x #receiver x #frequency x #packet
    bs x 6 x 121 x ??128
    """

    def __init__(self, input_dim, params):
        super(PrunedFeatureExtracter2d, self).__init__()
        self.feat_dim = params['feat_dim']
        self.planes = [input_dim, 13, 26, 51, 102, 102]

        self.cnns = nn.ModuleList()
        for i in range(len(self.planes)-1):
            in_planes, out_planes = self.planes[i], self.planes[i+1]
            self.cnns.append(Conv2dBlock(in_planes, out_planes, 3, 1, 1))

    def forward(self, x):
        for i, cnn in enumerate(self.cnns):
            x = cnn(x)
        x = torch.reshape(x, (x.size(0), -1))
        return x

class PrunedActRecognizer(nn.Module):
    def __init__(self, params):
        super(PrunedActRecognizer, self).__init__()
        self.feat_dim = params['feat_dim']
        self.softplus_beta = params['softplus_beta']
        self.num_classes = params['num_class']

        #self.fc1 = nn.Linear(self.feat_dim * 9, 26)
        self.fc1 = nn.Linear(918, 26)
        self.softplus = nn.Softplus(self.softplus_beta)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(26, self.num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.softplus(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PrunedConvNet(nn.Module):
    def __init__(self, input_dim, params):
        super(PrunedConvNet, self).__init__()
        self.extracter = PrunedFeatureExtracter2d(input_dim, params)
        self.recognizer = PrunedActRecognizer(params)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.extracter(x)
        logits = self.recognizer(feat)
        return logits

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0):
        super(Conv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(output_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        return x

class Conv2dBlock2(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0):
        super(Conv2dBlock2, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(output_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size, stride, padding=padding)
        self.norm2 = nn.BatchNorm2d(output_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        return x