import copy
import gc
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import grad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import ResNet20
from abc import abstractmethod


class ResNet20Wrapper:
    def __init__(self, model_path, label_names, bottlenecks=None):
        if bottlenecks is None:
            bottlenecks = {'avgpool': 'avgpool'}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model
        print(model_path)
        self.model = ResNet20.CifarResNet()
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.activation = {}
        # self.activation = []
        self.gradient = {}
        self.bottlenecks_tensors = {}
        self.cutted_model = None

        # Define transforms for input images
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        # Define label names
        # Read label names from file
        with open(label_names, 'r') as f:
            self.label_names = f.read().splitlines()

        def save_activation(name):
            """ Creates hooks to the activations
            Args:
                name (string): Name of the layer to hook into
            """

            def hook(mod, inp, out):
                """ Saves the activation hook to dictionary
                """
                self.bottlenecks_tensors[name] = out

            return hook

        for name, mod in self.model._modules.items():
            if name in bottlenecks.keys():
                mod.register_forward_hook(save_activation(bottlenecks[name]))

    def _make_gradient_tensors(self, y, bottleneck_name):
        """
        Makes gradient tensor for logit y w.r.t. layer with activations

        Args:
            y (int): Index of logit (class)
            bottleneck_name (string): Name of layer activations

        Returns:
            (torch.tensor): Gradients of logit w.r.t. to activations

        """
        acts = self.bottlenecks_tensors[bottleneck_name]
        return grad(self.ends[:, y], acts)

    # def __call__(self, x):
    #     """ Calls prediction on wrapped model pytorch.
    #     """
    #     self.ends = self.model(x)
    #     return self.ends

    # 用来获取模型中间层输出的hook
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def get_gradients(self, name):
        def hook(model, input, output):
            self.gradient[name] = output[0].detach()

        return hook

    # def forward_hook(self, model, ten_in, ten_out):
    #     self.activation.append(copy.deepcopy(ten_out.cpu().detach().numpy()))
    #
    # def backward_hook(self, model, grad_in, grad_out):
    #     self.gradient.append(copy.deepcopy(grad_out[0].cpu().detach().numpy()))

    def run_examples(self, images, bottleneck_layer):
        # Preprocess images
        # preprocessed_images = torch.stack([self.transform(Image.fromarray(img)) for img in images])

        # Move images to the appropriate device
        # preprocessed_images = preprocessed_images.to(self.device)

        # Forward pass through the model to get activations in the specified bottleneck layer
        # activations = self.model(images, bottleneck_layer)
        self.activation = {}
        # print(images.shape)
        # print(images.shape)
        Preprocess_images = torch.tensor(images, dtype=torch.float32)
        Preprocess_images = Preprocess_images.permute(0, 3, 1, 2)
        self.model.eval()  # Set the model to evaluation mode

        # 钩子函数
        hook_handle = self.model.avgpool.register_forward_hook(self.get_activation('avgpool'))
        self.model(Preprocess_images)
        activations = self.activation['avgpool']

        # 取消fc层
        # feature_model = copy.deepcopy(self.model)
        # feature_model.fc = nn.Identity()  # 相当于取消fc层, 这样
        # activations = feature_model(Preprocess_images)

        # Move activations to CPU and convert to numpy array
        activations = activations.detach().cpu().numpy()
        # print("model钩子函数出来的activation形状",activations.shape)
        # model钩子函数出来的activation形状 (100, 64, 1, 1)
        # 移除钩子函数
        hook_handle.remove()

        return activations

    def get_image_shape(self):
        return (32, 32, 3)  # Assuming CIFAR-10 images are RGB with size 32x32

    def label_to_id(self, class_name):
        return self.label_names.index(class_name)



    # 以下这一版参考 https://github.com/MKowal2/ACE_TCAV_Pytorch/blob/main/ace.py#L634
    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(acts).to(device), requires_grad=True)
        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        self.cutted_model = self._get_cutted_model(bottleneck_name).to(device)
        self.cutted_model.eval()
        outputs = self.cutted_model(inputs)
        # 手动执行反向传播

        grads = -grad(outputs[:, y[0]], inputs, create_graph=True)[0]
        # print(grads)
        grads = grads.detach().cpu().numpy()




        self.cutted_model = None
        gc.collect()
        # print(grads.reshape(-1))
        return grads

    def _get_cutted_model(self, bottleneck):
        # get layers only after bottleneck
        new_model_list = OrderedDict()
        add_to_list = False
        for name, layer in self.model.named_children():
            # for name, layer in self.model.named_modules():
            if add_to_list:
                if not 'aux' in name:
                    if name == 'fc':
                        new_model_list['flatten'] = torch.nn.Flatten()
                        new_model_list[name] = layer
                    new_model_list[name] = layer
            if name == bottleneck:
                add_to_list = True
        cutted_model = torch.nn.Sequential(new_model_list)
        return cutted_model
    # pytorch版tcav参考 https://github.com/AntonotnaWang/TCAV-pytorch/blob/main/TCAV_pytorch.ipynb
    # https://github.com/mlomnitz/tcav_pytorch/blob/master/model_wrapper.py
    # def get_gradient(self, imgs, class_id, bottleneck_layer):
    #     # Placeholder for gradient computation (not implemented in this example)
    #     self.gradient = {}
    #     Preprocess_images = torch.tensor(imgs, dtype=torch.float32)
    #     Preprocess_images = Preprocess_images.permute(0, 3, 1, 2)
    #
    #     # self.model.avgpool.register_backward_hook(self.get_gradient('avgpool'))
    #     model_output = self.model(Preprocess_images)
    #     # Zero gradients
    #     self.model.zero_grad()
    #     # acts = self.bottlenecks_tensors[bottleneck_layer]
    #     self.model.avgpool.register_forward_hook(self.get_activation('avgpool'))
    #     acts = self.activation['avgpool']
    #     # print(acts)
    #     # 第二个元素是输入张量的梯度。这个输入张量是调用 grad 函数时传入的张量，计算相对于它的梯度。,第一个元素才是我们需要的
    #     gradients = grad(model_output[:, class_id], acts)[0]
    #     # print("gradient de 形状",gradients.shape)
    #
    #     return gradients
    # # https://github.com/agil27/TCAV_PyTorch/blob/master/tcav/model_wrapper.py
    # def get_gradient(self, imgs, c, layer_name):
    #     def save_gradient(grad):
    #         self.gradients = grad
    #
    #     Preprocess_images = torch.tensor(imgs, dtype=torch.float32)
    #     Preprocess_images = Preprocess_images.permute(0, 3, 1, 2)
    #
    #
    #
    #     # self.model.avgpool.register_backward_hook(self.get_gradient('avgpool'))
    #     model_output = self.model(Preprocess_images)
    #     # self.model.zero_grad()
    #
    #
    #
    #
    #     activation = self.bottlenecks_tensors[layer_name]
    #     activation.register_hook(save_gradient)
    #
    #     # 重置梯度为 None
    #     self.gradients = None
    #     logit = model_output[:, c]
    #     logit.backward(torch.ones_like(logit), retain_graph=True)
    #     # gradients = grad(logit, activation, retain_graph=True)[0]
    #     # gradients = gradients.cpu().detach().numpy()
    #     gradients = self.gradients.cpu().detach().numpy()
    #     # print(gradients)
    #     return gradients
