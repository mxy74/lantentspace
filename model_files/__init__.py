# 获取要被诊断的DNN模型
def get_DNN_model(dataset, model_name):
	if dataset == "CIFAR10":
		if model_name == "AlexNet":
			from .CIFAR10.models import AlexNet
			return AlexNet.AlexNet()
		if model_name == "kjl_AlexNet":
			from .CIFAR10.models import kjl_AlexNet
			return kjl_AlexNet.AlexNet()
		if model_name == "ResNet20":
			from .CIFAR10.models import ResNet20
			return ResNet20.CifarResNet()
		if model_name == "ResNet32":
			from .CIFAR10.models import ResNet32
			return ResNet32.CifarResNet()
		if model_name == "ResNet44":
			from .CIFAR10.models import ResNet44
			return ResNet44.CifarResNet()
		if model_name == "ResNet56":
			from .CIFAR10.models import ResNet56
			return ResNet56.CifarResNet()
		if model_name == "VGG11_BN":
			from .CIFAR10.models import VGG
			return VGG.VGG("A")
		if model_name == "VGG13_BN":
			from .CIFAR10.models import VGG
			return VGG.VGG("B")
		if model_name == "VGG16_BN":
			from .CIFAR10.models import VGG
			return VGG.VGG("C")
		if model_name == "VGG19_BN":
			from .CIFAR10.models import VGG
			return VGG.VGG("D")
		if model_name == "MobileNetV2_x0_5":
			from .CIFAR10.models import MobileNetV2
			return MobileNetV2.MobileNetV2(width_mult=0.5)
		if model_name == "MobileNetV2_x0_75":
			from .CIFAR10.models import MobileNetV2
			return MobileNetV2.MobileNetV2(width_mult=0.75)
		if model_name == "MobileNetV2_x1_0":
			from .CIFAR10.models import MobileNetV2
			return MobileNetV2.MobileNetV2(width_mult=1.0)
		if model_name == "MobileNetV2_x1_4":
			from .CIFAR10.models import MobileNetV2
			return MobileNetV2.MobileNetV2(width_mult=1.4)
		if model_name == "ShuffleNetV2_x0_5":
			from .CIFAR10.models import ShuffleNetV2
			return ShuffleNetV2.ShuffleNetV2(config_type="x0_5")
		if model_name == "ShuffleNetV2_x1_0":
			from .CIFAR10.models import ShuffleNetV2
			return ShuffleNetV2.ShuffleNetV2(config_type="x1_0")
		if model_name == "ShuffleNetV2_x1_5":
			from .CIFAR10.models import ShuffleNetV2
			return ShuffleNetV2.ShuffleNetV2(config_type="x1_5")
		if model_name == "ShuffleNetV2_x2_0":
			from .CIFAR10.models import ShuffleNetV2
			return ShuffleNetV2.ShuffleNetV2(config_type="x2_0")

	elif dataset == "SteeringAngle":
		if model_name == "ResNet34_regre":
			from .SteeringAngle.models import ResNet_regre
			return ResNet_regre.ResNet34_regre()
		elif model_name == "ResNet50_regre":
			from .SteeringAngle.models import ResNet_regre
			return ResNet_regre.ResNet50_regre()
		elif model_name == "ResNet101_regre":
			from .SteeringAngle.models import ResNet_regre
			return ResNet_regre.ResNet101_regre()
	elif dataset == "GTSRB":
		if model_name == "ResNet18":
			from.GTSRB.models import resnet
			return resnet.ResNet18()

# 获取模型鲁棒性预测器
classify_model_array = ["AlexNet","ResNet20", "ResNet32", "ResNet44", "ResNet56", "VGG11_BN", "VGG13_BN", "VGG16_BN", "VGG19_BN", "MobileNetV2_x0_5", "MobileNetV2_x0_75", "MobileNetV2_x1_0", "MobileNetV2_x1_4", "ShuffleNetV2_x0_5", "ShuffleNetV2_x1_0", "ShuffleNetV2_x1_5", "ShuffleNetV2_x2_0"]
regre_model_array = ["ResNet34_regre", "ResNet50_regre"]
def get_rob_predictor(dataset, model_name):
	if dataset == "CIFAR10":
		if model_name in classify_model_array:
			from .CIFAR10.models import Rob_predictor
			return Rob_predictor.Rob_predictor()
	elif dataset == "SteeringAngle":
		if model_name == "ResNet34_regre":
			from .SteeringAngle.models import Rob_predictor
			return Rob_predictor.Rob_predictor(number=512)
		elif model_name == "ResNet50_regre":
			from .SteeringAngle.models import Rob_predictor
			return Rob_predictor.Rob_predictor(number=2048)

	# 实际用不到，先填上GTSRB部分防止报错
	elif dataset == "GTSRB":
		if model_name == "ResNet18":
			from .CIFAR10.models import Rob_predictor
			return Rob_predictor.Rob_predictor()

# 获取生成式模型
def get_generative_model(dataset):
	if dataset == "CIFAR10":
		from .CIFAR10.models import BigGAN
		return BigGAN.Generator()
	elif dataset == "SteeringAngle":
		from .SteeringAngle.models import cont_cond_cnn_generator_discriminator
		return cont_cond_cnn_generator_discriminator.cont_cond_cnn_generator()
	elif dataset == "GTSRB":
		from .GTSRB.models import CDCGAN_size32
		return CDCGAN_size32.generator(128)
# 获取标签到潜向量的映射网络
def get_mapping(dataset):
	if dataset == "SteeringAngle":
		from .SteeringAngle.models import ResNet_embed
		return ResNet_embed.model_y2h(128)

# 获得VAE
def get_VAE(dataset, in_dim=208, latent_dim=2):
	if dataset == "CIFAR10":
		from .CIFAR10.models import VAE_chatGPT as VAE
		return VAE.VanillaVAE(in_dim=in_dim, latent_dim=latent_dim)

# 获得有监督的sa
def get_supervised_cnn_ae(dataset):
	if dataset == "CIFAR10":
		from .CIFAR10.models import Supervised_cnn_ae as ae
		return ae.Autoencoder()
