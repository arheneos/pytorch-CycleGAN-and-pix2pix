import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

def robust_log_l1(pred, target):
    s_pred = torch.sign(pred) * torch.log1p(torch.abs(pred))
    s_target = torch.sign(target) * torch.log1p(torch.abs(target))
    return torch.nn.functional.mse_loss(s_pred, s_target)


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        A domain: HR simulation data (256x256), from group['mask'][:] in HDF5.
        B domain: LR real AFM data (64x64), from .npy files.
        G_A (down_4x): HR sim (256) -> LR real-style (64)
        G_B (sr_4x):   LR real (64) -> HR sim-style (256)  <-- the upscaler
        Identity loss is disabled (asymmetric sizes make it ill-defined).
        """
        parser.set_defaults(no_dropout=True)
        parser.add_argument("--netG_A", type=str, default="down_4x",
                            help="Generator A architecture (HR->LR): down_4x | afm_optimized | ...")
        parser.add_argument("--netG_B", type=str, default="sr_4x",
                            help="Generator B architecture (LR->HR): sr_4x | afm_optimized | ...")
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.0,
                help="Identity loss weight. Must be 0 for asymmetric SR setup (A and B have different sizes).",
            )

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        use_idt = self.isTrain and opt.lambda_identity > 0.0
        if use_idt:
            self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]
        else:
            self.loss_names = ["D_A", "G_A", "cycle_A", "D_B", "G_B", "cycle_B"]
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if use_idt:
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # G_A: HR sim (256) -> LR real-style (64)
        # G_B: LR real (64) -> HR sim-style (256)  <-- the upscaler
        netG_A = getattr(opt, "netG_A", opt.netG)
        netG_B = getattr(opt, "netG_B", opt.netG)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, netG_A, opt.norm, not opt.no_dropout,
                                        opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, netG_B, opt.norm, not opt.no_dropout,
                                        opt.init_type, opt.init_gain)

        if self.isTrain:  # define discriminators
            # D_A: real_B (LR 64) vs fake_B (LR 64)
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type,
                                            opt.init_gain)
            # D_B: real_A (HR 256) vs fake_A (HR 256)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type,
                                            opt.init_gain)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                raise ValueError("lambda_identity must be 0 for the asymmetric SR setup (A=HR 256, B=LR 64).")
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # AFM 데이터의 구조 보존을 위해 StructuralLoss(SSIM + L1) 사용
            self.criterionCycle = networks.StructuralLoss(alpha=0.5).to(self.device)
            self.criterionIdt = networks.StructuralLoss(alpha=0.5).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        if self.isTrain:
            noise_std = 0.01  # 노이즈 강도 설정
            noise_B = torch.randn_like(self.fake_B) * noise_std
            self.rec_A = self.netG_B(self.fake_B + noise_B)  # G_B(G_A(A) + noise)
        else:
            self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        if self.isTrain:
            noise_std = 0.01
            noise_A = torch.randn_like(self.fake_A) * noise_std
            self.rec_B = self.netG_A(self.fake_A + noise_A)  # G_A(G_B(B) + noise)
        else:
            self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        if isinstance(pred_real, list):  # Multiscale case
            loss_D_real = 0
            for pred in pred_real:
                loss_D_real += self.criterionGAN(pred, True)
            loss_D_real /= len(pred_real)
        else:
            loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = netD(fake.detach())
        if isinstance(pred_fake, list):  # Multiscale case
            loss_D_fake = 0
            for pred in pred_fake:
                loss_D_fake += self.criterionGAN(pred, False)
            loss_D_fake /= len(pred_fake)
        else:
            loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A, self.loss_idt_B = 0, 0

        # GAN loss
        pred_fake_B = self.netD_A(self.fake_B)
        if isinstance(pred_fake_B, list):
            self.loss_G_A = 0
            for pred in pred_fake_B:
                self.loss_G_A += self.criterionGAN(pred, True)
            self.loss_G_A /= len(pred_fake_B)
        else:
            self.loss_G_A = self.criterionGAN(pred_fake_B, True)

        pred_fake_A = self.netD_B(self.fake_A)
        if isinstance(pred_fake_A, list):
            self.loss_G_B = 0
            for pred in pred_fake_A:
                self.loss_G_B += self.criterionGAN(pred, True)
            self.loss_G_B /= len(pred_fake_A)
        else:
            self.loss_G_B = self.criterionGAN(pred_fake_A, True)
        # Cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        # NaN 체크 후 backward
        if torch.isnan(self.loss_G) or torch.isinf(self.loss_G):
            return False

        self.loss_G.backward()
        return True

    def optimize_parameters(self):
        """Update network weights; handling NaNs and OOM"""
        self.forward()

        # --- Generator 업데이트 ---
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()

        if self.backward_G():  # 정상적인 경우
            # [핵심] Gradient Clipping으로 폭주 방지
            torch.nn.utils.clip_grad_norm_(self.netG_A.parameters(), max_norm=0.25)
            torch.nn.utils.clip_grad_norm_(self.netG_B.parameters(), max_norm=0.25)
            self.optimizer_G.step()
        else:  # NaN 발생 시
            print("NaN detected in G! Clearing memory...")
            self.optimizer_G.zero_grad()
            self._clear_tensors()  # 아래 정의될 메모리 해제 함수
            return

        # --- Discriminator 업데이트 ---
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()

        self.backward_D_A()
        self.backward_D_B()

        torch.nn.utils.clip_grad_norm_(self.netD_A.parameters(), max_norm=0.25)
        torch.nn.utils.clip_grad_norm_(self.netD_B.parameters(), max_norm=0.25)
        self.optimizer_D.step()

    def _clear_tensors(self):
        """OOM 방지를 위해 살아있는 텐서 참조 해제"""
        self.fake_A = self.fake_B = self.rec_A = self.rec_B = None
        self.idt_A = self.idt_B = None
        self.loss_G = None
        torch.cuda.empty_cache()
