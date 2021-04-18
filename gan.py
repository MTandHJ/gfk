#!/usr/bin/env python



import argparse
from src.loadopts import *

METHOD = "GAN"
SAVE_FREQ = 2000
PRINT_FREQ = 200
VALID_FREQ = 2000
FMT = "{description}=" \
        "={dim_latent}-{acml_per_step}" \
        "={criterion_g}-{learning_policy_g}-{lr_g}-{steps_per_G}-{rtype}" \
        "={criterion_d}-{learning_policy_d}-{lr_d}-{steps_per_D}-{aug_policy}" \
        "={batch_size}"

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("-h5", "--hdf5", action="store_false", default=True,
                help="whether to load hdf5 dataset")
parser.add_argument("-m2m", "--mv2memory", action="store_true", default=False,
                help="whether to move the total data to memory")

# for model
parser.add_argument("-g", "--generator", type=str, default="dcgan-g")
parser.add_argument("-d", "--discriminator", type=str, default="dcgan-d")
parser.add_argument("--dim_latent", type=int, default=128)
parser.add_argument("-acml", "--acml_per_step", type=int, default=1,
                help="accumulative iterations per step")
parser.add_argument("--init_policy", choices=("ortho", "N02", "xavier", "kaiming"), default="N02",
                help="initialize the model")

# for generator
parser.add_argument("-cg", "--criterion_g", type=str, default="hinge")
parser.add_argument("-og", "--optimizer_g", type=str, choices=("sgd", "adam"), default="adam")
parser.add_argument("-lrg", "--lr_g", "--LR_G", "--learning_rate_g", type=float, default=0.0002)
parser.add_argument("-lpg", "--learning_policy_g", type=str, default="null", 
                help="learning rate scheduler defined in config.py")
parser.add_argument("-sng", "--need_sn_g", action="store_false", default=True,
                help="whether adopting spectral norm for generator")
parser.add_argument("--ema", action="store_false", default=True, help="exponential moving average")
parser.add_argument("--ema_mom", type=float, default=0.9999)
parser.add_argument("--ema_warmup", type=int, default=1000)
parser.add_argument("-spg", "--steps_per_G", type=int, default=1,
                help="total steps per G training procedure")

# for sampiling policy
parser.add_argument("--rtype", type=str, choices=("uniform", "normal", "tnormal"), 
                default="tnormal", help="the sampling strategy")
parser.add_argument("--low", type=float, default=-1., help="for uniform")
parser.add_argument("--high", type=float, default=1., help="for uniform")
parser.add_argument("--loc", type=float, default=0., help="for normal")
parser.add_argument("--scale", type=float, default=1., help="for normal")
parser.add_argument("--threshold", type=float, default=.5, help="for truncated normal")

# for discriminator
parser.add_argument("-cd", "--criterion_d", type=str, default="hinge")
parser.add_argument("-od", "--optimizer_d", type=str, choices=("sgd", "adam"), default="adam")
parser.add_argument("-lrd", "--lr_d", "--LR_D", "--learning_rate_d", type=float, default=0.0002)
parser.add_argument("-lpd", "--learning_policy_d", type=str, default="null", 
                help="learning rate scheduler defined in config.py")
parser.add_argument("-snd", "--need_sn_d", action="store_false", default=True,
                help="whether adopting spectral norm for discriminator")
parser.add_argument("--aug_policy", choices=("null", "diff_aug"), default="diff_aug",
                help="choose augmentation policy from: color, translation and cutout")
parser.add_argument("-spd", "--steps_per_D", type=int, default=2,
                help="total steps per D training procedure")

# for evaluation
parser.add_argument("--sampling_times", type=int, default=10000)
parser.add_argument("--e_batch_size", type=int, default=64)
parser.add_argument("--e_splits", type=int, default=10)
parser.add_argument("--resize", action="store_false", default=True)
parser.add_argument("--need_fid", action="store_false", default=True)
parser.add_argument("--need_is", action="store_false", default=True)

# basic settings
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.5,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=0.,
                help="weight decay")
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="GAN")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



def load_cfg():
    from src.base import Coach
    from models.base import Generator, Discriminator
    from src.dict2obj import Config
    from src.utils import gpu, load_checkpoint, set_seed

    cfg = Config()
    set_seed(opts.seed)

    # load model
    arch_g = load_model(model_type=opts.generator)(
        out_shape=get_shape(opts.dataset),
        dim_latent=opts.dim_latent
    )
    arch_d = load_model(model_type=opts.discriminator)(
        in_shape=get_shape(opts.dataset)
    )
    device = gpu(arch_g, arch_d)
    # initialization and spectral norm
    refine_model(arch_g, init_policy=opts.init_policy, need_sn=opts.need_sn_g)
    refine_model(arch_d, init_policy=opts.init_policy, need_sn=opts.need_sn_d)

    # load dataset
    trainset = load_dataset(
        dataset_type=opts.dataset,
        mode='train',
        hdf5=opts.hdf5,
        mv2memory=opts.mv2memory
    )
    trainloader = load_dataloader(
        dataset=trainset,
        batch_size=opts.batch_size,
        shuffle=True,
        show_progress=opts.progress
    )
    augmentor = load_augmentor(aug_policy=opts.aug_policy)

    # load optimizer and correspoding learning policy
    optimizer_g = load_optimizer(
        model=arch_g, optim_type=opts.optimizer_g, lr=opts.lr_g,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay
    )
    optimizer_d = load_optimizer(
        model=arch_d, optim_type=opts.optimizer_d, lr=opts.lr_d,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay
    )
    learning_policy_g = load_learning_policy(
        optimizer=optimizer_g, 
        learning_policy_type=opts.learning_policy_g,
        T_max=opts.steps
    )
    learning_policy_d = load_learning_policy(
        optimizer=optimizer_d, 
        learning_policy_type=opts.learning_policy_d,
        T_max=opts.steps
    )

    # load criteria
    criterion_g = load_loss_func(loss_type=opts.criterion_g, mode="gen").to(device)
    criterion_d = load_loss_func(loss_type=opts.criterion_d, mode="dis").to(device)

    sampler = load_sampler(
        rtype=opts.rtype,
        low=opts.low,
        high=opts.high,
        loc=opts.loc,
        scale=opts.scale,
        threshold=opts.threshold
    )

    # load generator
    generator = Generator(
        arch=arch_g, 
        device=device,
        sampler=sampler,
        dim_latent=opts.dim_latent, 
        criterion=criterion_g,
        optimizer=optimizer_g,
        learning_policy=learning_policy_g,
        ema=opts.ema,
        mom=opts.ema_mom,
        warmup_steps=opts.ema_warmup
    )
    discriminator = Discriminator(
        arch=arch_d, device=device,
        criterion=criterion_d, 
        optimizer=optimizer_d,
        augmentor=augmentor,
        learning_policy=learning_policy_d
    )

    # load the inception model for FID and IS evaluation
    if opts.need_fid or opts.need_is:
        inception_model = load_inception_model(resize=opts.resize)
    else:
        inception_model = None

    # generate the path for logging information and saving parameters
    cfg['info_path'], log_path = generate_path(
        method=METHOD, dataset_type=opts.dataset,
        generator=opts.generator,
        discriminator=opts.discriminator,
        description=opts.description
    )
    
    if opts.resume:
        cfg['start_step'] = load_checkpoint(
            path=cfg.info_path,
            models={"generator":generator, "discriminator":discriminator}
        )
    else:
        cfg['start_step'] = 0


    # load coach
    cfg['coach'] = Coach(
        generator=generator,
        discriminator=discriminator,
        inception_model=inception_model,
        trainloader=trainloader,
        device=device
    )

    return cfg, log_path


def evaluate(coach, step):
    from src.utils import imagemeter, tensor2img
    imgs = coach.generator.evaluate(batch_size=10)
    imgs = tensor2img(imgs)
    fp = imagemeter(imgs)
    writter.add_figure(f"Image-Step:{step}", fp, global_step=step)

    fid_score, is_score = coach.evaluate(
        dataset_type=opts.dataset,
        n=opts.sampling_times,
        batch_size=opts.e_batch_size,
        n_splits=opts.e_splits,
        need_fid=opts.need_fid,
        need_is=opts.need_is
    )
    writter.add_scalar("FID", fid_score, step)
    writter.add_scalar("IS", is_score, step)

    print(f">>> Current FID score: {fid_score:.6f}")
    print(f">>> Current IS  score: {is_score:.6f}")

def main(coach, start_step, info_path):
    from src.utils import save_checkpoint
    for freq, step in enumerate(range(start_step, opts.steps, opts.steps_per_G)):
        if freq % SAVE_FREQ == 0:
            save_checkpoint(
                path=info_path,
                state_dict={
                    "generator":coach.generator.state_dict(),
                    "discriminator":coach.discriminator.state_dict(),
                    "step": step
                }
            )
        
        if freq % PRINT_FREQ == 0:
            coach.progress.display(step=step)
            coach.progress.step()
        
        if freq % VALID_FREQ == 0:
            evaluate(coach, step)
            

        loss_g, loss_d= coach.train(
            batch_size=opts.batch_size,
            steps_per_G=opts.steps_per_G,
            steps_per_D=opts.steps_per_D,
            acml_per_step=opts.acml_per_step,
            step=step
        )
        writter.add_scalars("Loss", {"generator":loss_g, "discriminator":loss_d}, step)
    
    evaluate(coach, opts.steps)

    

if __name__ ==  "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(cfg.info_path, log_path)
    readme(cfg.info_path, opts)
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    cfg['coach'].save(cfg.info_path)
    writter.close()


    






    
