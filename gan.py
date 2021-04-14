#!/usr/bin/env python

"""
Goodfellow I., Pouget-Abadie J., Mirza M., Xu B., Warde-Farley D., Ozair S., Courville A. & Bengio Yoshua. 
Generative adversarial nets. ICLR, 2014.
"""


import argparse
from src.loadopts import *

METHOD = "GAN"
VALID_EPOCHS = 20
FMT = "{description}=" \
        "={dim_latent}" \
        "={criterion_g}-{learning_policy_g}-{optimizer_g}-{lr_g}-{rtype}" \
        "={criterion_d}-{learning_policy_d}-{optimizer_d}-{lr_d}" \
        "={batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("-g", "--generator", type=str, default="gan-g")
parser.add_argument("-d", "--discriminator", type=str, default="gan-d")
parser.add_argument("--dim_latent", type=int, default=128)

# for generator
parser.add_argument("-cg", "--criterion_g", type=str, default="bce")
parser.add_argument("-og", "--optimizer_g", type=str, choices=("sgd", "adam"), default="adam")
parser.add_argument("-lrg", "--lr_g", "--LR_G", "--learning_rate_g", type=float, default=0.0002)
parser.add_argument("-lpg", "--learning_policy_g", type=str, default="null", 
                help="learning rate scheduler defined in config.py")
parser.add_argument("--rtype", type=str, default="normal",
                help="the sampling strategy")
parser.add_argument("--low", type=float, default=0.)
parser.add_argument("--high", type=float, default=1.)
parser.add_argument("--loc", type=float, default=0.)
parser.add_argument("--scale", type=float, default=1.)

# for discriminator
parser.add_argument("-cd", "--criterion_d", type=str, default="bce")
parser.add_argument("-od", "--optimizer_d", type=str, choices=("sgd", "adam"), default="adam")
parser.add_argument("-lrd", "--lr_d", "--LR_D", "--learning_rate_d", type=float, default=0.002)
parser.add_argument("-lpd", "--learning_policy_d", type=str, default="null", 
                help="learning rate scheduler defined in config.py")
parser.add_argument("--aug_policy", type=str, default="",
                help="choose augmentation policy from: color, translation and cutout")

# for evaluation
parser.add_argument("--sampling_times", type=int, default=5000)
parser.add_argument("--e_batch_size", type=int, default=16)
parser.add_argument("--e_splits", type=int, default=1)
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
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentation which will be applied in training mode.")
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
        dim_input=opts.dim_latent
    )
    arch_d = load_model(model_type=opts.discriminator)(
        in_shape=get_shape(opts.dataset)
    )
    device = gpu(arch_g, arch_d)

    # load dataset
    trainset = load_dataset(
        dataset_type=opts.dataset,
        transform=opts.transform,
        train=True
    )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset,
        batch_size=opts.batch_size,
        train=True,
        show_progress=opts.progress
    )
    normalizer = load_normalizer(dataset_type=opts.dataset)
    augmentor = load_augmentor(policy=opts.aug_policy)

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
        T_max=opts.epochs
    )
    learning_policy_d = load_learning_policy(
        optimizer=optimizer_d, 
        learning_policy_type=opts.learning_policy_d,
        T_max=opts.epochs
    )

    # load criteria
    criterion_g = load_loss_func(loss_type=opts.criterion_g)
    criterion_d = load_loss_func(loss_type=opts.criterion_d)

    sampler = load_sampler(
        rtype=opts.rtype,
        low=opts.low,
        high=opts.high,
        loc=opts.loc,
        scale=opts.scale
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
    )
    discriminator = Discriminator(
        arch=arch_d, device=device,
        criterion=criterion_d, 
        optimizer=optimizer_d,
        normalizer=normalizer,
        augmentor=augmentor,
        learning_policy=learning_policy_d
    )

    # load the inception model for FID and IS evaluation
    if opts.need_fid or opts.need_is:
        inception_model = load_inception_model(
            resize=opts.resize, normalizer=normalizer
        )
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
        cfg['start_epoch'] = load_checkpoint(
            path=cfg.info_path,
            models={"generator":generator, "discriminator":discriminator}
        )
    else:
        cfg['start_epoch'] = 0


    # load coach
    cfg['coach'] = Coach(
        generator=generator,
        discriminator=discriminator,
        device=device,
        inception_model=inception_model
    )

    return cfg, log_path



def main(
    coach, trainloader,
    start_epoch, info_path
):
    from src.utils import save_checkpoint, imagemeter
    for epoch in range(start_epoch, opts.epochs):
        if epoch % VALID_EPOCHS == 0:
            save_checkpoint(
                path=info_path,
                state_dict={
                    "generator":coach.generator.state_dict(),
                    "discriminator":coach.discriminator.state_dict(),
                    "epoch": epoch
                }
            )

            imgs = coach.generator.evaluate(batch_size=10)
            fp = imagemeter(imgs)
            writter.add_figure(f"Image-Epoch:{epoch}", fp, global_step=epoch)

            fid_score, is_score = coach.evaluate(
                dataset_type=opts.dataset,
                n=opts.sampling_times,
                batch_size=opts.e_batch_size,
                n_splits=opts.e_splits,
                need_fid=opts.need_fid,
                need_is=opts.need_is
            )

            writter.add_scalar("FID", fid_score, epoch)
            writter.add_scalar("IS", is_score, epoch)

            print(f">>> Current FID score: {fid_score:.6f}")
            print(f">>> Current IS  score: {is_score:.6f}")

        loss_g, loss_d, validity = coach.train(trainloader, epoch=epoch)
        writter.add_scalars("Loss", {"generator":loss_g, "discriminator":loss_d}, epoch)
        writter.add_scalar("Validity", validity, epoch)

    imgs = coach.generator.evaluate(batch_size=10)
    fp = imagemeter(imgs)
    writter.add_figure(f"Image-Epoch:{epoch}", fp, global_step=epoch)


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


    






    
