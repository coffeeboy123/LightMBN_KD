import data_v1
import data_v2
from loss import make_loss
from model import make_model_student, make_model_teacher
from optim import make_optimizer, make_scheduler

# import engine_v1
# import engine_v2
import engine_v3
import os.path as osp
from option import args
import utils.utility as utility
from utils.model_complexity import compute_model_complexity
from torch.utils.collect_env import get_pretty_env_info
import yaml
import torch

def count_params(module, name=""):
    total = 0
    for n, p in module.named_parameters():
        if p.requires_grad:
            total += p.numel()
    print(f"{name}: {total:,}")
    return total


if __name__ == '__main__':
    if args.config != "":
        with open(args.config, "r") as f:
            config = yaml.full_load(f)
        for op in config:
            setattr(args, op, config[op])
    torch.backends.cudnn.benchmark = True

    # loader = data.Data(args)
    ckpt = utility.checkpoint(args)
    loader = data_v2.ImageDataManager(args)
    model_student = make_model_student(args, ckpt)
    model_teacher = make_model_teacher(args, ckpt)
    model_teacher.eval()
    for param in model_teacher.parameters():
        param.requires_grad = False
    optimzer = make_optimizer(args, model_student)
    loss = make_loss(args, ckpt)

    if args.teacher_pretrain != "":
        ckpt.load_pretrained_weights(model_teacher, args.teacher_pretrain)

    total_params = sum(p.numel() for p in model_student.parameters() if p.requires_grad)

    # 특징 추출기 파라미터만 따로 계산
    feature_params = 0
    feature_params += count_params(model_student.backone, "backone")
    feature_params += count_params(model_student.global_branch, "global_branch")
    feature_params += count_params(model_student.partial_branch, "partial_branch")
    feature_params += count_params(model_student.channel_branch, "channel_branch")
    feature_params += count_params(model_student.shared, "shared")
    feature_params += count_params(model_student.batch_drop_block.drop_batch_bottleneck, "drop_batch_bottleneck")

    print(f"\n[총 파라미터 수]: {total_params:,}")
    print(f"[특징 추출기 파라미터 수]: {feature_params:,}")

    start = -1
    if args.load != "":
        start, model, optimizer = ckpt.resume_from_checkpoint(
            osp.join(ckpt.dir, "model-latest.pth"), model_student, optimzer
        )
        start = start - 1
    if args.pre_train != "":
        ckpt.load_pretrained_weights(model_student, args.pre_train)

    scheduler = make_scheduler(args, optimzer, start)

    # print('[INFO] System infomation: \n {}'.format(get_pretty_env_info()))
    ckpt.write_log(
        "[INFO] Model parameters: {com[0]} flops: {com[1]}".format(
            com=compute_model_complexity(model_student, (1, 3, args.height, args.width))
        )
    )

    engine = engine_v3.Engine(args, model_student, model_teacher, optimzer, scheduler, loss, loader, ckpt)
    # engine = engine.Engine(args, model, loss, loader, ckpt)

    n = start + 1
    while not engine.terminate():
        n += 1
        engine.train()
        if args.test_every != 0 and n % args.test_every == 0:
            engine.test()
        elif n == args.epochs:
            engine.test()
