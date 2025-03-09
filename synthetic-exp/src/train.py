import os
from random import randint
import uuid
from functools import partial
import torch.nn.functional as F

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics, get_model_from_run
from tasks import get_task_sampler
from samplers import get_data_sampler, sample_transformation
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb

torch.backends.cudnn.benchmark = True


def funca(model, ys, ys_base, args, inds, xs):
    output = model(xs, ys, ys_base, args, inds=torch.arange(ys.shape[1], ys.shape[1]*2))
    return output[:, -1]

    
def train_step(model, xs, ys, ys_base, optimizer, loss_func, args):
    optimizer.zero_grad()
    output = model(xs, ys, ys_base, args, inds = torch.arange(ys.shape[1], ys.shape[1]*2))
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    base_task_sampler = get_task_sampler(
        'linear_regression',
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples
    
    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args, 
        ) # [BZ, SEQ LEN, MODEL DIM (n_dims_truncated:) = 0]
        
        pad_n_points = 41 - curriculum.n_points
        if pad_n_points > 0:
            pad_zeros = torch.zeros(bsize, pad_n_points, n_dims)
            xs = torch.cat([xs, pad_zeros], dim=1)
        
        task = task_sampler(**task_sampler_args)
        base_task = base_task_sampler(**task_sampler_args) # sample linear regression W1

        ys_base = base_task.evaluate(xs)
        ys = task.evaluate(xs, base_task.w_b)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.cuda(), ys.cuda(), ys_base.cuda(), optimizer, loss_func, args)

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
        
            ys_base = base_task.evaluate(xs)
            ys = task.evaluate(xs, base_task.w_b)
            

            total_inner_product = 0
            for each_sample in range(bsize):
                ysclone = ys[each_sample].unsqueeze(0).cuda()
                ys_baseclone = ys_base[each_sample].unsqueeze(0).cuda()
                jac_func = partial(funca, model, ysclone.cuda(), ys_baseclone, args, torch.arange(ysclone.shape[1], ysclone.shape[1]*2))
                xsclone = xs[each_sample].unsqueeze(0).clone().cuda().requires_grad_(True)
                with torch.enable_grad():
                    jacobian = torch.autograd.functional.jacobian(jac_func, xsclone)

                jacobian = F.normalize(jacobian[0, 0], p=2, dim=1)

                intermediate = 2*(xs.cuda() @ base_task.w_b.cuda()) 
                projection = base_task.w_b.cuda().squeeze(-1) + intermediate[:, -1, 0].view(intermediate.size(0), 1) * base_task.w_b.cuda().squeeze(-1) 
                projection = projection.unsqueeze(-1) 
                projection = F.normalize(projection[each_sample, :], p=2, dim=0)
                inner_prod = jacobian @ projection.cuda()
                inner_prod = inner_prod[-1, 0].item()
                total_inner_product += inner_prod
            inner_prod = total_inner_product / bsize

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "inner_product": inner_prod,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )
    
    if args.training.pretrained is not None:
        pretrained_model, pretrained_conf = get_model_from_run(args.training.pretrained)
        model = pretrained_model
    else:
        model = build_model(args.model)
    model.cuda()
    model.train()
    train(model, args)
    
    if not args.test_run:
        _ = get_run_metrics(args.out_dir, args, skip_baselines=True)  # precompute metrics for eval, skip_baselines=True


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
