from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import verify_results
import detectron2.utils.comm as comm

from data.dataset import register_openlogo


def setup(args):

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="openlogo")

    return cfg


def main(args):

    cfg = setup(args)
    register_openlogo("openlogo_train", "data/datasets/openlogo", "train", "supervised_imageset")
    register_openlogo("openlogo_val", "data/datasets/openlogo", "val", "supervised_imageset")
    trainer = DefaultTrainer(cfg)

    if args.eval_only:

        model = trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = trainer.test(cfg, model)

        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(trainer.test_with_TTA(cfg, model))

        return res

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    return trainer.train()



if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
