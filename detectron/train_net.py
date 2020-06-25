from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import verify_results
import detectron2.utils.comm as comm

from dataset import register_openlogo
from evaluator import OpenLogoDetectionEvaluator
from visualizer import visualize


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
    show = True

    register_openlogo(cfg.DATASETS.TRAIN[0], "datasets/data/openlogo", "trainval", "supervised_imageset")
    register_openlogo(cfg.DATASETS.TEST[0], "datasets/data/openlogo", "test", "supervised_imageset")
    trainer = DefaultTrainer(cfg)

    evaluator = OpenLogoDetectionEvaluator(cfg.DATASETS.TEST[0])

    if args.eval_only:

        model = trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        if show:
            visualize(cfg, amount=20)


        res = trainer.test(cfg, model, evaluators=[evaluator])

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
