
import argparse
import logging
import os.path as osp
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.registry import RUNNERS
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    '''
    # catseg_vitl14
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/catseg_vitl14_4xb2-80k_globe230k-512x512.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/catseg_clipvitl14_frozenall_224',help='the dir to save logs and models')

    # lseg_vitl14
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/lseg_vitl14_4xb2-80k_globe230k-512x512.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/lseg_clipvitl14_frozenall_224',help='the dir to save logs and models')

    # san_vitl14
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/san_vitl14_4xb2-80k_globe230k-512x512.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/san_clipvitl14',help='the dir to save logs and models')
   
    # fusioner_vitl14
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/fusioner_vitl14_4xb2-80k_globe230k-512x512.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/fusioner_clipvitl14',help='the dir to save logs and models')
 
    # HG-RSOVSSeg_UnHGU_UnTAM_vitl14
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/HG-RSOVSSeg_unhgu_untam_vitl14_4xb2-80k_globe230k-512x512.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/HG-RSOVSSeg_UnHGU_UnTAM_clipvitl14_frozenall',help='the dir to save logs and models')
    
    # HG-RSOVSSeg_UnTAM_vitl14
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/HG-RSOVSSeg_untam_vitl14_4xb2-80k_globe230k-512x512.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/HG-RSOVSSeg_UnTAM_clipvitl14_frozenall',help='the dir to save logs and models')
    
    # HG-RSOVSSeg_vitb16
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/HG-RSOVSSeg_vitb16_4xb2-80k_globe230k-512x512.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/HG-RSOVSSeg_clipvitb16_frozenall_512_128-64/',help='the dir to save logs and models')

    # HG-RSOVSSeg_vitb32
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/HG-RSOVSSeg_vitb32_4xb2-80k_globe230k-512x512.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/HG-RSOVSSeg_clipvitb32_frozenall_512_128-64/',help='the dir to save logs and models')
    
    # HG-RSOVSSeg_vitl14_10
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/HG-RSOVSSeg_vitl14_4xb2-80k_globe230k-512x512_10.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/HG-RSOVSSeg_clipvitl14_frozenall_512_128-64_10/',help='the dir to save logs and models')
    
    # HG-RSOVSSeg_vitl14_30
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/HG-RSOVSSeg_vitl14_4xb2-80k_globe230k-512x512_30.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/HG-RSOVSSeg_clipvitl14_frozenall_512_128-64_30/',help='the dir to save logs and models')

    # HG-RSOVSSeg_vitl14_50
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/HG-RSOVSSeg_vitl14_4xb2-80k_globe230k-512x512_50.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/HG-RSOVSSeg_clipvitl14_frozenall_512_128-64_50/',help='the dir to save logs and models')
    '''
    # HG-RSOVSSeg_vitl14
    parser.add_argument('--config', default='code/mmsegmentation/projects/HG-RSOVSSeg/configs/Globe230k_my_model_512/HG-RSOVSSeg_vitl14_4xb2-80k_globe230k-512x512.py',help='train config file path')
    parser.add_argument('--work-dir', default='result/HG-RSOVSSeg/Globe230k_512/HG-RSOVSSeg_clipvitl14_frozenall_512_128-64/',help='the dir to save logs and models')

    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument('--amp', action='store_true', default=False, help='enable automatic-mixed-precision training')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file. If the value to '
                             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                             'Note that the quotation marks are necessary and that no white space '
                             'is allowed.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    current_directory = os.getcwd()
    # load config
    cfg = Config.fromfile(os.path.join(current_directory, args.config))
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = os.path.join(current_directory, args.work_dir)
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(os.path.join(current_directory, args.config)))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
