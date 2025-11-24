"""评测框架主入口"""
import argparse
import logging
from pathlib import Path

from runner import run_from_config


def setup_logger(output_dir: Path) -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger('STVG_Eval')
    logger.setLevel(logging.INFO)
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 文件输出
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / 'eval.log')
    file_handler.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def main():
    parser = argparse.ArgumentParser(
        description='STVG MLLM Evaluation Framework'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory in config'
    )
    
    args = parser.parse_args()
    
    # 读取配置
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖输出目录
    if args.output_dir:
        config['evaluation']['output_dir'] = args.output_dir
    
    output_dir = Path(config['evaluation']['output_dir'])
    
    # 设置日志
    logger = setup_logger(output_dir)
    logger.info(f"Starting evaluation with config: {args.config}")
    
    # 运行评测
    try:
        metrics = run_from_config(args.config, logger=logger)
        logger.info("Evaluation completed successfully!")
        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()