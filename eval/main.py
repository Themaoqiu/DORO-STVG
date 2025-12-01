import argparse
import logging
from pathlib import Path

from eval.runner import run_from_config


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger('STVG_Eval')
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / 'eval.log')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.propagate = False
    
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
    
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.output_dir:
        config['evaluation']['output_dir'] = args.output_dir
    
    model_name = config['model']['name']
    output_dir_template = config['evaluation']['output_dir']
    output_dir_str = output_dir_template.format(model_name=model_name)
    output_dir = Path(output_dir_str)
    
    logger = setup_logger(output_dir)
    logger.info(f"Starting evaluation with config: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        metrics = run_from_config(args.config, logger=logger)
        logger.info("Evaluation completed successfully!")
        return metrics
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()