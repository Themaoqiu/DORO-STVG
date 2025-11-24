"""VidSTG数据集适配器 - 支持JSON格式"""
import json
from pathlib import Path
from typing import List, Dict
from .base_dataset import BaseSTVGDataset
from ..core.schema import STVGSample


class VidSTGDataset(BaseSTVGDataset):
    """VidSTG数据集适配器"""
    
    def _load_annotations(self) -> List[dict]:
        """
        加载VidSTG的JSON标注文件
        
        支持两种路径格式:
        1. 单个JSON文件: /path/to/vidstg_test.json
        2. 包含多个JSON的目录: /path/to/annotations/ (会读取所有.json文件)
        """
        annotations = []
        
        if self.annotation_path.is_file() and self.annotation_path.suffix == '.json':
            # 单个JSON文件
            print(f"[VidSTG] Loading single JSON file: {self.annotation_path}")
            with open(self.annotation_path, 'r') as f:
                data = json.load(f)
                # 判断是列表还是单个对象
                if isinstance(data, list):
                    annotations.extend(data)
                else:
                    annotations.append(data)
        
        elif self.annotation_path.is_dir():
            # 目录: 读取所有JSON文件
            json_files = list(self.annotation_path.glob('*.json'))
            print(f"[VidSTG] Loading {len(json_files)} JSON files from: {self.annotation_path}")
            
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        annotations.extend(data)
                    else:
                        annotations.append(data)
        else:
            raise ValueError(f"Invalid annotation path: {self.annotation_path}")
        
        print(f"[VidSTG] Loaded {len(annotations)} raw annotations")
        return annotations
    
    def _parse_to_standard_format(self) -> List[STVGSample]:
        """
        将VidSTG JSON格式转换为标准格式
        
        VidSTG JSON结构:
        {
            "video_id": "5159741010",
            "video_path": "1025/5159741010.mp4",
            "frame_count": 219,
            "fps": 29.97,
            "trajectories": [[{bbox, tid}, ...], ...],  # 每帧的bbox列表
            "relation_instances": [                      # 关系实例
                {
                    "subject_tid": 0,
                    "object_tid": 1,
                    "predicate": "in_front_of",
                    "begin_fid": 0,
                    "end_fid": 210
                }
            ]
        }
        """
        samples = []
        
        for video_data in self.raw_data:
            video_id = video_data['video_id']
            
            # 为每个relation_instance创建一个样本
            for rel_idx, rel_inst in enumerate(video_data.get('relation_instances', [])):
                item_id = f"{video_id}_{rel_idx}"
                
                # 提取时间边界
                begin_fid = rel_inst['begin_fid']
                end_fid = rel_inst['end_fid']
                
                # 提取对应的bboxes
                subject_tid = rel_inst['subject_tid']
                object_tid = rel_inst['object_tid']
                predicate = rel_inst['predicate']
                
                # 构建bbox字典 {frame_id: [[x1,y1,x2,y2]]}
                gt_bboxes = self._extract_bboxes(
                    video_data['trajectories'],
                    subject_tid,
                    begin_fid,
                    end_fid
                )
                
                # 构建查询文本
                subject_cat = self._get_category(video_data['subject/objects'], subject_tid)
                object_cat = self._get_category(video_data['subject/objects'], object_tid)
                query = f"{subject_cat} {predicate} {object_cat}"
                
                # 创建标准样本
                sample = STVGSample(
                    item_id=item_id,
                    video_path=self._get_video_path(video_data['video_path']),
                    query=query,
                    gt_temporal_bound=(begin_fid, end_fid),
                    gt_bboxes=gt_bboxes,
                    metadata={
                        'video_id': video_id,
                        'video_name': video_id,
                        'dataset': 'vidstg',
                        'qtype': self._infer_qtype(predicate),  # 根据谓词推断问题类型
                        'predicate': predicate,
                        'subject_tid': subject_tid,
                        'object_tid': object_tid,
                        'fps': video_data.get('fps', 30.0),
                        'frame_count': video_data.get('frame_count', 0)
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _extract_bboxes(
        self, 
        trajectories: List[List[Dict]], 
        tid: int, 
        begin_fid: int, 
        end_fid: int
    ) -> Dict[int, List[List[float]]]:
        """
        从trajectories中提取指定tid和时间范围的bboxes
        
        Args:
            trajectories: 每帧的bbox列表
            tid: trajectory ID
            begin_fid: 起始帧
            end_fid: 结束帧
            
        Returns:
            {frame_id: [[x1, y1, x2, y2]]}
        """
        bboxes = {}
        
        for fid in range(begin_fid, end_fid):
            if fid >= len(trajectories):
                break
            
            frame_boxes = trajectories[fid]
            for box_info in frame_boxes:
                if box_info['tid'] == tid:
                    bbox = box_info['bbox']
                    # 转换为 [x1, y1, x2, y2] 格式
                    bboxes[fid] = [[
                        bbox['xmin'],
                        bbox['ymin'],
                        bbox['xmax'],
                        bbox['ymax']
                    ]]
                    break
        
        return bboxes
    
    def _get_category(self, subject_objects: List[Dict], tid: int) -> str:
        """获取tid对应的类别名称"""
        for obj in subject_objects:
            if obj['tid'] == tid:
                return obj['category']
        return 'object'
    
    def _infer_qtype(self, predicate: str) -> str:
        """
        根据谓词推断问题类型
        
        VidSTG有两种类型:
        - declarative: 陈述句 (e.g., "person in_front_of bicycle")
        - interrogative: 疑问句 (可能需要额外标注)
        """
        # 简单策略: 默认为declarative
        # 实际使用时可根据数据集的具体标注调整
        return 'declarative'
    
    def _get_video_path(self, relative_path: str) -> str:
        """
        获取完整视频路径
        
        Args:
            relative_path: 例如 "1025/5159741010.mp4"
            
        Returns:
            完整路径
        """
        video_path = self.video_dir / relative_path
        
        if not video_path.exists():
            print(f"[Warning] Video not found: {video_path}")
        
        return str(video_path)