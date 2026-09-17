import os
import logging
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import wandb
import cv2
from torchvision import transforms
from PIL import Image
import random
from collections import OrderedDict
import threading
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import psutil

# 캐시 데이터는 메모리(RAM)에 저장됩니다.
# self.frame_cache = OrderedDict()에 비디오 프레임이 텐서 형태로 저장됩니다.
# 디스크에는 저장되지 않고 프로그램 종료시 캐시가 삭제됩니다.

class CustomVideoDataset(Dataset):
    """
    Custom 비디오 데이터셋 클래스
    비디오 프레임을 로드하고 전처리하는 데이터셋 구현
    네트워크 지연을 최소화하기 위한 효율적인 캐싱 시스템 포함
    학습 중 백그라운드 멀티프로세스 캐싱 지원
    """
    def __init__(self, cfg, mode, is_train, *, img_size=(64, 32), max_frames=60):
        """
        Args:
            cfg: 설정 객체
            mode: 'train', 'valid', 'test' 중 하나
            is_train: 학습 모드 여부
        """
        self.cfg = cfg
        self.mode = mode
        self.is_train = is_train
        
        # 데이터셋 로드 로깅
        logging.info(f"Load Dataset: {mode}")
        
        # 추론 모드에서는 wandb를 초기화하지 않음
        
        self.img_size = img_size
        
        # 최대 프레임 수 설정
        self.max_frames = max_frames
        
        # YAML 설정을 통한 캐싱 시스템 설정
        cache_config = getattr(cfg.TRAIN, 'CACHE', None)
        self.cache_enabled = (
            is_train and  # 검증/테스트 시에는 캐싱 비활성화
            cache_config is not None and 
            getattr(cache_config, 'ENABLED', False)
        )
        
        if self.cache_enabled:
            print(f"🔧 Cache system enabled: {mode} mode")
            self.max_cache_size = getattr(cache_config, 'MAX_SIZE', 32)
            self.initial_cache_ratio = getattr(cache_config, 'INITIAL_RATIO', 0.1)
            self.background_cache_enabled = getattr(cache_config, 'BACKGROUND_ENABLED', False)
            self.cleanup_threshold = getattr(cache_config, 'CLEANUP_THRESHOLD', 0.85)
            self.force_reset_threshold = getattr(cache_config, 'FORCE_RESET_THRESHOLD', 0.9)
            print(f"   📦 Max cache size: {self.max_cache_size}")
            print(f"   🎯 Initial cache ratio: {self.initial_cache_ratio*100:.1f}%")
            print(f"   🔄 Background caching: {'on' if self.background_cache_enabled else 'off'}")
        else:
            print(f"🚫 Cache system disabled: {mode} mode - saving memory")
            self.max_cache_size = 0
            self.initial_cache_ratio = 0
            self.background_cache_enabled = False
            self.cleanup_threshold = 0.85
            self.force_reset_threshold = 0.9
        
        self.frame_cache = OrderedDict()  # LRU 캐시
        self.cache_lock = threading.Lock()  # 멀티스레딩 안전성
        
        # 백그라운드 캐싱 시스템
        self.cache_queue = queue.Queue(maxsize=100)  # 캐싱 요청 큐
        self.cache_executor = None
        self.background_cache_running = False
        
        # 캐싱 통계 변수
        self.cache_hits = 0
        self.cache_misses = 0
        self.background_cache_loads = 0
        self.cache_loading_times = []
        self.start_time = time.time()
        
        # 기본 이미지 변환 설정
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 학습 시 데이터 증강 적용
        if is_train and cfg.AUGMENTATION.ENABLED:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5) if cfg.AUGMENTATION.RANDOM_FLIP else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(
                    brightness=cfg.AUGMENTATION.COLOR_JITTER.BRIGHTNESS,
                    contrast=cfg.AUGMENTATION.COLOR_JITTER.CONTRAST,
                    saturation=cfg.AUGMENTATION.COLOR_JITTER.SATURATION,
                    hue=cfg.AUGMENTATION.COLOR_JITTER.HUE
                ) if cfg.AUGMENTATION.COLOR_JITTER.ENABLED else transforms.Lambda(lambda x: x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 시간적 증강 설정 추가
        temporal_aug_config = getattr(cfg.AUGMENTATION, 'TEMPORAL_AUGMENTATION', None)
        self.temporal_augmentation_enabled = (
            is_train and 
            temporal_aug_config is not None and 
            getattr(temporal_aug_config, 'ENABLED', False)
        )
        
        if self.temporal_augmentation_enabled:
            self.frame_drop_ratio = getattr(temporal_aug_config, 'FRAME_DROP_RATIO', 0.1)
            self.temporal_shift_enabled = getattr(temporal_aug_config, 'TEMPORAL_SHIFT', True)
            self.speed_perturbation = getattr(temporal_aug_config, 'SPEED_PERTURBATION', 0.1)
            print(f"🎬 Temporal augmentation enabled:")
            print(f"   📉 Frame drop: {self.frame_drop_ratio*100:.1f}%")
            print(f"   ⏱️ Temporal shift: {'on' if self.temporal_shift_enabled else 'off'}")
            print(f"   🏃 Speed perturbation: ±{self.speed_perturbation*100:.1f}%")
        else:
            self.frame_drop_ratio = 0
            self.temporal_shift_enabled = False
            self.speed_perturbation = 0
        
        self.load_data()
        
        # 초기 캐싱 수행
        if self.cache_enabled:
            self._initial_cache_loading()
        
        # 백그라운드 캐싱 시작
        if self.background_cache_enabled:
            self._start_background_caching()

    def _calculate_optimal_cache_size(self):
        """
        시스템 메모리를 기반으로 최적 캐시 크기를 동적으로 계산 (큰 데이터셋용)
        """
        try:
            # 시스템 메모리 정보
            memory_info = psutil.virtual_memory()
            total_ram_gb = memory_info.total / (1024**3)
            available_ram_gb = memory_info.available / (1024**3)
            
            # 비디오 1개당 메모리 (MB)
            video_memory_mb = (self.max_frames * 3 * 224 * 224 * 4) / (1024**2)  # float32
            video_memory_gb = video_memory_mb / 1024
            
            # 매우 보수적인 캐시 크기 계산 (사용 가능한 RAM의 25% 이하)
            safe_ram_for_cache_gb = available_ram_gb * 0.25
            optimal_cache_size = int(safe_ram_for_cache_gb / video_memory_gb)
            
            # 최소/최대 제한 적용 (큰 데이터셋용 보수적 설정)
            min_cache_size = 16   # 최소 16개
            max_cache_size = 64   # 최대 64개 (매우 보수적)
            
            optimal_cache_size = max(min_cache_size, min(optimal_cache_size, max_cache_size))
            
            print(f"🔧 Dynamic cache size (large dataset):")
            print(f"   💾 Total RAM: {total_ram_gb:.1f} GB")
            print(f"   🆓 Available RAM: {available_ram_gb:.1f} GB")
            print(f"   📹 Per video: {video_memory_mb:.1f} MB")
            print(f"   🎯 Optimal cache size: {optimal_cache_size}")
            print(f"   📊 Estimated memory use: {optimal_cache_size * video_memory_gb:.1f} GB")
            
            return optimal_cache_size
            
        except Exception as e:
            logging.warning(f"Dynamic cache size calculation failed: {str(e)}; using default 30")
            return 32  # 기본값을 매우 보수적으로 설정

    def load_data(self):
        """데이터 로드 함수"""
        # 학습/검증/테스트 데이터 로드
        if self.mode == "train":
            with open(os.path.join(self.cfg.ANNO_PATH, "train.json"), "r") as f:
                self.anno_data = json.load(f)
        elif self.mode == "valid":
            with open(os.path.join(self.cfg.ANNO_PATH, "valid.json"), "r") as f:
                self.anno_data = json.load(f)
        else:  # test
            with open(os.path.join(self.cfg.ANNO_PATH, "test.json"), "r") as f:
                self.anno_data = json.load(f)
        
        # 비디오 단위로 데이터 재구성
        self.video_data = {}
        for data in self.anno_data:
            video_id = data["video_id"]
            if video_id not in self.video_data:
                self.video_data[video_id] = {
                    "class_id": data["class_id"],
                    "class_name": data["class_name"],
                    "frames_dir": data["frames_dir"],
                    "num_frames": min(data["num_frames"], self.max_frames)  # 최대 프레임 수 제한
                }
        
        # 비디오 ID 리스트 생성
        self.video_ids = list(self.video_data.keys())

    def _initial_cache_loading(self):
        """초기 캐싱 - 전체 데이터의 10%를 미리 로드 (큰 데이터셋용 메모리 안정성 확보)"""
        cache_count = int(len(self.video_ids) * self.initial_cache_ratio)
        cache_videos = random.sample(self.video_ids, cache_count)
        
        print(f"\n📦 Initial caching: {cache_count} videos ({self.mode} mode)")
        
        # 멀티프로세스를 사용한 초기 캐싱 (최소한의 워커 사용)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"InitCache-{self.mode}") as executor:
            with tqdm(total=cache_count, desc=f"🚀 {self.mode.upper()} 초기 캐싱", 
                      unit="videos", ncols=100, colour="green") as pbar:
                
                successful_loads = 0
                failed_loads = 0
                
                # 캐싱 작업 제출
                future_to_video = {
                    executor.submit(self._load_video_frames_sync, video_id): video_id 
                    for video_id in cache_videos
                }
                
                for future in as_completed(future_to_video):
                    video_id = future_to_video[future]
                    try:
                        load_time, frames = future.result()
                        
                        with self.cache_lock:
                            self.frame_cache[video_id] = frames
                            self.cache_loading_times.append(load_time)
                        
                        successful_loads += 1
                        
                        # 진행률 바 업데이트
                        pbar.set_postfix({
                            'Success': successful_loads,
                            'Failed': failed_loads,
                            'Avg Time': f"{np.mean(self.cache_loading_times):.2f}s"
                        })
                        pbar.update(1)
                        
                        # wandb에 실시간 캐싱 진행률 로깅
                        if wandb.run and self.is_train:
                            wandb.log({
                                f"cache/{self.mode}_loading_progress": successful_loads / cache_count * 100,
                                f"cache/{self.mode}_successful_loads": successful_loads,
                                f"cache/{self.mode}_avg_load_time": np.mean(self.cache_loading_times),
                                f"cache/{self.mode}_cache_size": len(self.frame_cache)
                            })
                        
                    except Exception as e:
                        failed_loads += 1
                        logging.warning(f"Caching failed for video {video_id}: {str(e)}")
                        pbar.set_postfix({
                            'Success': successful_loads,
                            'Failed': failed_loads,
                            'Avg Time': f"{np.mean(self.cache_loading_times) if self.cache_loading_times else 0:.2f}s"
                        })
                        pbar.update(1)
                        continue
        
        # 캐싱 완료 통계
        cache_efficiency = successful_loads / cache_count * 100 if cache_count > 0 else 0
        total_cache_time = time.time() - self.start_time
        
        print(f"✅ Initial caching complete")
        print(f"   📊 Succeeded: {successful_loads}/{cache_count} ({cache_efficiency:.1f}%)")
        print(f"   ⏱️  Total time: {total_cache_time:.2f}s")
        print(f"   📈 Mean load time: {np.mean(self.cache_loading_times):.2f}s/video")
        print(f"   💾 Cache size: {len(self.frame_cache)}/{self.max_cache_size}")
        
        # wandb에 최종 캐싱 통계 로깅
        if wandb.run and self.is_train:
            wandb.log({
                f"cache/{self.mode}_final_efficiency": cache_efficiency,
                f"cache/{self.mode}_total_cache_time": total_cache_time,
                f"cache/{self.mode}_final_cache_size": len(self.frame_cache),
                f"cache/{self.mode}_avg_load_time_final": np.mean(self.cache_loading_times) if self.cache_loading_times else 0
            })

    def _start_background_caching(self):
        """백그라운드 캐싱 시스템 시작"""
        if not self.background_cache_enabled:
            print(f"🔄 Background caching disabled ({self.mode} mode) - memory stability")
            return
        
        print(f"🔄 Background caching started ({self.mode} mode)")
        self.background_cache_running = True
        self.cache_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"BgCache-{self.mode}")
        
        # 백그라운드 캐싱 워커 시작
        self.cache_executor.submit(self._background_cache_worker)

    def _background_cache_worker(self):
        """백그라운드 캐싱 워커"""
        while self.background_cache_running:
            try:
                # 큐에서 캐싱 요청 가져오기 (1초 타임아웃)
                video_id = self.cache_queue.get(timeout=1.0)
                
                # 이미 캐시에 있으면 스킵
                if video_id in self.frame_cache:
                    self.cache_queue.task_done()
                    continue
                
                # 캐시가 가득 찬 경우 스킵
                with self.cache_lock:
                    if len(self.frame_cache) >= self.max_cache_size:
                        self.cache_queue.task_done()
                        continue
                
                # 비디오 로드
                try:
                    load_time, frames = self._load_video_frames_sync(video_id)
                    
                    with self.cache_lock:
                        # 다시 한 번 캐시 크기 확인
                        if len(self.frame_cache) < self.max_cache_size and video_id not in self.frame_cache:
                            self.frame_cache[video_id] = frames
                            self.background_cache_loads += 1
                            
                            # wandb 로깅
                            if wandb.run and self.is_train and self.background_cache_loads % 10 == 0:
                                wandb.log({
                                    f"cache/{self.mode}_background_loads": self.background_cache_loads,
                                    f"cache/{self.mode}_background_cache_size": len(self.frame_cache)
                                })
                    
                except Exception as e:
                    logging.warning(f"Background caching failed {video_id}: {str(e)}")
                
                self.cache_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Background caching worker error: {str(e)}")
                break

    def _load_video_frames_sync(self, video_id):
        """동기적으로 비디오 프레임을 로드하는 함수 (멀티프로세싱용)"""
        load_start = time.time()
        
        video_info = self.video_data[video_id]
        frames_dir = video_info["frames_dir"]
        num_frames = video_info["num_frames"]
        
        frames = []
        for frame_idx in range(num_frames):
            img_path = os.path.join(
                frames_dir,
                f"{video_id}_img_{frame_idx:04d}.jpg"
            )
            
            try:
                # PIL로 이미지 로드 및 크기 조정
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.img_size, Image.Resampling.BILINEAR)
                img = self.transform(img)
                frames.append(img)
            except Exception as e:
                logging.warning(f"Frame load failed {img_path}: {str(e)}")
                # 에러 시 빈 프레임으로 대체
                frames.append(torch.zeros(3, self.img_size[1], self.img_size[0]))
        
        # 프레임 수가 max_frames보다 적은 경우 패딩 추가
        if num_frames < self.max_frames:
            padding = [torch.zeros_like(frames[0]) for _ in range(self.max_frames - num_frames)]
            frames.extend(padding)
        
        # 프레임을 텐서로 변환 [L, C, H, W]
        frames_tensor = torch.stack(frames)
        load_time = time.time() - load_start
        
        return load_time, frames_tensor

    def apply_temporal_augmentation(self, frames_tensor):
        """
        시간적 데이터 증강 적용
        
        Args:
            frames_tensor: [num_frames, C, H, W] 텐서
        Returns:
            torch.Tensor: 증강된 프레임 텐서
        """
        if not self.temporal_augmentation_enabled:
            return frames_tensor
        
        num_frames = frames_tensor.size(0)
        
        # 1. 프레임 드롭아웃 (랜덤하게 일부 프레임 제거)
        if self.frame_drop_ratio > 0:
            num_drop = int(num_frames * self.frame_drop_ratio)
            if num_drop > 0:
                # 드롭할 프레임 인덱스 선택
                drop_indices = random.sample(range(num_frames), num_drop)
                keep_indices = [i for i in range(num_frames) if i not in drop_indices]
                
                if keep_indices:  # 최소 1개 프레임은 유지
                    frames_tensor = frames_tensor[keep_indices]
                    num_frames = len(keep_indices)
        
        # 2. 시간축 이동 (순서 섞기 - 작은 범위 내에서)
        if self.temporal_shift_enabled and num_frames > 10:
            # 전체를 작은 청크로 나누고 청크 내에서만 셔플
            chunk_size = min(30, num_frames // 4)  # 30프레임 또는 전체의 1/4
            if chunk_size > 5:
                chunks = []
                for i in range(0, num_frames, chunk_size):
                    chunk = frames_tensor[i:i+chunk_size]
                    
                    # 50% 확률로 청크 내 프레임 순서 섞기
                    if random.random() < 0.5:
                        indices = torch.randperm(chunk.size(0))
                        chunk = chunk[indices]
                    
                    chunks.append(chunk)
                
                frames_tensor = torch.cat(chunks, dim=0)
        
        # 3. 속도 변조 (프레임 샘플링으로 구현)
        if self.speed_perturbation > 0:
            # 속도 변화율 계산 (0.9배속 ~ 1.1배속)
            speed_factor = 1.0 + random.uniform(-self.speed_perturbation, self.speed_perturbation)
            target_frames = int(num_frames / speed_factor)
            
            if target_frames > 10 and target_frames != num_frames:
                # 새로운 인덱스 계산
                if target_frames < num_frames:
                    # 빠르게 (프레임 건너뛰기)
                    indices = torch.linspace(0, num_frames - 1, target_frames).long()
                else:
                    # 느리게 (프레임 보간/반복)
                    indices = torch.linspace(0, num_frames - 1, target_frames).long()
                    indices = torch.clamp(indices, 0, num_frames - 1)
                
                frames_tensor = frames_tensor[indices]
        
        return frames_tensor

    def normalize_frame_length(self, frames_tensor, target_length):
        """
        시간적 증강 후 프레임 수를 목표 길이로 맞춰주는 함수
        
        Args:
            frames_tensor: [num_frames, C, H, W] 텐서
            target_length: 목표 프레임 수
        Returns:
            torch.Tensor: 정규화된 프레임 텐서 [target_length, C, H, W]
        """
        current_length = frames_tensor.size(0)
        
        if current_length == target_length:
            return frames_tensor
        elif current_length > target_length:
            # 길이가 길면 균등하게 샘플링
            indices = torch.linspace(0, current_length - 1, target_length).long()
            return frames_tensor[indices]
        else:
            # 길이가 짧으면 패딩 (마지막 프레임 반복)
            padding_needed = target_length - current_length
            last_frame = frames_tensor[-1:].expand(padding_needed, -1, -1, -1)
            return torch.cat([frames_tensor, last_frame], dim=0)

    def _load_video_frames(self, video_id, use_cache=True):
        """
        비디오 프레임을 로드하는 함수
        
        🔍 캐시 메모리 관리 분석:
        ========================================
        1. 캐시 활성화 시:
           - 캐시 히트: 캐시된 데이터 반환
           - 캐시 미스: 직접 로드 후 캐시에 저장
        
        2. 캐시 비활성화 시:
           - 항상 직접 로드 (메모리 절약)
           - 캐시 저장/조회 수행하지 않음
        ========================================
        """
        # 캐시 시스템이 비활성화된 경우 직접 로드
        if not self.cache_enabled or not use_cache:
            load_time, frames_tensor = self._load_video_frames_sync(video_id)
            return frames_tensor
        
        # 캐시에서 확인 (캐시 활성화 시에만)
        if video_id in self.frame_cache:
            with self.cache_lock:
                # LRU 업데이트 - 최근 사용된 것을 맨 뒤로 이동
                frames = self.frame_cache.pop(video_id)  # 임시 제거
                self.frame_cache[video_id] = frames      # 맨 뒤에 다시 추가
                
                self.cache_hits += 1
                return frames  # 🔍 반환된 데이터는 여전히 캐시에도 존재함 (메모리 2배 점유)
        
        # 캐시 미스 기록
        self.cache_misses += 1
            
        # 백그라운드 캐싱 요청 추가 (학습 모드에서만)
        if self.background_cache_enabled and not self.cache_queue.full():
            # 인근 비디오들도 캐싱 요청에 추가
            try:
                current_idx = self.video_ids.index(video_id)
                for offset in [1, 2, 3, -1, -2]:  # 인근 5개 비디오
                    neighbor_idx = current_idx + offset
                    if 0 <= neighbor_idx < len(self.video_ids):
                        neighbor_id = self.video_ids[neighbor_idx]
                        if neighbor_id not in self.frame_cache:
                            try:
                                self.cache_queue.put_nowait(neighbor_id)
                            except queue.Full:
                                break
            except (ValueError, queue.Full):
                pass
        
        # 직접 로드 (캐시 미스인 경우)
        load_time, frames_tensor = self._load_video_frames_sync(video_id)
        
        # 캐시에 저장 (학습 모드이고 캐시 활성화시에만)
        if self.is_train and self.cache_enabled:
            with self.cache_lock:
                # 🔍 캐시 크기 제한 - 메모리 해제가 일어나는 유일한 지점
                if len(self.frame_cache) >= self.max_cache_size:
                    # ✅ 가장 오래된 항목 제거 (LRU) - 여기서 메모리 해제됨
                    removed_video_id, removed_frames = self.frame_cache.popitem(last=False)
                    
                    # 💡 메모리 해제 확인 로깅 (디버깅용)
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"🗑️ Evicted from cache: {removed_video_id} (memory freed)")
                        # del removed_frames  # 명시적 해제 (선택사항)
                
                # 새 데이터를 캐시에 추가
                self.frame_cache[video_id] = frames_tensor
                
                # 💡 캐시 추가 시 메모리 사용량 증가
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    cache_memory_mb = len(self.frame_cache) * 44  # 대략적인 추정
                    print(f"📦 Cached: {video_id}, total memory: ~{cache_memory_mb}MB")
        
        # 🔍 반환되는 데이터:
        # - 캐시 미스: 새로 로드된 데이터, 사용 후 GC 대상
        # - 캐시 저장됨: 캐시와 반환값 두 곳에 존재 (메모리 2배 점유)
        return frames_tensor

    def __len__(self):
        """데이터셋의 전체 길이를 반환"""
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        데이터셋의 idx번째 비디오를 반환하는 함수
        
        🔍 PyTorch DataLoader 메모리 관리:
        ========================================
        1. 데이터 생명주기:
           ① __getitem__ 호출 → 데이터 로드/캐시에서 가져옴
           ② DataLoader가 배치 구성 → 여러 비디오를 하나의 배치로 묶음
           ③ GPU로 전송 (.to(device)) → VRAM에 복사
           ④ 모델 forward/backward → 연산 수행
           ⚠️ ⑤ 배치 처리 완료 후 → 자동으로 메모리에서 해제되지 않음!
        
        2. 메모리 해제 시점:
           ✅ 배치 변수가 스코프를 벗어날 때 (다음 배치 로드 시)
           ✅ 명시적 del 호출 시
           ✅ Python GC가 실행될 때
           ❌ 캐시된 데이터는 계속 유지됨
        
        3. 메모리 사용 패턴:
           - 현재 배치: RAM + VRAM에 동시 존재
           - 이전 배치: GC 대상 (자동 해제)
           - 캐시 데이터: 계속 RAM에 유지
           - 총 메모리: 캐시 + 현재배치 + GPU복사본
        
        4. 대용량 데이터셋 주의사항:
           ⚠️ 캐시 + 배치 + GPU 메모리가 누적됨
           ⚠️ PIN_MEMORY=True 시 추가 메모리 사용
        ========================================
        """
        video_id = self.video_ids[idx]
        video_info = self.video_data[video_id]
        class_id = video_info["class_id"]
        class_name = video_info["class_name"]
        
        # 🔍 메모리 할당 지점: 비디오 프레임 로드 (캐싱 시스템 사용)
        # - 캐시 히트: 기존 메모리 데이터 참조 (추가 할당 없음)
        # - 캐시 미스: 새로운 44MB 텐서 할당
        frames = self._load_video_frames(video_id)
        
        # 시간적 데이터 증강 적용 (학습 시에만)
        if self.temporal_augmentation_enabled:
            frames = self.apply_temporal_augmentation(frames)
            # 시간적 증강 후 프레임 수를 원래 길이로 정규화
            frames = self.normalize_frame_length(frames, self.max_frames)
        
        # 캐시 히트율 계산 및 로깅 (25번째 아이템마다)
        if self.cache_enabled and idx % 25 == 0 and (self.cache_hits + self.cache_misses) > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            
            # wandb가 초기화되어 있고 학습 모드일 때만 로깅
            try:
                if wandb.run and self.is_train:
                    wandb.log({
                        f"cache/{self.mode}_hit_rate": cache_hit_rate,
                        f"cache/{self.mode}_hits": self.cache_hits,
                        f"cache/{self.mode}_misses": self.cache_misses,
                        f"cache/{self.mode}_current_size": len(self.frame_cache),
                        f"cache/{self.mode}_utilization": len(self.frame_cache) / max(self.max_cache_size, 1) * 100,
                        f"cache/{self.mode}_background_loads": self.background_cache_loads
                    })
            except Exception as e:
                # wandb 초기화 안됨 또는 기타 에러 무시
                pass
        
        # wandb에 이미지 로깅 (학습 시에만, 배치의 첫 번째 비디오만)
        if self.is_train and idx == 0:
            try:
                if wandb.run:  # wandb가 초기화되어 있을 때만
                    # 메모리 효율을 위해 일부 프레임만 로깅
                    sample_frames = frames[::100]  # 100프레임마다 하나씩
                    frames_np = sample_frames.permute(0, 2, 3, 1).float().numpy()
                    frames_np = (frames_np * 255).astype(np.uint8)
                    wandb.log({
                        "sample_video": wandb.Video(frames_np, format="gif"),
                        "video_id": video_id,
                        "class": class_name,
                        "cache_size": len(self.frame_cache),
                        "cache_enabled": self.cache_enabled
                    })
            except Exception as e:
                logging.warning(f"wandb logging failed: {str(e)}")
        
        # 🔍 반환되는 딕셔너리:
        # - frames: [1800, 3, 32, 64] 텐서 (~44MB)
        # - 이 데이터는 DataLoader에 의해 배치로 구성됨
        # - 배치 크기 8 → 총 ~352MB (8 × 44MB)
        # ⚠️ 중요: 반환 후에도 캐시에 원본 데이터 유지됨 (메모리 중복)
        return {
            "idx": idx,
            "video_id": video_id,
            "frames": frames,  # [L, C, H, W] 🔍 메모리 사용: ~44MB
            "label": class_id
        }

    def get_cache_info(self):
        """캐시 정보 반환"""
        if not self.cache_enabled:
            return {
                "cache_size": 0,
                "max_cache_size": 0,
                "cache_ratio": 0,
                "hit_rate": 0,
                "hits": 0,
                "misses": 0,
                "background_loads": 0,
                "utilization": 0,
                "avg_load_time": 0,
                "enabled": False
            }
        
        with self.cache_lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests * 100 if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.frame_cache),
                "max_cache_size": self.max_cache_size,
                "cache_ratio": len(self.frame_cache) / len(self.video_ids) if len(self.video_ids) > 0 else 0,
                "hit_rate": hit_rate,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "background_loads": self.background_cache_loads,
                "utilization": len(self.frame_cache) / self.max_cache_size * 100 if self.max_cache_size > 0 else 0,
                "avg_load_time": np.mean(self.cache_loading_times) if self.cache_loading_times else 0,
                "enabled": True
            }

    def print_cache_status(self):
        """캐시 상태를 콘솔에 출력"""
        info = self.get_cache_info()
        
        if not info['enabled']:
            print(f"\n📊 [{self.mode.upper()}] Cache status: 🚫 disabled - memory saving mode")
            return
        
        print(f"\n📊 [{self.mode.upper()}] Cache status:")
        print(f"   💾 Size: {info['cache_size']}/{info['max_cache_size']} ({info['utilization']:.1f}%)")
        print(f"   🎯 Hit rate: {info['hit_rate']:.1f}% ({info['hits']}H/{info['misses']}M)")
        print(f"   🔄 Background loads: {info['background_loads']}")
        print(f"   ⚡ Mean load: {info['avg_load_time']:.2f}s")
        print(f"   📈 Data coverage: {info['cache_ratio']:.1%}")

    def visualize_cache_performance(self):
        """캐시 성능을 시각화"""
        if not self.cache_loading_times:
            print(f"⚠️  {self.mode.upper()} mode: no cache loading stats; skipping visualization.")
            return
        
        info = self.get_cache_info()
        
        # 캐시 통계가 없는 경우 건너뛰기
        if info['hits'] == 0 and info['misses'] == 0:
            print(f"⚠️  {self.mode.upper()} mode: no cache usage stats; skipping visualization.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{self.mode.upper()} Dataset - Cache Performance', fontsize=14)
        
        # 1. 로딩 시간 히스토그램
        ax1.hist(self.cache_loading_times, bins=20, alpha=0.7, color='skyblue')
        ax1.set_title('Cache Loading Time Distribution')
        ax1.set_xlabel('Loading Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(self.cache_loading_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.cache_loading_times):.2f}s')
        ax1.legend()
        
        # 2. 캐시 히트율 (캐시 사용 통계가 있는 경우만)
        if info['hits'] > 0 or info['misses'] > 0:
            labels = ['Cache Hits', 'Cache Misses']
            sizes = [max(info['hits'], 1), max(info['misses'], 1)]  # 최소값 1로 설정하여 NaN 방지
            colors = ['lightgreen', 'lightcoral']
            
            # 실제 히트율이 0%인 경우 특별 처리
            if info['hits'] == 0:
                sizes = [1, 99]  # 시각적으로 0%를 표현
                labels = ['Cache Hits (0%)', 'Cache Misses (100%)']
            elif info['misses'] == 0:
                sizes = [99, 1]  # 시각적으로 100%를 표현
                labels = ['Cache Hits (100%)', 'Cache Misses (0%)']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Cache Hit Rate: {info["hit_rate"]:.1f}%')
        else:
            ax2.text(0.5, 0.5, 'No Cache Usage Yet', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Cache Hit Rate: N/A')
        
        # 3. 캐시 사용률
        ax3.bar(['Used', 'Available'], 
               [info['cache_size'], info['max_cache_size'] - info['cache_size']],
               color=['orange', 'lightgray'])
        ax3.set_title(f'Cache Utilization: {info["utilization"]:.1f}%')
        ax3.set_ylabel('Number of Videos')
        
        # 4. 로딩 시간 추이
        ax4.plot(range(len(self.cache_loading_times)), self.cache_loading_times, 
                alpha=0.7, color='purple')
        ax4.set_title('Loading Time Trend')
        ax4.set_xlabel('Cache Loading Order')
        ax4.set_ylabel('Loading Time (seconds)')
        
        plt.tight_layout()
        
        # wandb에 이미지 로깅
        try:
            if wandb.run and self.is_train:
                wandb.log({f"cache/{self.mode}_performance_plot": wandb.Image(fig)})
        except Exception as e:
            logging.warning(f"wandb image logging failed: {str(e)}")
        
        plt.close(fig)
        print(f"📊 {self.mode.upper()} cache performance visualization complete")

    def cleanup(self):
        """리소스 정리"""
        if self.background_cache_enabled and self.cache_executor:
            print(f"🔄 Background caching stopped ({self.mode} mode)")
            self.background_cache_running = False
            
            # 큐에 남은 작업들 완료 대기
            try:
                self.cache_queue.join()
            except:
                pass
            
            # Executor 종료
            self.cache_executor.shutdown(wait=True)
            
    def __del__(self):
        """소멸자에서 리소스 정리"""
        try:
            self.cleanup()
        except:
            pass

    def cleanup_cache_memory(self, target_utilization=0.7):
        """
        캐시 메모리 정리 함수
        
        Args:
            target_utilization: 목표 캐시 사용률 (0.7 = 70%)
        
        🔍 메모리 정리 전략:
        ========================================
        1. 현재 캐시 사용률이 목표치를 초과하는 경우에만 실행
        2. LRU 방식으로 오래된 데이터부터 제거
        3. 목표 사용률에 도달할 때까지 반복
        4. 명시적 가비지 컬렉션 수행
        ========================================
        """
        if not self.cache_enabled:
            return
        
        with self.cache_lock:
            current_size = len(self.frame_cache)
            target_size = int(self.max_cache_size * target_utilization)
            
            if current_size <= target_size:
                return  # 정리할 필요 없음
            
            # 제거할 항목 수 계산
            items_to_remove = current_size - target_size
            removed_items = []
            
            print(f"🧹 Cache memory cleanup:")
            print(f"   Current: {current_size}/{self.max_cache_size} ({current_size/self.max_cache_size*100:.1f}%)")
            print(f"   Target: {target_size}/{self.max_cache_size} ({target_utilization*100:.1f}%)")
            print(f"   Removing: {items_to_remove} items")
            
            # LRU 방식으로 제거 (가장 오래된 것부터)
            for _ in range(items_to_remove):
                if self.frame_cache:
                    removed_video_id, removed_frames = self.frame_cache.popitem(last=False)
                    removed_items.append(removed_video_id)
                    del removed_frames  # 명시적 해제
            
            print(f"✅ Cache cleanup complete: {len(removed_items)} items removed")
            
            # 가비지 컬렉션 수행
            import gc
            gc.collect()
    
    def get_memory_usage_mb(self):
        """
        현재 캐시의 메모리 사용량 계산 (MB)
        
        Returns:
            float: 메모리 사용량 (MB)
        """
        if not self.frame_cache:
            return 0.0
        
        # 비디오 1개당 메모리 계산
        video_memory_mb = (self.max_frames * 3 * 224 * 224 * 4) / (1024**2)  # float32
        total_memory_mb = len(self.frame_cache) * video_memory_mb
        
        return total_memory_mb
    
    def force_cache_reset(self):
        """
        캐시 강제 초기화 함수 (메모리 부족 시 긴급 사용)
        
        🔍 긴급 메모리 해제:
        ========================================
        1. 모든 캐시 데이터 제거
        2. 통계 초기화
        3. 명시적 가비지 컬렉션
        4. GPU 캐시도 함께 정리
        ========================================
        """
        if not self.cache_enabled:
            return
        
        with self.cache_lock:
            cache_size_before = len(self.frame_cache)
            memory_before_mb = self.get_memory_usage_mb()
            
            # 모든 캐시 데이터 제거
            for video_id, frames in self.frame_cache.items():
                del frames
            
            self.frame_cache.clear()
            
            # 통계 초기화 (선택사항)
            # self.cache_hits = 0
            # self.cache_misses = 0
            
            print(f"🚨 Cache force-cleared:")
            print(f"   Items removed: {cache_size_before}")
            print(f"   Memory freed: {memory_before_mb:.1f}MB")
            
            # 가비지 컬렉션 수행
            import gc
            gc.collect()
            
            # GPU 캐시도 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"   GPU cache cleared as well")
    
    def get_detailed_memory_info(self):
        """
        상세한 메모리 정보 반환
        
        Returns:
            dict: 메모리 사용량 상세 정보
        """
        cache_memory_mb = self.get_memory_usage_mb()
        cache_memory_gb = cache_memory_mb / 1024
        
        # 시스템 메모리 정보
        memory_info = psutil.virtual_memory()
        total_ram_gb = memory_info.total / (1024**3)
        used_ram_gb = (memory_info.total - memory_info.available) / (1024**3)
        cache_ram_percent = (cache_memory_gb / total_ram_gb) * 100
        
        return {
            "cache_size": len(self.frame_cache),
            "max_cache_size": self.max_cache_size,
            "cache_memory_mb": cache_memory_mb,
            "cache_memory_gb": cache_memory_gb,
            "cache_utilization": len(self.frame_cache) / self.max_cache_size * 100,
            "total_ram_gb": total_ram_gb,
            "used_ram_gb": used_ram_gb,
            "cache_ram_percent": cache_ram_percent,
            "estimated_peak_memory_gb": cache_memory_gb * 2,  # 캐시 + 배치 데이터
        }
    
    def print_detailed_memory_status(self):
        """
        상세한 메모리 상태 출력
        """
        info = self.get_detailed_memory_info()
        
        print(f"\n📊 [{self.mode.upper()}] Detailed memory status:")
        print(f"   💾 Cache size: {info['cache_size']}/{info['max_cache_size']} ({info['cache_utilization']:.1f}%)")
        print(f"   🔢 Cache memory: {info['cache_memory_mb']:.1f}MB ({info['cache_memory_gb']:.2f}GB)")
        print(f"   📈 RAM share: {info['cache_ram_percent']:.1f}% ({info['cache_memory_gb']:.1f}GB / {info['total_ram_gb']:.1f}GB)")
        print(f"   ⚠️  Estimated peak: {info['estimated_peak_memory_gb']:.1f}GB (cache + batch)")
        
        # 경고 메시지
        if info['cache_ram_percent'] > 50:
            print(f"   🚨 Warning: cache exceeds 50% of RAM")
        elif info['cache_ram_percent'] > 30:
            print(f"   🔶 Note: cache is using 30% of RAM")
        else:
            print(f"   ✅ Safe: cache memory use is within range")
